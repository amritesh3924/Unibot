from dotenv import load_dotenv
import os
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import Optional
import sys
from pathlib import Path
import time

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from retrieval.rag_system import RAGSystem
from ingestion.scraper import CollegeScraper
from ingestion.categorize import detect_content_category


load_dotenv(override=True)


class QueryCollegeKnowledgeBaseInput(BaseModel):
    query: str = Field(
        description="The question or query about the college to search in the knowledge base."
    )
    content_type: Optional[str] = Field(
        default=None,
        description="Optional filter for specific content type (e.g., 'syllabus', 'pdf', 'course_info')."
    )


class ScrapeCollegeWebsiteInput(BaseModel):
    base_url: str = Field(
        description="The base URL of the college website to scrape."
    )


def get_rag_tools(rag_system: RAGSystem):
    """Get RAG-related tools for querying college information"""

    def query_college_knowledge_base(
        query: str,
        content_type: str = None
    ) -> str:
        """
        Search the college knowledge base for information related
        to the query.

        Use this tool when the user asks questions about the college,
        courses, admissions, faculty, facilities, or any college-related
        information.

        Args:
            query: The question or query about the college
            content_type: Optional filter for specific content type
                (e.g., "syllabus", "pdf", "course_info")
        """

        try:
            query_lower = query.lower()

            # ---------------------------------------------------------
            # 1. Detect content type / category
            # ---------------------------------------------------------
            detected_type = None
            detected_category = None

            if (
                "syllabus" in query_lower
                or "syllabi" in query_lower
            ):
                detected_type = "syllabus"
                detected_category = "academic"

            elif (
                "course" in query_lower
                and (
                    "structure" in query_lower
                    or "curriculum" in query_lower
                )
            ):
                detected_type = "course_info"
                detected_category = "academic"

            elif "admission" in query_lower:
                detected_category = "admission"

            elif (
                "faculty" in query_lower
                or "professor" in query_lower
                or "staff" in query_lower
                or "teacher" in query_lower
            ):
                detected_category = "academic"

            # NOTE: HOD/"head of department" queries deliberately do NOT
            # set detected_category here. HOD names on BMSIT's site are
            # commonly published in department newsletters, which
            # ingestion/categorize.py tags as "document", not "academic" -
            # confirmed directly against real indexed content. Filtering
            # this primary search to "academic" would exclude exactly the
            # chunks it needs to find, and the fallback-on-empty-results
            # below doesn't reliably catch this: a wrong category filter
            # often still returns *some* (just irrelevant) results, so it
            # never triggers. The dedicated HOD-targeted search further
            # below already handles this case correctly, unfiltered.

            elif "placement" in query_lower:
                detected_category = "placement"

            elif (
                "facility" in query_lower
                or "infrastructure" in query_lower
            ):
                detected_category = "facilities"

            search_type = content_type or detected_type

            # ---------------------------------------------------------
            # 2. Primary retrieval
            # ---------------------------------------------------------
            rag_start = time.perf_counter()

            docs = rag_system.search(
                query,
                k=4,
                filter_type=search_type,
                filter_category=detected_category
            )

            rag_elapsed = time.perf_counter() - rag_start

            print(
                f"[TIMING] RAG search: "
                f"{rag_elapsed:.3f}s | results={len(docs)}"
            )

            # ---------------------------------------------------------
            # 3. Fallback when filters return nothing
            # ---------------------------------------------------------
            if not docs and (
                search_type or detected_category
            ):
                fallback_start = time.perf_counter()

                docs = rag_system.search(
                    query,
                    k=4
                )

                fallback_elapsed = (
                    time.perf_counter() - fallback_start
                )

                print(
                    f"[TIMING] RAG fallback: "
                    f"{fallback_elapsed:.3f}s | "
                    f"results={len(docs)}"
                )

            # ---------------------------------------------------------
            # 4. Targeted retrieval for HOD queries
            #
            # This is important because an initial semantic search
            # ---------------------------------------------------------
            # 4. Targeted retrieval for HOD and Leadership queries
            #
            # Always prioritize current official governance/Academic Council
            # documents and exclude outdated student newsletters (2018-2023)
            # which contain obsolete former HOD names.
            # ---------------------------------------------------------
            is_hod_query = any(
                term in query_lower
                for term in [
                    "hod",
                    "head of department",
                    "department head",
                    "head of the department",
                    "principal",
                    "vice principal",
                    "dean"
                ]
            )

            if is_hod_query:
                hod_query = (
                    f"{query} "
                    "Head of Department HOD Academic Council Approved leadership"
                )

                hod_start = time.perf_counter()

                hod_docs = rag_system.search(
                    hod_query,
                    k=8
                )

                hod_elapsed = time.perf_counter() - hod_start

                print(
                    f"[TIMING] HOD targeted RAG: "
                    f"{hod_elapsed:.3f}s | "
                    f"results={len(hod_docs)}"
                )

                # Filter out historical newsletters and student magazines
                def is_outdated_source(d):
                    src = d.metadata.get("source", "").lower()
                    return "newsletter" in src or "news-letter" in src

                candidate_docs = [
                    d for d in hod_docs + docs
                    if not is_outdated_source(d)
                ]

                # If all candidates were filtered, fall back to unfiltered
                if not candidate_docs:
                    candidate_docs = hod_docs + docs

                # Score and rank: prioritize official Academic Council and department pages
                def hod_rank_score(d):
                    src = d.metadata.get("source", "").lower()
                    content = d.page_content.lower()
                    score = 0
                    if "academic" in src and "council" in src:
                        score += 50
                    if "academic council" in content:
                        score += 30
                    if "dep-" in src or "faculty" in src:
                        score += 20
                    if "hod" in content or "head of department" in content:
                        score += 10
                    return score

                # Deduplicate by (source, content)
                seen_chunks = set()
                deduped_candidates = []
                for d in candidate_docs:
                    key = (d.metadata.get("source", "Unknown"), d.page_content.strip())
                    if key not in seen_chunks:
                        seen_chunks.add(key)
                        deduped_candidates.append(d)

                deduped_candidates.sort(key=hod_rank_score, reverse=True)
                docs = deduped_candidates[:4]

                print(
                    f"[TIMING] HOD clean results: "
                    f"{len(docs)} (prioritized official leadership sources)"
                )

            # ---------------------------------------------------------
            # 5. General query expansion only when nothing was found
            # ---------------------------------------------------------
            if not docs:
                expanded_query = query

                if (
                    "syllabus" in query_lower
                    or "syllabi" in query_lower
                ):
                    expanded_query = (
                        query + " course curriculum"
                    )

                elif (
                    "faculty" in query_lower
                    or "professor" in query_lower
                    or "teacher" in query_lower
                ):
                    expanded_query = (
                        query + " staff department"
                    )

                elif "admission" in query_lower:
                    expanded_query = (
                        query + " requirements eligibility"
                    )

                if expanded_query != query:
                    expansion_start = time.perf_counter()

                    docs = rag_system.search(
                        expanded_query,
                        k=4
                    )

                    expansion_elapsed = (
                        time.perf_counter() - expansion_start
                    )

                    print(
                        f"[TIMING] Expanded RAG: "
                        f"{expansion_elapsed:.3f}s | "
                        f"results={len(docs)}"
                    )

            # ---------------------------------------------------------
            # 6. Build response context
            # ---------------------------------------------------------
            if docs:
                context_parts = []

                # Deduplicate URLs while preserving order
                sources_list = []
                seen_sources = set()

                for i, doc in enumerate(docs, 1):
                    source = doc.metadata.get(
                        "source",
                        "Unknown"
                    )
                    title = doc.metadata.get(
                        "title",
                        ""
                    )
                    doc_type = doc.metadata.get(
                        "type",
                        "document"
                    )
                    content = doc.page_content[:1200]

                    # Source link
                    if source != "Unknown":
                        source_link = (
                            f"[{source}]({source})"
                        )
                    else:
                        source_link = (
                            f"Source {i}: {source}"
                        )

                    context_parts.append(
                        f"Source {i}: {source_link} | "
                        f"Type: {doc_type}\n"
                        f"{title}\n"
                        f"{content}\n"
                    )

                    # Add source only once
                    if (
                        source != "Unknown"
                        and source not in seen_sources
                    ):
                        seen_sources.add(source)
                        sources_list.append(source)

                # -----------------------------------------------------
                # 7. Sources section
                # -----------------------------------------------------
                sources_section = "\n\n**SOURCES:**\n"

                for i, src in enumerate(
                    sources_list,
                    1
                ):
                    sources_section += (
                        f"{i}. [{src}]({src})\n"
                    )

                return (
                    "Relevant information from college "
                    "knowledge base:\n\n"
                    + "\n---\n".join(context_parts)
                    + sources_section
                )

            return (
                "No relevant information found "
                "in the knowledge base."
            )

        except Exception as e:
            return (
                f"Error querying knowledge base: {str(e)}"
            )

    # -------------------------------------------------------------
    # Website scraper tool
    # -------------------------------------------------------------
    # Note:
    # scrape_college_website needs to be synchronous for LangChain
    # tools, so we create a synchronous wrapper around the async
    # scraper.
    # -------------------------------------------------------------
    import asyncio
    import nest_asyncio

    def scrape_college_website_sync(base_url: str) -> str:
        """Synchronous wrapper for scraping"""

        try:
            # Allow nested event loops
            try:
                nest_asyncio.apply()
            except Exception:
                pass

            async def run_scrape():
                scraper = CollegeScraper(
                    base_url=base_url,
                    max_pages=500,
                    max_depth=8,
                    include_pdfs=True
                )

                return await scraper.scrape_website()

            try:
                loop = asyncio.get_event_loop()

                if loop.is_running():
                    # If loop is already running,
                    # execute scraping in a separate thread.
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            lambda: asyncio.run(
                                run_scrape()
                            )
                        )

                        scraped_content = future.result(
                            timeout=300
                        )
                else:
                    scraped_content = asyncio.run(
                        run_scrape()
                    )

            except RuntimeError:
                scraped_content = asyncio.run(
                    run_scrape()
                )

            if scraped_content:
                texts = []
                metadatas = []
                successful = 0
                failed = 0

                for content in scraped_content:
                    try:
                        url = content.get(
                            "url",
                            "Unknown"
                        )
                        text = content.get(
                            "content",
                            ""
                        )
                        title = content.get(
                            "title",
                            ""
                        )
                        content_type = content.get(
                            "type",
                            "scraped_content"
                        )

                        if text and len(text) > 50:
                            texts.append(text)

                            metadata = {
                                "source": url,
                                "type": content_type,
                                "title": title
                            }

                            # Auto-detect content type/category from the
                            # URL - shared with the production ingestion
                            # pipeline (ingestion/categorize.py) so both
                            # paths stay consistent.
                            detected_type, detected_category = detect_content_category(
                                url, content_type
                            )
                            metadata["type"] = detected_type
                            if detected_category:
                                metadata["category"] = detected_category

                            metadatas.append(metadata)
                            successful += 1

                        else:
                            failed += 1

                    except Exception as e:
                        print(
                            f"Error processing content: {e}"
                        )
                        failed += 1

                # Add documents with duplicate checking
                if texts:
                    try:
                        print(
                            f"\n📊 Adding {len(texts)} items "
                            f"to knowledge base "
                            f"(checking for duplicates)..."
                        )

                        rag_system.add_documents(
                            texts,
                            metadatas,
                            skip_duplicates=True
                        )

                        return (
                            f"Successfully scraped "
                            f"{len(scraped_content)} pages "
                            f"from {base_url}. "
                            f"Added {successful} pages "
                            f"to knowledge base "
                            f"({failed} skipped)."
                        )

                    except Exception as e:
                        return (
                            f"Scraped {len(scraped_content)} "
                            f"pages but error adding to "
                            f"knowledge base: {str(e)}"
                        )

                else:
                    return (
                        f"Scraped {len(scraped_content)} "
                        f"pages but no valid content "
                        f"to add."
                    )

            else:
                return (
                    f"No content was scraped from "
                    f"{base_url}. Please check the URL "
                    f"and try again."
                )

        except Exception as e:
            return (
                f"Error scraping website: {str(e)}"
            )

    # -------------------------------------------------------------
    # RAG tool
    # -------------------------------------------------------------
    rag_query_tool = StructuredTool.from_function(
        func=query_college_knowledge_base,
        name="query_college_knowledge_base",
        description=(
            "MANDATORY: Search the college knowledge base "
            "for information. You MUST use this tool FIRST "
            "for ANY question about the college, courses, "
            "admissions, faculty, facilities, policies, "
            "events, or any other college-related information. "
            "Do not answer college-related questions without "
            "using this tool first. The tool returns relevant "
            "information from the knowledge base along with "
            "source URLs."
        ),
        args_schema=QueryCollegeKnowledgeBaseInput
    )

    # -------------------------------------------------------------
    # Scraper tool
    # -------------------------------------------------------------
    scrape_tool = StructuredTool.from_function(
        func=scrape_college_website_sync,
        name="scrape_college_website",
        description=(
            "Scrape the college website and add content to "
            "the knowledge base. Use this when you need to "
            "gather information from the college website. "
            "Provide the base URL of the college website."
        ),
        args_schema=ScrapeCollegeWebsiteInput
    )

    return [
        rag_query_tool,
        scrape_tool
    ]