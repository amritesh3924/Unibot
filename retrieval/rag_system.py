"""
RAG (Retrieval Augmented Generation) system for UniBot
Handles document storage, embedding, and retrieval for college information
"""
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Optional
import os
import time
import hashlib
from dotenv import load_dotenv

load_dotenv(override=True)

class RAGSystem:
    def __init__(self, persist_directory: str = "./chroma_db_fastembed", collection_name: str = "college_data"):
        """
        Initialize the RAG system with vector store
        
        Args:
            persist_directory: Directory to persist the vector database
            collection_name: Name of the collection in Chroma
        """
        # Verify API key is set
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is not set. "
                "Please set it in your .env file or environment variables."
            )
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        # Local embeddings via FastEmbed (BAAI/bge-small-en-v1.5, 384-dim).
        # No Gemini embedding calls / no quota dependency for embedding.
        #
        # IMPORTANT: explicitly pin cache_dir. FastEmbedEmbeddings defaults
        # to a cache under the OS temp dir (e.g. /tmp/fastembed_cache),
        # which most hosting platforms wipe on every container restart -
        # that would silently trigger a ~130MB re-download from
        # HuggingFace on every cold start (slow, and a hard failure if the
        # platform blocks outbound HF access at runtime). Pointing this at
        # a path inside the app directory lets the Docker image bake the
        # model in at build time, so runtime never needs network access
        # for embeddings.
        fastembed_cache_dir = os.getenv("FASTEMBED_CACHE_DIR", "./fastembed_cache")
        self.embeddings = FastEmbedEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            cache_dir=fastembed_cache_dir
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.vectorstore = None
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """Initialize or load the existing vector store"""
        if os.path.exists(self.persist_directory):
            # Load existing vector store
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
        else:
            # Create new vector store
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
    
    def get_existing_sources(self) -> set:
        """Get set of all source URLs already in the knowledge base"""
        try:
            collection = self.vectorstore._collection
            if collection:
                results = collection.get(limit=10000)  # Get up to 10k documents
                if results and 'metadatas' in results:
                    sources = set()
                    for metadata in results['metadatas']:
                        if metadata and 'source' in metadata:
                            sources.add(metadata['source'])
                    return sources
        except Exception as e:
            print(f"⚠️  Could not check existing sources: {e}")
        return set()
    
    def add_documents(self, texts: List[str], metadatas: Optional[List[dict]] = None, skip_duplicates: bool = True):
        """
        Add documents to the vector store with chunk-level duplicate checking
        and rate-limit-safe batching.
        """

        if not texts:
            return

        if metadatas is None:
            metadatas = [{}] * len(texts)

        if len(metadatas) != len(texts):
            metadatas = metadatas[:len(texts)] + [{}] * (len(texts) - len(metadatas))

        # Create Document objects
        documents = []

        for text, meta in zip(texts, metadatas):
            if text and len(text.strip()) > 0:
                if "source" not in meta:
                    meta["source"] = "Unknown"

                documents.append(
                    Document(
                        page_content=text,
                        metadata=meta
                    )
                )

        if not documents:
            print("Warning: No valid documents to add after filtering")
            return

        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)

        if not chunks:
            print("Warning: No chunks created from documents")
            return

        # ---------------------------------------------------------
        # CHUNK-LEVEL DUPLICATE CHECK
        # ---------------------------------------------------------
        existing_hashes = set()

        if skip_duplicates:
            try:
                collection = self.vectorstore._collection

                results = collection.get(
                    limit=10000,
                    include=["documents", "metadatas"]
                )

                existing_documents = results.get("documents") or []
                existing_metadatas = results.get("metadatas") or []

                for doc_content, metadata in zip(
                    existing_documents,
                    existing_metadatas
                ):
                    if not doc_content:
                        continue

                    metadata = metadata or {}
                    source = metadata.get("source", "Unknown")

                    chunk_hash = metadata.get("chunk_hash")

                    if not chunk_hash:
                        chunk_hash = hashlib.sha256(
                            f"{source}\n{doc_content}".encode("utf-8")
                        ).hexdigest()

                    existing_hashes.add(chunk_hash)

            except Exception as e:
                print(f"⚠️ Could not check existing chunks: {e}")

        # Remove chunks already present in ChromaDB
        new_chunks = []
        skipped_count = 0

        for chunk in chunks:
            source = chunk.metadata.get("source", "Unknown")

            chunk_hash = hashlib.sha256(
                f"{source}\n{chunk.page_content}".encode("utf-8")
            ).hexdigest()

            chunk.metadata["chunk_hash"] = chunk_hash

            if skip_duplicates and chunk_hash in existing_hashes:
                skipped_count += 1
                continue

            new_chunks.append(chunk)

        if skipped_count > 0:
            print(
                f"  ⏭️ Skipped {skipped_count} chunks already in knowledge base"
            )

        if not new_chunks:
            print("  ℹ️ All chunks were already indexed")
            return

        chunks = new_chunks

        # ---------------------------------------------------------
        # RATE-LIMIT-SAFE BATCHING
        # ---------------------------------------------------------
        batch_size = 10
        total_chunks = len(chunks)
        added_count = 0

        try:
            for i in range(0, total_chunks, batch_size):
                batch = chunks[i:i + batch_size]

                while True:
                    try:
                        self.vectorstore.add_documents(batch)

                        added_count += len(batch)

                        print(
                            f"  [BATCH] Added batch {i//batch_size + 1}: "
                            f"{added_count}/{total_chunks} chunks",
                            flush=True
                        )

                        break

                    except Exception as e:
                        error_text = str(e)

                        if (
                            "429" in error_text
                            or "RESOURCE_EXHAUSTED" in error_text
                        ):
                            print(
                                "  ⏳ Gemini quota reached. "
                                "Waiting 65 seconds before retrying...",
                                flush=True
                            )

                            time.sleep(65)

                        else:
                            raise

                # Keep us safely below the 100 requests/minute limit.
                if i + batch_size < total_chunks:
                    time.sleep(30)

            print(
                f"[OK] Added {added_count} new chunks to knowledge base"
            )

        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise
    
    def add_scraped_content(self, url: str, content: str, title: Optional[str] = None, content_type: str = "scraped_content"):
        """
        Add scraped content from a URL to the vector store
        
        Args:
            url: Source URL of the content
            content: Text content scraped from the URL
            title: Optional title of the page
            content_type: Type of content (e.g., "scraped_content", "pdf", "syllabus", "course_info")
        """
        metadata = {"source": url, "type": content_type}
        if title:
            metadata["title"] = title
        
        # Detect content type from URL if not specified
        if content_type == "scraped_content":
            url_lower = url.lower()
            if "syllabus" in url_lower or "syllabi" in url_lower:
                metadata["type"] = "syllabus"
            elif "pdf" in url_lower or url_lower.endswith(".pdf"):
                metadata["type"] = "pdf"
            elif "course" in url_lower or "curriculum" in url_lower:
                metadata["type"] = "course_info"
            elif "admission" in url_lower:
                metadata["type"] = "admission_info"
            elif "faculty" in url_lower or "staff" in url_lower:
                metadata["type"] = "faculty_info"
        
        self.add_documents([content], [metadata])
    
    def search(self, query: str, k: int = 4, filter_type: Optional[str] = None, filter_category: Optional[str] = None, use_hybrid: bool = True) -> List[Document]:
        """
        Search for relevant documents using semantic similarity or hybrid search
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            filter_type: Optional filter by content type (e.g., "syllabus", "pdf", "course_info")
            filter_category: Optional filter by category (e.g., "academic", "admission", "facilities")
            use_hybrid: If True, use hybrid search (semantic + keyword), else semantic only
            
        Returns:
            List of relevant Document objects
        """
        if self.vectorstore is None:
            return []
        
        # Build filter if needed
        where_filter = None
        if filter_type:
            where_filter = {"type": filter_type}
        elif filter_category:
            where_filter = {"category": filter_category}
        
        if use_hybrid:
            # Hybrid search: combine semantic similarity with keyword matching
            try:
                # Get semantic results
                semantic_docs = self.vectorstore.similarity_search(query, k=k*2, filter=where_filter) if where_filter else self.vectorstore.similarity_search(query, k=k*2)
                
                # Get keyword-based results (using BM25-like approach via metadata search)
                # Extract keywords from query
                query_keywords = set(query.lower().split())
                
                # Score documents based on keyword matches in content and metadata
                scored_docs = []
                all_docs = self.vectorstore.similarity_search(query, k=min(k*3, 50))  # Get more candidates
                
                for doc in all_docs:
                    score = 0.0
                    content_lower = doc.page_content.lower()
                    metadata_str = str(doc.metadata).lower()
                    
                    # Count keyword matches
                    for keyword in query_keywords:
                        if len(keyword) > 2:  # Only meaningful keywords
                            score += content_lower.count(keyword) * 0.1
                            score += metadata_str.count(keyword) * 0.2
                    
                    scored_docs.append((doc, score))
                
                # Sort by score and combine with semantic results
                scored_docs.sort(key=lambda x: x[1], reverse=True)
                keyword_docs = [doc for doc, score in scored_docs[:k] if score > 0]
                
                # Combine semantic and keyword results, removing duplicates
                combined = {}
                for doc in semantic_docs[:k]:
                    doc_id = id(doc)  # Use object id as key
                    combined[doc_id] = doc
                
                for doc in keyword_docs:
                    doc_id = id(doc)
                    if doc_id not in combined:
                        combined[doc_id] = doc
                
                result = list(combined.values())[:k]
                
                # Apply filter if needed
                if where_filter:
                    filtered_result = []
                    for doc in result:
                        if filter_type and doc.metadata.get("type") == filter_type:
                            filtered_result.append(doc)
                        elif filter_category and doc.metadata.get("category") == filter_category:
                            filtered_result.append(doc)
                    return filtered_result[:k] if filtered_result else result[:k]
                
                return result[:k]
            except Exception as e:
                # Fall back to semantic search if hybrid fails
                print(f"Hybrid search failed, using semantic only: {e}")
                if where_filter:
                    try:
                        return self.vectorstore.similarity_search(query, k=k, filter=where_filter)
                    except:
                        return self.vectorstore.similarity_search(query, k=k)
                else:
                    return self.vectorstore.similarity_search(query, k=k)
        else:
            # Semantic search only
            if where_filter:
                try:
                    return self.vectorstore.similarity_search(query, k=k, filter=where_filter)
                except:
                    return self.vectorstore.similarity_search(query, k=k)
            else:
                return self.vectorstore.similarity_search(query, k=k)
    
    def search_with_scores(self, query: str, k: int = 4):
        """
        Search with similarity scores
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of tuples (Document, score)
        """
        if self.vectorstore is None:
            return []
        
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def get_relevant_context(self, query: str, k: int = 4) -> str:
        """
        Get formatted context string from relevant documents
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            Formatted string with relevant context
        """
        docs = self.search(query, k=k)
        
        if not docs:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            title = doc.metadata.get("title", "")
            content = doc.page_content
            
            context_parts.append(f"[Source {i}: {source}]\n{title}\n{content}\n")
        
        return "\n---\n".join(context_parts)
    
    def clear_database(self):
        """Clear all documents from the vector store"""
        if os.path.exists(self.persist_directory):
            import shutil
            shutil.rmtree(self.persist_directory)
        self._initialize_vectorstore()
    
    def get_stats(self) -> dict:
        """Get statistics about the vector store"""
        if self.vectorstore is None:
            return {"total_documents": 0}
        
        # Get collection count
        collection = self.vectorstore._collection
        count = collection.count() if collection else 0
        
        return {
            "total_documents": count,
            "persist_directory": self.persist_directory,
            "collection_name": self.collection_name
        }