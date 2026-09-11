"""
Shared content type/category detection, used when indexing scraped
content into the knowledge base.

This logic previously only existed inside agents/tools.py's
scrape_college_website tool - which is the *live* scraping tool that was
deliberately removed from the chat agent's active tool list (it was
accidentally firing mid-conversation and causing timeouts; see
agents/unibot.py). Because of that, the actual production ingestion
pipeline (index_bmsit.py) never ran this code, so no document in the real
knowledge base has ever had a 'category' field - meaning every
filter_category=... search in agents/tools.py's RAG queries silently
falls back to a generic search instead of actually filtering.

Factored out here so both the production ingestion path and the (unused
but still-present) live-scrape tool apply identical rules.
"""
from typing import Optional, Tuple


def detect_content_category(url: str, content_type: str = "") -> Tuple[str, Optional[str]]:
    """
    Detect a (type, category) pair for a scraped item from its URL.

    Args:
        url: the source URL of the scraped page/PDF
        content_type: the content's existing 'type' value, if any
            (e.g. 'pdf', 'scraped_content') - used as a fallback/hint

    Returns:
        (type, category) - category may be None if nothing matched, in
        which case the caller should simply omit the 'category' key
        rather than writing a null/placeholder value.
    """
    url_lower = url.lower()

    if "syllabus" in url_lower or "syllabi" in url_lower:
        return "syllabus", "academic"

    if url_lower.endswith(".pdf") or content_type == "pdf":
        if "course" in url_lower or "curriculum" in url_lower:
            return "pdf", "academic"
        return "pdf", "document"

    if "course" in url_lower or "curriculum" in url_lower:
        return "course_info", "academic"

    if "admission" in url_lower:
        return "admission_info", "admission"

    if "faculty" in url_lower or "staff" in url_lower:
        return "faculty_info", "academic"

    if "placement" in url_lower:
        return content_type or "scraped_content", "placement"

    if "facility" in url_lower or "infrastructure" in url_lower:
        return content_type or "scraped_content", "facilities"

    return content_type or "scraped_content", None
