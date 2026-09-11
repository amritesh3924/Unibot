"""
One-time, OPTIONAL migration for scrape_tracker.json.

Why this exists:
-----------------
The old tracker format was a flat list of "scraped" URLs with no success/
failure distinction - a URL landed in that list whether extraction
actually succeeded or silently failed. ingestion/scrape_tracker.py's
automatic fallback already handles this safely with zero manual steps:
on load, it treats every legacy entry as "failed/retryable" rather than
trusting it as a real success. That's correct, but not maximally
efficient - it means the ~133 PDFs that genuinely succeeded before will
also get re-downloaded and re-processed on your next scrape run (wasted
time, though harmless: the indexing step's skip_duplicates already
prevents them from being added to the DB twice).

What this script does instead:
-------------------------------
Cross-references your existing tracker against what's ACTUALLY indexed in
chroma_db_fastembed right now (ground truth). Any tracked URL that
genuinely has a matching 'source' in the DB is marked succeeded (skipped
on future runs); everything else is marked failed/retryable. Run this
once, before your next scrape, to skip re-processing PDFs that already
made it into your knowledge base.

Usage:
    python utils/migrate_scrape_tracker.py

Run this from the project root, same place you'd run index_bmsit.py.
No API key needed - this only touches RAGSystem/Chroma, which uses local
FastEmbed embeddings, not an LLM provider.
"""
import json
import os
import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
os.chdir(parent_dir)

from retrieval.rag_system import RAGSystem
from ingestion.scrape_tracker import TRACKER_FILE


def main():
    if not os.path.exists(TRACKER_FILE):
        print(f"No {TRACKER_FILE} found at the project root - nothing to migrate.")
        return

    with open(TRACKER_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'succeeded' in data or 'failed' in data:
        print(
            f"{TRACKER_FILE} is already in the current format "
            f"({len(data.get('succeeded', []))} succeeded, "
            f"{len(data.get('failed', {}))} failed). Nothing to migrate."
        )
        return

    legacy_urls = data.get('urls', [])
    if not legacy_urls:
        print(f"{TRACKER_FILE} has no 'urls' to migrate.")
        return

    print(f"Found {len(legacy_urls)} legacy-format tracked URLs.")
    print("Loading the actual Chroma DB to check what's really indexed...")

    rag = RAGSystem()
    if rag.vectorstore is None:
        print("ERROR: Could not load the vector store - aborting migration.")
        return

    collection = rag.vectorstore._collection
    result = collection.get(limit=10000, include=['metadatas'])
    indexed_sources = set()
    for metadata in result.get('metadatas', []) or []:
        source = (metadata or {}).get('source')
        if source:
            indexed_sources.add(source)

    print(f"Found {len(indexed_sources)} unique source URLs actually indexed in the DB.")

    succeeded = []
    failed = {}
    for url in legacy_urls:
        if url in indexed_sources:
            succeeded.append(url)
        else:
            failed[url] = 'legacy_unverified'

    new_data = {
        'succeeded': succeeded,
        'failed': failed,
    }
    with open(TRACKER_FILE, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    print(f"\nMigration complete:")
    print(f"  {len(succeeded)} URLs confirmed indexed -> marked succeeded (will be skipped)")
    print(f"  {len(failed)} URLs not found in the DB -> marked retryable")
    print(f"\nYou can now re-run your scraper; only the {len(failed)} retryable URLs will be attempted.")


if __name__ == "__main__":
    main()