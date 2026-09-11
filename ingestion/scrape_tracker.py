"""
Scrape tracker to remember what URLs have already been scraped.

IMPORTANT: this distinguishes real successes from failures. Only a real
success is permanent - failures (timeouts, dead links, unreadable PDFs,
low-quality extractions) stay eligible for retry on the next run, instead
of being silently skipped forever.

Legacy compatibility: the old format was a flat {'urls': [...]} list with
no success/failure distinction. On load(), any legacy entries are placed
into `failed` with reason 'legacy_unverified' - i.e. they become eligible
for retry - since we can't otherwise tell which of them actually
succeeded. This is a one-time, automatic, safe default: re-processing an
already-successful PDF just costs a little extra time (the indexing step
already skips true duplicates), it doesn't create bad data.

For a more precise one-time migration that cross-references the existing
Chroma DB to avoid re-processing genuinely successful PDFs, see
utils/migrate_scrape_tracker.py.
"""
import json
import os
from typing import Set, Dict, List
from datetime import datetime

TRACKER_FILE = "scrape_tracker.json"


class ScrapeTracker:
    """Track scraped URLs, split into real successes and retryable failures."""

    def __init__(self):
        self.succeeded: Set[str] = set()
        self.failed: Dict[str, str] = {}  # url -> failure_reason
        self.load()

    def load(self):
        """Load previously tracked URLs from file, migrating the legacy format if needed."""
        if not os.path.exists(TRACKER_FILE):
            self.succeeded = set()
            self.failed = {}
            return

        try:
            with open(TRACKER_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'succeeded' in data or 'failed' in data:
                # Current format
                self.succeeded = set(data.get('succeeded', []))
                self.failed = dict(data.get('failed', {}))
            else:
                # Legacy format: {'urls': [...]}. We can't tell which of
                # these actually succeeded, so treat them all as retryable
                # failures rather than silently trusting them as done.
                legacy_urls = data.get('urls', [])
                self.succeeded = set()
                self.failed = {url: 'legacy_unverified' for url in legacy_urls}
                print(
                    f"Migrated {len(legacy_urls)} legacy tracker entries -> "
                    f"marked as retryable (reason: legacy_unverified). "
                    f"Run utils/migrate_scrape_tracker.py first if you want to "
                    f"cross-reference against the existing Chroma DB instead, "
                    f"to avoid re-processing PDFs that already succeeded."
                )

            print(
                f"Loaded scrape tracker: {len(self.succeeded)} succeeded, "
                f"{len(self.failed)} failed/retryable"
            )
        except Exception as e:
            print(f"WARNING: Could not load scrape tracker: {e}")
            self.succeeded = set()
            self.failed = {}

    def save(self):
        """Save tracker state to file."""
        try:
            data = {
                'succeeded': list(self.succeeded),
                'failed': self.failed,
                'last_updated': datetime.now().isoformat(),
                'total_succeeded': len(self.succeeded),
                'total_failed': len(self.failed)
            }
            with open(TRACKER_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"WARNING: Could not save scrape tracker: {e}")

    def is_succeeded(self, url: str) -> bool:
        """True only if this URL has been genuinely, successfully processed before."""
        return url in self.succeeded

    def mark_succeeded(self, url: str):
        """Mark URL as genuinely, successfully processed. Permanent - never retried again."""
        self.succeeded.add(url)
        self.failed.pop(url, None)

    def mark_failed(self, url: str, reason: str = 'unknown'):
        """Mark URL as failed this attempt, with a reason. Stays eligible for retry."""
        if url not in self.succeeded:
            self.failed[url] = reason

    def get_failure_reasons(self) -> Dict[str, int]:
        """Count of failed URLs grouped by reason, for a summary report."""
        counts: Dict[str, int] = {}
        for reason in self.failed.values():
            counts[reason] = counts.get(reason, 0) + 1
        return counts

    def get_retryable_urls(self) -> List[str]:
        """URLs that failed before and are eligible to be retried."""
        return list(self.failed.keys())

    # --- Backward-compatible aliases -----------------------------------
    # These map to the *success* semantics. Existing callers that just
    # want "should I skip this URL" (i.e. HTML page crawling, which only
    # ever called mark_scraped() after a real success) keep working
    # unchanged. Callers that need failure tracking use the new
    # mark_failed()/is_succeeded() API above directly.

    def is_scraped(self, url: str) -> bool:
        """Alias for is_succeeded() - kept for backward compatibility."""
        return self.is_succeeded(url)

    def mark_scraped(self, url: str):
        """Alias for mark_succeeded() - kept for backward compatibility."""
        self.mark_succeeded(url)

    def mark_multiple_scraped(self, urls: list):
        """Mark multiple URLs as succeeded."""
        for url in urls:
            self.mark_succeeded(url)

    def get_count(self) -> int:
        """Get count of successfully tracked URLs."""
        return len(self.succeeded)

    def clear(self):
        """Clear all tracked URLs (for a fresh start)."""
        self.succeeded.clear()
        self.failed.clear()
        if os.path.exists(TRACKER_FILE):
            os.remove(TRACKER_FILE)
        print("Scrape tracker cleared")