"""
College website scraper for UniBot
Scrapes and extracts content from college website pages, including PDFs
"""
from playwright.async_api import async_playwright, Browser, Page
from typing import List, Dict, Optional
import asyncio
from urllib.parse import urljoin, urlparse
import re
import os
from datetime import datetime
from .pdf_processor import PDFProcessor
from .scrape_tracker import ScrapeTracker
from .scrape_checkpoint import ScrapeCheckpoint

class CollegeScraper:
    def __init__(self, base_url: str, max_pages: int = 500, max_depth: int = 8, include_pdfs: bool = True):
        """
        Initialize the college scraper
        
        Args:
            base_url: Base URL of the college website (e.g., "https://college.edu")
            max_pages: Maximum number of pages to scrape (increased default for deeper crawling)
            max_depth: Maximum depth to crawl from base URL (increased for deeper crawling)
            include_pdfs: Whether to extract content from PDF files
        """
        self.base_url = base_url.rstrip('/')
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.include_pdfs = include_pdfs
        self.scraped_urls = set()
        self.playwright = None
        self.browser = None
        self.pdf_processor = PDFProcessor() if include_pdfs else None
        self.tracker = ScrapeTracker()  # Track scraped URLs across runs
        self.checkpoint = ScrapeCheckpoint()  # Checkpoint for resuming
    
    async def setup(self):
        """Initialize Playwright browser"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
    
    async def cleanup(self):
        """Close browser and playwright"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and belongs to the college website."""
        try:
            parsed = urlparse(url)
            base_parsed = urlparse(self.base_url)

            # Must use HTTP/HTTPS
            if parsed.scheme not in ("http", "https"):
                return False

            # Must be the exact same hostname
            if parsed.hostname != base_parsed.hostname:
                return False

            # Reject paths containing obvious external-domain patterns
            path_lower = parsed.path.lower()

            blocked_patterns = [
                "linkedin.com",
                "youtube.com",
                "facebook.com",
                "instagram.com",
                "twitter.com",
                "x.com",
                "gmail.com",
                "google.com",
            ]

            if any(pattern in path_lower for pattern in blocked_patterns):
                return False

            # Skip common non-content files
            skip_extensions = (
                ".doc", ".docx", ".xls", ".xlsx",
                ".zip", ".rar",
                ".jpg", ".jpeg", ".png",
                ".gif", ".svg", ".ico", ".webp"
            )

            if path_lower.endswith(skip_extensions):
                return False

            # Skip anchors and special URLs
            if "#" in url:
                return False

            return True

        except Exception:
            return False
    
    async def _extract_text_content(self, page: Page) -> Dict[str, str]:
        """
        Extract text content from a page
        
        Returns:
            Dictionary with 'title', 'content', and 'url'
        """
        try:
            # Don't wait for networkidle - just wait a bit for content
            # The page should already be loaded from scrape_page
            await page.wait_for_timeout(1000)  # Wait 1 second for any dynamic content
            
            # Extract title
            title = await page.title()
            
            # Extract main content - try to get main content area
            content_selectors = [
                'main',
                'article',
                '[role="main"]',
                '.content',
                '.main-content',
                '#content',
                '#main',
                'body'
            ]
            
            content = ""
            for selector in content_selectors:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        content = await element.inner_text()
                        if len(content) > 100:  # If we got substantial content
                            break
                except:
                    continue
            
            # If no good content found, get body text
            if len(content) < 100:
                try:
                    body = await page.query_selector('body')
                    if body:
                        content = await body.inner_text()
                except:
                    # If body extraction fails, try getting text directly
                    try:
                        content = await page.evaluate("() => document.body.innerText")
                    except:
                        content = ""
            
            # Clean up content
            content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
            content = content.strip()
            
            return {
                'title': title,
                'content': content,
                'url': page.url
            }
        except Exception as e:
            print(f"Error extracting content from {page.url}: {e}")
            # Try to get at least something
            try:
                title = await page.title()
                content = await page.evaluate("() => document.body ? document.body.innerText : ''")
                content = re.sub(r'\s+', ' ', content).strip()
                return {
                    'title': title,
                    'content': content,
                    'url': page.url
                }
            except:
                return {
                    'title': '',
                    'content': '',
                    'url': page.url
                }
    
    async def _get_links(self, page: Page) -> List[str]:
        """Extract all links from a page - fast version, includes PDF detection"""
        try:
            # Very fast link extraction with short timeout
            # Also extract PDF links specifically
            links_data = await asyncio.wait_for(
                page.evaluate("""
                    () => {
                        const links = Array.from(document.querySelectorAll('a[href]'));
                        const allLinks = links.map(link => link.href);
                        // Also check for PDF links in page content
                        const pdfLinks = [];
                        links.forEach(link => {
                            const href = link.href;
                            if (href && (href.toLowerCase().endsWith('.pdf') || 
                                href.toLowerCase().includes('.pdf') ||
                                link.textContent.toLowerCase().includes('pdf'))) {
                                pdfLinks.push(href);
                            }
                        });
                        return { all: allLinks, pdfs: pdfLinks };
                    }
                """),
                timeout=2.0  # 2 second timeout - be fast!
            )
            
            all_links = links_data.get('all', []) if isinstance(links_data, dict) else links_data
            pdf_links = links_data.get('pdfs', []) if isinstance(links_data, dict) else []
            
            # Filter and normalize links
            valid_links = []
            for link in all_links:
                if not link:
                    continue
                
                # Convert relative URLs to absolute
                absolute_url = urljoin(page.url, link)
                
                if self._is_valid_url(absolute_url):
                    valid_links.append(absolute_url)
            
            # Add PDF links explicitly (they might be filtered out by _is_valid_url)
            for pdf_link in pdf_links:
                absolute_pdf = urljoin(page.url, pdf_link)
                if absolute_pdf not in valid_links and self._is_valid_url(absolute_pdf):
                    valid_links.append(absolute_pdf)
            
            return list(set(valid_links))  # Remove duplicates
        except Exception as e:
            # Fallback to simple link extraction
            try:
                links = await asyncio.wait_for(
                    page.evaluate("""
                        () => {
                            const links = Array.from(document.querySelectorAll('a[href]'));
                            return links.map(link => link.href);
                        }
                    """),
                    timeout=2.0
                )
                valid_links = []
                for link in links:
                    if not link:
                        continue
                    absolute_url = urljoin(page.url, link)
                    if self._is_valid_url(absolute_url):
                        valid_links.append(absolute_url)
                return list(set(valid_links))
            except:
                return []
    
    def _get_url_depth(self, url: str) -> int:
        """Calculate depth of URL from base URL"""
        base_path = urlparse(self.base_url).path
        url_path = urlparse(url).path
        
        base_depth = len([p for p in base_path.split('/') if p])
        url_depth = len([p for p in url_path.split('/') if p])
        
        return url_depth - base_depth
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL to avoid duplicates (remove trailing slash, lowercase)"""
        url = url.rstrip('/')
        # Remove fragment
        if '#' in url:
            url = url.split('#')[0]
        return url.lower()
    
    async def scrape_pdf(self, url: str) -> Optional[Dict[str, str]]:
        """
        Scrape a PDF file
        
        Args:
            url: URL of the PDF file
            
        Returns:
            Dictionary with PDF content or None if failed
        """
        if not self.pdf_processor or not PDFProcessor.is_pdf_url(url):
            return None
        
        normalized_url = self._normalize_url(url)
        
        if normalized_url in self.scraped_urls:
            return None
        
        print(f"  📄 Processing PDF: {url}")
        
        try:
            content = self.pdf_processor.process_pdf_url(url)
            if content:
                self.scraped_urls.add(normalized_url)
                content['url'] = url
                quality_score = content.get('quality_score', 0.0)
                page_count = content.get('page_count', 0)
                char_count = len(content['content'])
                
                # Show quality indicator
                quality_indicator = "✓" if quality_score >= 0.7 else "⚠" if quality_score >= 0.5 else "⚠"
                print(f"  {quality_indicator} Successfully extracted PDF: {char_count} characters, {page_count} pages, quality: {quality_score:.2f}")
                return content
            else:
                print(f"  ✗ PDF extraction failed or quality too low (filtered out)")
                self.scraped_urls.add(normalized_url)  # Mark as tried even if failed
                return None
        except Exception as e:
            print(f"  ✗ Error processing PDF {url}: {e}")
            self.scraped_urls.add(normalized_url)
            return None
    
    async def scrape_page(self, url: str) -> Optional[Dict[str, str]]:
        """
        Scrape a single page (HTML or PDF)
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary with page content or None if failed
        """
        # Check if it's a PDF first
        if self.pdf_processor and PDFProcessor.is_pdf_url(url):
            return await self.scrape_pdf(url)
        
        # Store original URL for metadata
        original_url = url
        # Normalize URL for deduplication
        normalized_url = self._normalize_url(url)
        
        if normalized_url in self.scraped_urls:
            return None
        
        if not self._is_valid_url(original_url):
            return None
        
        if self._get_url_depth(original_url) > self.max_depth:
            return None
        
        try:
            page = await self.browser.new_page()
            # Use 'domcontentloaded' instead of 'networkidle' for faster loading
            # Increase timeout to 60 seconds
            # Fast page load - be aggressive with timeouts
            try:
                await asyncio.wait_for(
                    page.goto(original_url, wait_until='domcontentloaded', timeout=30000),
                    timeout=35.0
                )
                await page.wait_for_timeout(300)  # Wait 0.3 seconds only
            except (asyncio.TimeoutError, Exception):
                try:
                    await asyncio.wait_for(
                        page.goto(original_url, wait_until='load', timeout=30000),
                        timeout=35.0
                    )
                except:
                    await page.close()
                    self.scraped_urls.add(normalized_url)
                    return None
            
            # Extract content first (most important)
            content = await self._extract_text_content(page)
            
            # Extract links quickly with timeout - don't block if it fails
            links = []
            try:
                links = await asyncio.wait_for(self._get_links(page), timeout=2.0)
            except:
                pass  # Continue without links if extraction fails
            
            # Always mark as scraped
            self.scraped_urls.add(normalized_url)
            
            # Use the original requested URL for metadata (not page.url which might redirect)
            # This ensures each page has its unique URL stored
            content['url'] = original_url  # Store the URL we requested, not where we ended up
            content['links'] = links  # Include links if we got them
            
            if len(content['content']) > 50:  # Only return if we got meaningful content
                await page.close()
                return content
            else:
                await page.close()
                return None
        except Exception as e:
            print(f"Error scraping {original_url}: {e}")
            # Mark as scraped even on error to avoid infinite retries
            self.scraped_urls.add(normalized_url)
            try:
                await page.close()
            except:
                pass
            return None
    
    async def scrape_website(self, start_urls: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """
        Scrape the college website starting from given URLs
        
        Args:
            start_urls: List of URLs to start scraping from. If None, starts from base_url
            
        Returns:
            List of dictionaries with scraped content
        """
        if start_urls is None:
            start_urls = [self.base_url]
        
        await self.setup()
        
        # Try to resume from checkpoint (auto-resume, no user input for scripts)
        checkpoint = self.checkpoint.get_checkpoint()
        if checkpoint.get('scraped_content') and checkpoint.get('last_updated'):
            print(f"📋 Found checkpoint from {checkpoint['last_updated']}")
            print("  Auto-resuming from checkpoint...")
            print("  (To start fresh, delete scrape_checkpoint.json)")
            scraped_content = checkpoint['scraped_content']
            pdf_urls = checkpoint['pdf_urls']
            visited_urls = set(checkpoint['visited_urls'])
            urls_to_visit = checkpoint['urls_to_visit']
            print(f"  Resuming: {len(scraped_content)} pages, {len(pdf_urls)} PDFs, {len(urls_to_visit)} URLs to visit")
        else:
            scraped_content = []
            # Normalize start URLs
            urls_to_visit = [self._normalize_url(url) for url in start_urls]
            visited_urls = set()
            pdf_urls = []  # Initialize PDF URLs list
        
        try:
            # Save checkpoint every 10 pages
            checkpoint_interval = 10
            last_checkpoint = len(scraped_content)
            
            while urls_to_visit and len(scraped_content) < self.max_pages:
                current_url = urls_to_visit.pop(0)
                original_url = current_url  # Keep original for PDF detection
                
                # Normalize URL
                current_url = self._normalize_url(current_url)
                
                # Check both visited_urls, scraped_urls, and tracker
                if (current_url in visited_urls or 
                    current_url in self.scraped_urls or 
                    self.tracker.is_scraped(current_url)):
                    if self.tracker.is_scraped(current_url):
                        print(f"  ⏭️  Skipping (already scraped): {current_url[:60]}...", flush=True)
                    continue
                
                visited_urls.add(current_url)
                
                # Check if it's a PDF (check both normalized and original URL)
                is_pdf = (self.include_pdfs and 
                         (PDFProcessor.is_pdf_url(current_url) or 
                          PDFProcessor.is_pdf_url(original_url) or
                          current_url.lower().endswith('.pdf') or
                          original_url.lower().endswith('.pdf')))
                
                if is_pdf:
                    # Use original URL if available, otherwise normalized
                    pdf_url = original_url if original_url else current_url
                    if pdf_url not in pdf_urls:
                        print(f"Found PDF: {pdf_url[:80]}... (queued for processing)", flush=True)
                        pdf_urls.append(pdf_url)
                    self.scraped_urls.add(current_url)  # Dedup within this crawl only -
                    # NOT tracker.mark_scraped() here: this PDF has only been
                    # *discovered*, not processed yet. Marking it "succeeded"
                    # at discovery time (before extraction is even attempted)
                    # was the root cause of failed PDFs being silently
                    # skipped forever - see the actual mark_succeeded/
                    # mark_failed calls in the PDF processing loop below,
                    # which run after a real extraction attempt.
                    continue
                
                # Progress logging with percentage and time
                progress = len(scraped_content)
                percentage = (progress / self.max_pages * 100) if self.max_pages > 0 else 0
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] [{progress}/{self.max_pages}] ({percentage:.1f}%) Scraping: {current_url[:70]}...", flush=True)
                
                # Scrape page with fast timeout
                # IMPORTANT: fetch using original_url (preserves case),
                # current_url is only for dedup/tracking (lowercased).
                try:
                    content = await asyncio.wait_for(
                        self.scrape_page(original_url),
                        timeout=45.0  # 15 second timeout - be aggressive!
                    )
                    
                    if content:
                        content_length = len(content['content'])
                        # Mark as scraped in tracker
                        self.tracker.mark_scraped(current_url)
                        print(f"  ✓ Scraped: {content_length:,} chars", flush=True)
                        scraped_content.append(content)
                        
                        # Process all links for deep crawling - don't limit
                        links = content.get('links', [])
                        if links:
                            new_links_count = 0
                            pdf_links_found = 0
                            
                            # Process all links, not just first 50, for deep crawling
                            for link in links:
                                normalized_link = self._normalize_url(link)
                                
                                # Check if it's a PDF link FIRST
                                if self.include_pdfs and PDFProcessor.is_pdf_url(link):
                                    if link not in pdf_urls and normalized_link not in self.scraped_urls:
                                        pdf_urls.append(link)
                                        pdf_links_found += 1
                                        self.scraped_urls.add(normalized_link)
                                        # Not tracker.mark_scraped() here either -
                                        # same reasoning as the discovery point
                                        # above: this is a newly-found link, not
                                        # a processed one.
                                    continue
                                
                                if (normalized_link not in visited_urls and 
                                    normalized_link not in urls_to_visit and 
                                    normalized_link not in self.scraped_urls):
                                    if self._is_valid_url(link):
                                        link_depth = self._get_url_depth(normalized_link)
                                        if link_depth <= self.max_depth:
                                            urls_to_visit.append(normalized_link)
                                            new_links_count += 1
                            
                            if new_links_count > 0:
                                print(f"  → {new_links_count} new links (depth: {self.max_depth})", flush=True)
                            if pdf_links_found > 0:
                                print(f"  📄 Found {pdf_links_found} PDF link(s)", flush=True)
                    else:
                        print(f"  ✗ No content", flush=True)
                    
                    # Save checkpoint periodically (regardless of content)
                    if len(scraped_content) - last_checkpoint >= checkpoint_interval:
                        self.checkpoint.save(scraped_content, pdf_urls, visited_urls, urls_to_visit, len(scraped_content))
                        last_checkpoint = len(scraped_content)
                        print(f"  💾 Checkpoint saved ({len(scraped_content)} pages)", flush=True)
                except asyncio.TimeoutError:
                    print(f"  ⚠ Timeout, skipping...", flush=True)
                    self.scraped_urls.add(self._normalize_url(current_url))
                    # Save checkpoint on error too
                    if len(scraped_content) - last_checkpoint >= checkpoint_interval:
                        self.checkpoint.save(scraped_content, pdf_urls, visited_urls, urls_to_visit, len(scraped_content))
                        last_checkpoint = len(scraped_content)
                    continue
                except Exception as e:
                    print(f"  ⚠ Error: {str(e)[:50]}", flush=True)
                    self.scraped_urls.add(self._normalize_url(current_url))
                    # Save checkpoint on error too
                    if len(scraped_content) - last_checkpoint >= checkpoint_interval:
                        self.checkpoint.save(scraped_content, pdf_urls, visited_urls, urls_to_visit, len(scraped_content))
                        last_checkpoint = len(scraped_content)
                    continue
        finally:
            await self.cleanup()
        
        # Process PDFs if enabled (after scraping is complete)
        # PDFs are processed separately and don't count towards max_pages limit
        # Use asyncio for timeout support
        if self.include_pdfs and pdf_urls:
            print(f"\n{'='*60}")
            print(f"Processing {len(pdf_urls)} PDF files (separate from page limit)...")
            print(f"{'='*60}\n")
            
            pdf_count = 0
            failed_count = 0
            skipped_count = 0
            
            for idx, pdf_url in enumerate(pdf_urls, 1):
                # Show progress every 10 PDFs or on important milestones
                if idx % 10 == 0 or idx == 1 or idx == len(pdf_urls):
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp}] Processing PDF ({idx}/{len(pdf_urls)}) - Success: {pdf_count}, Failed: {failed_count}", flush=True)

                # Save tracker progress periodically, not just at the very
                # end of the whole scrape. Without this, a Ctrl+C (or
                # crash, or timeout) partway through a long PDF run loses
                # ALL of that run's succeeded/failed progress - it only
                # existed in memory - forcing every already-processed PDF
                # to be redone from scratch on the next run.
                if idx % 25 == 0:
                    self.tracker.save()
                
                # Check if already succeeded before - failures are retried
                if self.tracker.is_succeeded(pdf_url):
                    skipped_count += 1
                    continue
                
                try:
                    # Add timeout to PDF processing to prevent hanging
                    try:
                        pdf_content = await asyncio.wait_for(
                            asyncio.to_thread(PDFProcessor.extract_text, pdf_url),
                            timeout=30.0  # 30 second timeout per PDF
                        )
                    except asyncio.TimeoutError:
                        print(f"  ⏱️  PDF timeout ({idx}/{len(pdf_urls)}): {pdf_url[:60]}...", flush=True)
                        pdf_content = {'success': False, 'failure_reason': 'timeout'}
                    
                    if pdf_content.get('success') and len(pdf_content.get('content', '')) > 50:
                        pdf_content['type'] = 'pdf'
                        pdf_content['metadata'] = {
                            'source_type': 'pdf', 
                            'file_name': os.path.basename(pdf_url),
                            'content_category': 'syllabus' if 'syllabus' in pdf_url.lower() else 'document'
                        }
                        scraped_content.append(pdf_content)
                        # Only a real, successful extraction is marked
                        # succeeded - this is what makes it permanent.
                        self.tracker.mark_succeeded(pdf_url)
                        pdf_count += 1
                        
                        # Log quality if available
                        quality_score = pdf_content.get('quality_score')
                        if quality_score is not None:
                            quality_indicator = "✓" if quality_score >= 0.7 else "⚠"
                            print(f"  {quality_indicator} PDF quality: {quality_score:.2f} - {os.path.basename(pdf_url)[:50]}", flush=True)
                    else:
                        failed_count += 1
                        reason = pdf_content.get('failure_reason', 'unknown')
                        # Mark as failed (NOT succeeded) - stays eligible
                        # for retry on the next run, instead of being
                        # silently skipped forever.
                        self.tracker.mark_failed(pdf_url, reason)
                        print(f"  ✗ PDF failed ({reason}): {os.path.basename(pdf_url)[:50]}", flush=True)
                except Exception as e:
                    failed_count += 1
                    # Mark as failed with the exception, not succeeded -
                    # stays eligible for retry.
                    self.tracker.mark_failed(pdf_url, f'exception_{type(e).__name__}')
                    # Only show error for first few failures, then be quiet
                    if failed_count <= 5:
                        print(f"  ⚠ PDF error ({idx}/{len(pdf_urls)}): {str(e)[:60]}...", flush=True)
            
            print(f"\n{'='*60}")
            print(f"PDF Processing Summary:")
            print(f"  ✓ Successfully processed: {pdf_count}")
            print(f"  ✗ Failed (retryable next run): {failed_count}")
            print(f"  ⏭️  Skipped (already succeeded before): {skipped_count}")
            print(f"  📊 Total: {len(pdf_urls)} PDFs")
            failure_reasons = self.tracker.get_failure_reasons()
            if failure_reasons:
                print(f"\n  Failure breakdown (all retryable PDFs, not just this run):")
                for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1]):
                    print(f"    {reason}: {count}")
            print(f"{'='*60}")
        
        # Save final checkpoint
        self.checkpoint.save(scraped_content, pdf_urls, visited_urls, urls_to_visit, len(scraped_content))
        
        # Save tracker at the end
        self.tracker.save()
        print(f"\n📋 Scrape tracker updated: {self.tracker.get_count()} URLs tracked")
        print(f"💾 Final checkpoint saved: {len(scraped_content)} pages, {len(pdf_urls)} PDFs")
        
        return scraped_content