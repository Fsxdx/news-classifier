import asyncio
import csv
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, List

import aiohttp
from aiohttp import ClientResponseError, ClientSession, TCPConnector
from bs4 import BeautifulSoup

logger = logging.getLogger("LentaParser")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


class AsyncLentaParser:
    """
    Asynchronous parser for Lenta.ru news articles by date.
    """

    BASE_URL = "https://lenta.ru/news"
    HTML_PARSER = "html.parser"

    def __init__(
            self,
            from_date: str,
            out_csv: Path,
            max_workers: int = 4,
    ) -> None:
        self.from_date = datetime.strptime(from_date, "%d.%m.%Y").date()
        self.out_csv = out_csv
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.timeout = aiohttp.ClientTimeout(total=60)
        self.session: Optional[ClientSession] = None
        self._downloaded = 0

    async def __aenter__(self) -> "AsyncLentaParser":
        connector = TCPConnector(use_dns_cache=True, ttl_dns_cache=3600, limit=100)
        self.session = ClientSession(connector=connector, timeout=self.timeout)
        self._csv_file = self.out_csv.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._csv_file,
            fieldnames=["url", "title", "text", "topic", "tags", "date"],
        )
        self._writer.writeheader()
        return self

    async def __aexit__(
            self, exc_type, exc_val, exc_tb
    ) -> None:
        if self.session:
            await self.session.close()
        self.executor.shutdown(wait=True)
        self._csv_file.close()
        logger.info(f"Finished: {self._downloaded} articles saved to {self.out_csv}")

    async def fetch_html(self, url: str) -> str:
        """Fetch HTML content asynchronously."""
        logger.info(f"Fetching page: {url}")
        assert self.session, "Session not initialized"
        async with self.session.get(url, allow_redirects=False) as resp:
            resp.raise_for_status()
            text = await resp.text(encoding="utf-8")
            logger.info(f"Fetched page: {url} (size={len(text)})")
            return text

    def parse_article(self, html: str) -> Dict[str, Optional[str]]:
        """Parse article HTML and extract metadata and text."""
        soup = BeautifulSoup(html, self.HTML_PARSER)

        def find_text(selector: str) -> Optional[str]:
            elem = soup.select_one(selector)
            return elem.get_text(strip=True) if elem else None

        tags = find_text(".rubric-header__link._active")
        title = find_text(".topic-body__title")
        topic = find_text(".rubric-header__title")
        body = soup.select('div.topic-body__content')
        if not body:
            raise ValueError("Article body not found")
        text = "\n".join(p.get_text(strip=True) for p in body[0].find_all("p"))

        return {"title": title, "text": text, "topic": topic, "tags": tags}

    def extract_urls(self, html: str) -> List[str]:
        """Return list of full article URLs from a news listing page."""
        soup = BeautifulSoup(html, self.HTML_PARSER)
        items = soup.select(".card-full-news")
        return [f"https://lenta.ru{a.get('href')}" for a in items if a.get('href')]

    async def process_day(self, date_str: str) -> int:
        """Fetch and parse all articles for a given date. Returns count."""
        page_url = f"{self.BASE_URL}/{date_str}"
        try:
            listing_html = await self.fetch_html(page_url)
        except (ClientResponseError, asyncio.TimeoutError) as e:
            logger.error(f"Error fetching listing page {page_url}: {e}")
            return 0

        urls = self.extract_urls(listing_html)

        fetch_tasks = [self.fetch_html(url) for url in urls]
        pages = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        results = []
        loop = asyncio.get_running_loop()
        for url, content in zip(urls, pages):
            if isinstance(content, Exception):
                logger.warning(f"Skipping fetch of {url}: {content}")
                continue
            logger.info(f"Scheduling parse for article: {url}")
            parse_task = loop.run_in_executor(self.executor, self.parse_article, content)
            try:
                data = await parse_task
                data["url"] = url
                data["date"] = date_str
                logger.info(f"Parsed article: {url}")
                results.append(data)
            except Exception as e:
                logger.warning(f"Failed to parse {url}: {e}")

        if results:
            self._writer.writerows(results)
            self._downloaded += len(results)
            logger.info(f"Saved {len(results)} articles from {page_url}")

        return len(results)

    async def run(self) -> None:
        """Main entry: iterate from from_date to today and process each day."""
        current = self.from_date
        end = datetime.now(timezone.utc).date()
        while current <= end:
            date_path = current.strftime("%Y/%m/%d")
            logger.info(f"Processing date: {date_path}")
            count = await self.process_day(date_path)
            logger.info(f"Completed date: {date_path}, {count} articles, total={self._downloaded}")
            current += timedelta(days=1)