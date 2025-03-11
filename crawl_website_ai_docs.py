import os
import sys
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
import time
from utils import chunk_text, process_chunk, insert_chunk


from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI


load_dotenv()
openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))


async def process_and_store_document(url:str, markdown:str):
    """Process a document and store its chunks in parallel"""

    # split into chunks:
    chunks = chunk_text(markdown)

    #process chunks in parallel
    tasks = [
        process_chunk(chunk, i, url) 
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)

    #Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk)
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)




async def crawl_parallel(urls: List[str], maximum_concurrent: int = 5):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],

    )
    
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        #Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(maximum_concurrent)

        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    
                    await process_and_store_document(url, result.markdown_v2.raw_markdown)

                    # save into file:
                    # filename = str(time.time()) + ".md"
                    # with open(filename, "w") as f:
                    #     f.write(result.markdown_v2.raw_markdown)

                else:
                    print(f"Failed: {url} - Error: {result.error_message}")

        # Process all URL in Parallel with limited concurrency            
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

def get_pydantic_ai_docs_urls() -> List[str]:
    """ Get URLs from pydantic AI docs sitemap"""
    sitemap_url = "https://ai.pydantic.dev/sitemap.xml"

    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()


        #Parse the XML
        root = ElementTree.fromstring(response.content)

        # extract all URLs from the sitemap
        namespaces = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls=[loc.text for loc in root.findall('.//ns:loc', namespaces)]

        return urls
    
    except Exception as e:
        print(f"Error Fetching sitemap: {e}")
        return []


async def main():  
    # get all URLs from the sitemap
    urls = get_pydantic_ai_docs_urls()
    if not urls:
        print("No URLs found to crawl")
        return
    print(f"{len(urls)} URLs found to crawl")

    #crawl each url in parallel
    await crawl_parallel(urls=urls)


if __name__ == "__main__":
    asyncio.run(main())
