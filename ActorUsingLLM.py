from __future__ import annotations
from urllib.parse import urljoin
from apify import Actor, Request
from bs4 import BeautifulSoup
from httpx import AsyncClient
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from datetime import datetime
import openai
import asyncio

# Step 0: Setup OpenAPI key

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY") 
openai.api_key = openai_api_key

# 1. Determine relevant BBC Sport URLs using LLM
async def determine_relevant_urls(user_query: str, openai_api_key: str) -> list[str]:
    """Use OpenAI LLM to select relevant BBC Sport section URLs based on the user query."""
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=openai_api_key
    )

    # Prompt instructs LLM to select relevant sport section URLs
    template = """
    You are a sports news assistant. Based on the user query below, choose the most relevant BBC Sport section URLs.
    Query: "{user_query}"

    Choose from these URLs (select ALL that apply, separated by new lines):
    - cricket: <https://www.bbc.com/sport/cricket>
    - football: <https://www.bbc.com/sport/football>
    - tennis: <https://www.bbc.com/sport/tennis>
    - formula 1: <https://www.bbc.com/sport/formula1>
    - golf: <https://www.bbc.com/sport/golf>
    - rugby: <https://www.bbc.com/sport/rugby-union>
    - Athletics: <https://www.bbc.com/sport/athletics>
    - cycling: <https://www.bbc.com/sport/cycling>
    - general sports: <https://www.bbc.com/sport>

    Respond with only the URLs, one per line. If you choose multiple, separate them with new lines.
    """

    prompt = PromptTemplate(
        input_variables=["user_query"],
        template=template
    )
    chain = prompt | llm | StrOutputParser()
    try:
        urls = await chain.ainvoke({"user_query": user_query})
        urls = urls.strip()
        if "\n" in urls:
            # Handle multiple URLs returned by LLM
            url_list = [url.strip() for url in urls.split("\n") if url.strip().startswith("https://www.bbc.com/sport/") and url.strip() != "https://www.bbc.com/sport"]
            if not url_list:
                Actor.log.warning("No specific sport URLs returned by LLM.")
                return []
            return list(set(url_list))
        else:
            # Handle single sport label (e.g., "football")
            sport_name = urls.lower()
            if sport_name == "general sports":
                Actor.log.info("LLM selected general sports.")
                return ["https://www.bbc.com/sport"]
            # If it's a relative URL like "/sport/cricket", construct the full URL
            if not sport_name.startswith("https://"):
                return [f"https://www.bbc.com/sport/{sport_name}"]
            # If it's already a full URL (starts with https://), return it as is
            return [sport_name]
    except Exception as e:
        Actor.log.error(f"Error determining URLs: {e}")
        Actor.log.warning("LLM failed to provide URLs, skipping scraping.")
        return []


async def scrape_sport_headlines_with_url(url: str, client: AsyncClient) -> list[dict]:
    """Scrape sport news headlines and URLs from a given BBC Sport URL, targeting the specified HTML structure."""
    try:
        Actor.log.info(f"Scraping URL: {url}")
        response = await client.get(url, follow_redirects=True)
        soup = BeautifulSoup(response.content, 'html.parser')
        headlines_data = []
        seen_urls = set()
        promo_links = soup.find_all('a', class_='ssrcss-sxweo-PromoLink exn3ah95', href=True)
        if promo_links:
            for link_tag in promo_links:
                headline_paragraph = link_tag.find('p', class_='ssrcss-1b1mki6-PromoHeadline exn3ah910')
                if headline_paragraph:
                    headline = headline_paragraph.find('span').get_text(strip=True) if headline_paragraph.find('span') else headline_paragraph.get_text(strip=True)
                    article_url = urljoin(url, link_tag['href'])
                    if article_url not in seen_urls:
                        seen_urls.add(article_url)
                        headlines_data.append({'headline': headline, 'url': article_url, 'source': url ,'scraped_at_readable': datetime.utcnow().strftime('%d %b %Y, %H:%M UTC')})
            return headlines_data

        # Fallback to generic method
        for link in soup.find_all('a', href=True):
            if link['href'].startswith('/sport/') and link.find('span'):
                article_url = urljoin(url, link['href'])
                seen_urls.add(article_url)
                headlines_data.append({
                    'headline': link.find('span').get_text(strip=True),
                    'url': article_url,
                    'source': url,
                    'scraped_at_readable': datetime.utcnow().strftime('%d %b %Y, %H:%M UTC')
                })
        return headlines_data
    except Exception as e:
        print(f"Error scraping headlines from {url}: {e}")
        return []

# 3. Main Apify Actor entry point
async def main() -> None:
    """Main function to run the LLM-based web scraper on Apify."""
    async with Actor:
        actor_input = await Actor.get_input() or {}
        user_query = actor_input.get('user_query', 'latest sports news')
        openai_api_key = actor_input.get('openai_api_key', '').strip()
        output_to_file = actor_input.get('output_to_file', False)

        if not openai_api_key:
            Actor.log.error("OPENAI_API_KEY is required.")
            await Actor.exit(1)

        # use LLM to get relevant URLs
        Actor.log.info(f"Using LLM to determine URLs from query: '{user_query}'")
        urls = await determine_relevant_urls(user_query, openai_api_key)

        if not urls:
            Actor.log.warning("No URLs to scrape, exiting.")
            await Actor.exit()

        # Scrape all relevant URLs
        Actor.log.info(f"Scraping headlines from {len(urls)} URLs...")
        async with AsyncClient(headers={"User-Agent": "LLM Based BBC Sport Web Scraper"}) as client:
            tasks = [scrape_sport_headlines_with_url(url, client) for url in urls]
            results = await asyncio.gather(*tasks)

        # Collect and structure scraped data
        extracted_headlines = []
        for data_list in results:
            extracted_headlines.extend(data_list)

        # Push each article to Apify dataset using mapped keys
        for article in extracted_headlines:
            await Actor.push_data({
                "Article Headline": article["headline"],
                "Article URL": article["url"],
                "BBC Sport Section": article["source"],
                "Scraped At (Readable)": article["scraped_at_readable"]
            })
