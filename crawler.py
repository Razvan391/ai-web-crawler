import asyncio
import os
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler
from google import genai

# Încarcă variabilele din .env
load_dotenv()

# Inițializare client Google Gemini
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

async def scrape_website(url: str) -> str:
    """Extrage conținutul Markdown dintr-o pagină web."""
    print(f"🔍 Crawling: {url}...")
    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(url=url)
        return result.markdown

def extract_insights_with_ai(markdown_content: str, prompt: str) -> str:
    """Procesează conținutul extras folosind Google Gemini."""
    print("🤖 Analizare conținut cu Gemini...")
    
    # Limita de text (Gemini are fereastră uriașă de context)
    truncated_content = markdown_content[:20000]

    full_prompt = f"""Ești un asistent specializat în structurarea datelor extrase din web. 
Extrage doar informațiile relevante cerute de utilizator.

Prompt utilizator: {prompt}

Conținut site:
{truncated_content}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=full_prompt,
    )
    return response.text

async def main():
    target_url = "https://news.ycombinator.com/"  # Exemplu: Hacker News
    user_prompt = "Extrage primele 5 știri principale și fă o scurtă sinteză a fiecăreia în limba română."

    # 1. Scraping
    markdown_data = await scrape_website(target_url)

    # 2. Extragere cu AI
    if markdown_data:
        summary = extract_insights_with_ai(markdown_data, user_prompt)
        print("\n--- Rezultat Gemini ---\n")
        print(summary)

if __name__ == "__main__":
    asyncio.run(main())