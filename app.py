import asyncio
import os
import json
import streamlit as st
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler
from google import genai
from pydantic import BaseModel, Field

# Încarcă variabilele de mediu (.env)
load_dotenv()

# Setare pagină Streamlit
st.set_page_config(page_title="AI Web Crawler", page_icon="🤖", layout="wide")
st.title("🤖 AI-Powered Web Crawler & Extractor")
st.write("Introdu o adresă web și cerința ta pentru a extrage date structurate folosind **Google Gemini**.")

# Verificare API Key
gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    st.error("🔑 API Key-ul Gemini nu a fost găsit în fișierul `.env`!")
    st.stop()

client = genai.Client(api_key=gemini_key)

# 1. Definirea Structurii JSON dorite folosind Pydantic
class ExtractedArticle(BaseModel):
    title: str = Field(description="Titlul articolului sau al paginii")
    summary: str = Field(description="Un rezumat de 2-3 propoziții al conținutului")
    key_points: list[str] = Field(description="O listă cu principalele idei sau concluzii")
    sentiment: str = Field(description="Tonul textului: Pozitiv, Neutru sau Negativ")

# Funcția de Scraping
async def scrape_website(url: str) -> str:
    async with AsyncWebCrawler(verbose=False) as crawler:
        result = await crawler.arun(url=url)
        return result.markdown

# Funcția de Extragere AI Structurată
def extract_structured_data(markdown_content: str, prompt: str) -> ExtractedArticle:
    truncated_content = markdown_content[:15000]

    full_prompt = f"""Procesează următorul conținut web conform cerinței utilizatorului.
    
Cerință utilizator: {prompt}

Conținut pagina web:
{truncated_content}
"""

    # Folosim response_schema pentru a obliga Gemini să returneze structura Pydantic
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=full_prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": ExtractedArticle,
        },
    )
    
    # Parsează răspunsul JSON în modelul Pydantic
    return ExtractedArticle.model_validate_json(response.text)

# --- Interfața Grafică (UI) ---
url_input = st.text_input("🌐 URL-ul paginii web:", placeholder="https://news.ycombinator.com/")
prompt_input = st.text_area("📝 Ce informații dorești să extragi?", value="Extrage principalele noutăți și rezumă-le.")

if st.button("🚀 Pornește Crawler-ul", type="primary"):
    if not url_input:
        st.warning("Te rugăm să introduci un URL valid.")
    else:
        with st.spinner("🔍 Extragere conținut web și analiză AI în curs..."):
            try:
                # Rulare scraper asincron
                markdown_data = asyncio.run(scrape_website(url_input))
                
                if markdown_data:
                    # Rulare AI
                    result_data = extract_structured_data(markdown_data, prompt_input)
                    
                    st.success("✅ Extragere finalizată cu succes!")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader(f"📌 {result_data.title}")
                        st.write(f"**Rezumat:** {result_data.summary}")
                        st.write("**Idei Principale:**")
                        for point in result_data.key_points:
                            st.markdown(f"* {point}")
                            
                    with col2:
                        st.metric("Ton/Sentiment", result_data.sentiment)
                        
                        # Conversie în JSON pentru descărcare
                        json_str = result_data.model_dump_json(indent=2)
                        st.download_button(
                            label="📥 Descarcă Rezultat (JSON)",
                            data=json_str,
                            file_name="extracted_data.json",
                            mime="application/json"
                        )
                else:
                    st.error("Nu s-a putut extrage conținut de la acest URL.")
            except Exception as e:
                st.error(f"A apărut o eroare: {str(e)}")