import asyncio
import os
import chromadb
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_api_key)

# ChromaDB salvează datele local într-un folder numit 'chroma_db'
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="web_knowledge_base")

async def scrape_website(url: str) -> str:
    """Extrage conținutul Markdown dintr-un site."""
    print(f"🌐 Scraping: {url}...")
    async with AsyncWebCrawler(verbose=False) as crawler:
        result = await crawler.arun(url=url)
        return result.markdown

def get_text_embedding(text: str) -> list[float]:
    """Generat vectori (embeddings) folosind modelul text-embedding-004 de la Google."""
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text,
    )
    return response.embedding.values

def index_content_to_vector_db(markdown_text: str, source_url: str):
    """Împarte textul în chunks și le stochează în baza de date vectorială."""
    print("✂️ Împărțire conținut în chunk-uri și vectorizare...")
    
    # Împărțim textul în bucăți de ~1000 caractere cu o suprapunere de 200
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(markdown_text)

    # Introducem fiecare chunk în ChromaDB
    for i, chunk in enumerate(chunks):
        embedding = get_text_embedding(chunk)
        chunk_id = f"{source_url}_chunk_{i}"
        
        collection.upsert(
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{"source": source_url, "chunk_index": i}],
            ids=[chunk_id]
        )
    print(f"✅ Am salvat {len(chunks)} vectori în ChromaDB pentru URL-ul: {source_url}")

def query_rag_system(user_query: str, top_k: int = 3) -> str:
    """Caută cele mai relevante chunk-uri și generează răspunsul cu Gemini."""
    print(f"🔍 Căutare vectorială pentru întrebarea: '{user_query}'...")
    
    # Vectorizăm întrebarea utilizatorului
    query_embedding = get_text_embedding(user_query)
    
    # Căutăm cele mai apropiate k bucăți de text din baza de date
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    retrieved_chunks = results['documents'][0]
    context = "\n---\n".join(retrieved_chunks)
    
    # Trimiterea contextului selectat către Gemini
    prompt = f"""Ești un asistent inteligent RAG. Răspunde la întrebare DOAR pe baza contextului furnizat mai jos.
Dacă informația nu se află în context, spune clar că nu ai găsit date suficiente pe site.

Context extras din baza vectorială:
{context}

Întrebare utilizator: {user_query}
Răspuns:
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

async def main():
    target_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    
    # 1. Pasul de Ingestie (Scraping -> Chunking -> VectorDB)
    raw_markdown = await scrape_website(target_url)
    if raw_markdown:
        index_content_to_vector_db(raw_markdown, target_url)
    
    # 2. Pasul de RAG (Interogare inteligentă)
    print("\n--- TESTARE SISTEM RAG ---")
    query = "What are the main applications of artificial intelligence mentioned?"
    answer = query_rag_system(query)
    
    print("\n🤖 Răspuns Generat prin RAG:")
    print(answer)

if __name__ == "__main__":
    asyncio.run(main())