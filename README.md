# 🤖 AI Web Crawler & Data Extractor

Un crawler web inteligent și asincron construit în Python, capabil să extragă conținut din site-uri web și să îl proceseze structurat folosind modele LLM (**Google Gemini**).

## 🚀 Caracteristici
* **Scraping Asincron:** Utilizează `Crawl4AI` pentru a transforma HTML-ul paginilor în Markdown curat.
* **Procesare AI:** Folosește API-ul `Google Gemini` (modelul `gemini-2.5-flash`) pentru sinteză și extragere de informații specifice.
* **Securitate:** Management securizat al cheilor API prin fișiere `.env`.

## 🛠️ Tehnologii Utilizate
* **Python 3.10+**
* **Crawl4AI** - Web scraping optimizat pentru LLM
* **Google GenAI SDK** - Integrare cu Google Gemini
* **python-dotenv** - Gestionare variabile de mediu

## 📦 Instalare și Rulare

1. **Clonează depozitul:**
   ```bash
   git clone [https://github.com/Razvan391/ai-web-crawler.git](https://github.com/Razvan391/ai-web-crawler.git)
   cd ai-web-crawler