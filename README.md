---
title: XNL Task Chatboat
emoji: 📊
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: 5.21.0
app_file: app.py
pinned: false
---

![Alt text](roject_Wireframe.png)

# Financial Assistant Chatbot

This project is a Gradio-based web application designed to answer financial queries by combining multiple data sources and processing methods. It integrates:

* **Gradio UI:** For user interaction.
* **Gemini LLM (Google Generative AI):** To generate responses.
* **Alpha Vantage API:** For real-time stock data.
* **NewsAPI:** To fetch the latest company news.
* **FAISS with SentenceTransformer:** For context retrieval from pre-defined financial documents.
* **Branching logic:** To handle various query types (image analysis, sentiment analysis, news lookup, direct stock quotes, and general questions).

## Setup

### Prerequisites

* Python 3.8+
* Required libraries:
    * `gradio`
    * `google.generativeai`
    * `faiss` (or `faiss-cpu` depending on your environment)
    * `numpy`
    * `sentence-transformers`
    * `requests`
    * `Pillow`
    * `base64`
    * `re`

### Installation

1.  Clone the repository or place the project code in your working directory.
2.  Install dependencies using pip:

    ```bash
    pip install gradio google-generativeai faiss-cpu numpy sentence-transformers requests pillow
    ```

3.  Set up API Keys:
    * Replace the placeholder API keys in the code (or set them as environment variables) for:
        * Gemini LLM: `gemini_api_key`
        * Alpha Vantage: `av_api_key`
        * NewsAPI: `news_api_key`
    * For example, you can export keys in your shell or configure them in a `.env` file and load them via Python.

4.  Run the application:

    ```bash
    python your_script.py
    ```

    This will launch the Gradio interface locally where you can interact with the chatbot.

## API Usage

### Gemini LLM API

* **Purpose:** Handles various tasks such as generating natural language responses, converting company names to ticker symbols, and performing prompt-based analyses (sentiment, image interpretation).
* **Usage in Code:**
    * The function `call_gemini_llm(prompt, gemini_api_key, model="gemini-2.0-flash")` configures the Gemini API using the provided key, constructs a prompt, and returns the generated text.
    * It’s used across multiple modules (e.g., ticker conversion, context-based general query responses, and image analysis).
* **Notes on Fine-Tuning:** While direct fine-tuning of Gemini LLM may not be exposed, prompt engineering is used to adapt responses. For specialized behavior, consider adjusting prompt templates or using additional context in the prompt.

### Alpha Vantage API

* **Purpose:** Fetches real-time stock data such as current price, change, and change percent for a given ticker symbol.
* **Usage in Code:**
    * The function `get_stock_quote_alpha_vantage(symbol, av_api_key)` builds the API request URL, sends the query, and parses the JSON response.
    * The function extracts key fields from the “Global Quote” returned by Alpha Vantage.
* **Important Considerations:** Ensure that you monitor API usage quotas and handle error responses gracefully. The code includes basic error handling to capture exceptions or missing data.

### NewsAPI

* **Purpose:** Retrieves recent news headlines about a specified company, which can be used for sentiment analysis or just to display current news.
* **Usage in Code:**
    * The function `fetch_company_news(company, news_api_key, count=5)` constructs the query parameters (including language, sorting, and count), sends the request, and extracts headlines from the returned JSON.
    * These headlines are then used by `perform_sentiment_analysis` to generate a sentiment summary via Gemini LLM.

## LLM Fine-Tuning and Prompt Engineering

### Fine-Tuning vs. Prompt Engineering

* **Fine-Tuning:** Direct fine-tuning of large language models like Gemini is typically handled on the backend by the provider. In this project, we rely on prompt engineering to steer the model’s output.
* **Prompt Engineering Techniques:**
    * **Role Specification:** Prompts often begin with a statement like “You are a financial analyst…” to set the context.
    * **Clear Instructions:** Each branch of the logic (image analysis, sentiment analysis, ticker conversion) constructs specific prompts to direct the LLM.
    * **Context Inclusion:** For general queries, relevant financial documents are fetched from a vector database and included in the prompt to guide the LLM’s response.

### Customizing Prompts

* **Ticker Conversion Example:** The prompt instructs the LLM to output only the stock ticker symbol, ensuring concise conversion.
* **Image Analysis:** The image file is converted to a base64 string and inserted into a detailed prompt that explains the type of analysis required.
* **Sentiment Analysis:** A prompt that lists recent news headlines is built to have the LLM provide a sentiment summary with reasoning.

## Context Retrieval via Vector Database

### Embedding Model Setup

* **Tool:** SentenceTransformer model “all-MiniLM-L6-v2” is used to compute text embeddings.
* **Purpose:** Converts financial documents into dense vector representations for similarity search.

### FAISS Integration

* **Index Creation:**
    * The embeddings of all context documents are computed and stored in a NumPy array.
    * An IVF (Inverted File) index is created using FAISS for efficient approximate nearest neighbor search.
    * The index is trained on these embeddings and then used to add vectors for search.
* **Context Retrieval Function:**
    * `retrieve_context(query, top_k=2)` computes an embedding for the query, searches the FAISS index, and retrieves the top-k most similar documents.
    * This retrieved context is then appended to the prompt for the LLM, helping it generate a more informed answer.

## Fallback Mechanisms

### Branching Logic

The application’s main function `process_chat(user_query, image_file)` uses a series of conditional checks to determine which branch of logic to follow:

* **Image/Graph Analysis Branch:** Triggered if an image is provided or if the query includes keywords like “upload image” or “analyze chart.”
* **Sentiment Analysis Branch:** If the query mentions “sentiment analysis,” the function attempts to extract a company name.
* **News Headlines Branch:** Checks if the query contains “news.”
* **Direct Stock Quote Branch:** Looks for phrases such as “stock price” or “quote for.”
* **General Query Branch (Fallback):** When none of the above conditions are met, the application defaults to the general query process.

### Error Handling and Default Responses

* **API Error Handling:** Each API call (Gemini LLM, Alpha Vantage, NewsAPI) is wrapped in `try-except` blocks to catch and return descriptive error messages.
* **Graceful Fallback:** If specific branches (like ticker extraction or sentiment analysis) fail to extract necessary information, the system responds with a clear fallback message instructing the user on how to modify their query.

## Development Code (Base Fine Tune LLM)

```python
import google.generativeai as genai
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import re
from PIL import Image
import base64

# ---------------------------
# Setup the embedding model
# ---------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------
# Improved financial context documents (vector database)
# ---------------------------
documents = [
    # Existing entries
    {
        "text": "Tesla (TSLA) saw a 12% increase in Q1 2025 after beating revenue expectations. Investor optimism has grown with rising production numbers.",
        "metadata": {"company": "Tesla", "metric": "stock price", "sentiment": "positive", "date": "2025-04-15"}
    },
    {
        "text": "Apple (AAPL) reported a slight dip in its share price following mixed quarterly results, sparking a cautious outlook among analysts.",
        "metadata": {"company": "Apple", "metric": "stock price", "sentiment": "negative", "date": "2025-03-30"}
    },
    {
        "text": "The NASDAQ Composite hit a milestone, reaching 15000 points driven by strong performances in the technology sector.",
        "metadata": {"index": "NASDAQ Composite", "metric": "index value", "sentiment": "positive", "date": "2025-04-10"}
    },
    {
        "text": "Inflation concerns continue as consumer prices rose by 3.2% in March 2025, leading to uncertainty in the bond markets.",
        "metadata": {"metric": "inflation rate", "sentiment": "negative", "date": "2025-03-31"}
    },
    {
        "text": "Gold has maintained its status as a safe-haven asset, with prices stabilizing amidst ongoing global economic uncertainties.",
        "metadata": {"asset": "Gold", "metric": "price", "sentiment": "neutral", "date": "2025-04-05"}
    },
    {
        "text": "Amazon (AMZN) has seen its stock surge by 8% following the announcement of new innovations in its cloud computing services.",
        "metadata": {"company": "Amazon", "metric": "stock price", "sentiment": "positive", "date": "2025-04-12"}
    },
    {
        "text": "Market analysts remain divided over the outlook for the S&P 500, citing concerns about potential market corrections amid high valuations.",
        "metadata": {"index": "S&P 500", "metric": "index value", "sentiment": "neutral", "date": "2025-04-08"}
    },
    {
        "text": "Emerging market currencies have experienced volatility, with significant fluctuations observed due to political unrest in several regions.",
        "metadata": {"asset": "emerging market currencies", "risk": "political instability", "sentiment": "negative", "date": "2025-04-09"}
    },
    {
        "text": "Compound interest remains a fundamental concept in finance, underscoring the benefits of early and consistent investments for long-term wealth accumulation.",
        "metadata": {"concept": "compound interest", "sentiment": "positive"}
    },
    {
        "text": "New regulations in the European financial sector are expected to reshape market dynamics, with a focus on enhancing transparency and investor protection.",
        "metadata": {"region": "Europe", "regulator": "European Commission", "sentiment": "positive", "date": "2025-04-07"}
    },
    
    # New entries
    {
        "text": "Tesla (TSLA) stock has faced significant challenges in Q1 2025, with deliveries tracking approximately 31,000 units lower than Q1 2024. Wall Street analysts have revised delivery estimates downward to around 356,000 vehicles.",
        "metadata": {"company": "Tesla", "metric": "deliveries", "sentiment": "negative", "date": "2025-03-14"}
    },
    {
        "text": "Vector databases are transforming financial analysis by enabling efficient processing of unstructured data for fraud detection, risk analysis, and pattern recognition in market trends.",
        "metadata": {"technology": "vector databases", "industry": "finance", "application": "risk analysis", "sentiment": "positive", "date": "2025-03-01"}
    },
    {
        "text": "Bitcoin has experienced increased institutional adoption in 2025, with several major banks now offering cryptocurrency custody services to their wealth management clients.",
        "metadata": {"asset": "Bitcoin", "metric": "institutional adoption", "sentiment": "positive", "date": "2025-04-02"}
    },
    {
        "text": "The Federal Reserve has maintained its cautious approach to interest rates, signaling potential cuts later in 2025 if inflation continues to moderate toward the 2% target.",
        "metadata": {"institution": "Federal Reserve", "metric": "interest rates", "sentiment": "neutral", "date": "2025-03-25"}
    },
    {
        "text": "ESG-focused investment funds have seen record inflows in early 2025, reflecting growing investor demand for sustainability-oriented financial products.",
        "metadata": {"investment_strategy": "ESG", "metric": "fund inflows", "sentiment": "positive", "date": "2025-04-01"}
    },
    {
        "text": "Commercial real estate continues to face headwinds in 2025, with office vacancies remaining elevated as companies maintain flexible work arrangements post-pandemic.",
        "metadata": {"asset": "commercial real estate", "metric": "vacancies", "sentiment": "negative", "date": "2025-03-20"}
    },
    {
        "text": "Small-cap stocks have outperformed larger indices in Q1 2025, suggesting investors are finding value in smaller companies amid high valuations in tech giants.",
        "metadata": {"asset_class": "small-cap stocks", "metric": "performance", "sentiment": "positive", "date": "2025-04-03"}
    },
    {
        "text": "Venture capital investments in AI startups have reached $45 billion in Q1 2025, representing a 30% increase year-over-year as the technology's commercial applications expand.",
        "metadata": {"sector": "artificial intelligence", "metric": "venture capital", "sentiment": "positive", "date": "2025-04-10"}
    },
    {
        "text": "Global supply chain disruptions have eased in early 2025, though regional conflicts continue to create bottlenecks in certain industries and trade routes.",
        "metadata": {"economic_factor": "supply chain", "metric": "disruptions", "sentiment": "mixed", "date": "2025-03-28"}
    },
    {
        "text": "Financial advisors are increasingly recommending dynamic withdrawal strategies for retirement planning, moving away from the traditional 4% rule due to changing market conditions.",
        "metadata": {"financial_planning": "retirement", "concept": "withdrawal strategies", "sentiment": "neutral", "date": "2025-02-15"}
    },
    {
        "text": "Private equity firms have accumulated record levels of dry powder in 2025, with over $2.3 trillion available for investments as they wait for more favorable valuation environments.",
        "metadata": {"investment_type": "private equity", "metric": "dry powder", "sentiment": "neutral", "date": "2025-03-15"}
    },
    {
        "text": "Lithium prices have stabilized after a volatile 2024, as new mining capacity comes online to meet the growing demand from electric vehicle manufacturers.",
        "metadata": {"commodity": "lithium", "metric": "price", "sentiment": "positive", "date": "2025-04-05"}
    },
    {
        "text": "Embedded finance solutions are gaining traction across industries, with non-financial companies increasingly integrating payment and lending services into their customer experiences.",
        "metadata": {"industry": "fintech", "innovation": "embedded finance", "sentiment": "positive", "date": "2025-03-22"}
    },
    {
        "text": "Dividend-yielding stocks have seen renewed interest in Q1 2025 as investors seek income amid persistent inflation and relatively high interest rates.",
        "metadata": {"investment_strategy": "dividend investing", "metric": "investor interest", "sentiment": "positive", "date": "2025-03-31"}
    },
    {
        "text": "The VIX index, a key measure of market volatility, has averaged 18 points in Q1 2025, indicating relatively calm market conditions despite ongoing economic uncertainties.",
        "metadata": {"indicator": "VIX", "metric": "volatility", "sentiment": "positive", "date": "2025-04-01"}
    },
    {
        "text": "New trade agreements between ASEAN nations and the European Union are expected to boost economic activity in both regions, with implementation planned for late 2025.",
        "metadata": {"economic_policy": "trade agreements", "regions": ["ASEAN", "European Union"], "sentiment": "positive", "date": "2025-03-18"}
    },
    {
        "text": "Q1 2025 earnings season has begun with mixed results, as 65% of S&P 500 companies reporting so far have exceeded analyst expectations despite challenging economic conditions.",
        "metadata": {"financial_reporting": "earnings", "index": "S&P 500", "sentiment": "mixed", "date": "2025-04-14"}
    },
    {
        "text": "Art and collectibles have shown strong performance as alternative investments in early 2025, with auction records broken across several categories as investors seek diversification.",
        "metadata": {"investment_type": "alternative", "asset": "art and collectibles", "sentiment": "positive", "date": "2025-03-10"}
    },
    {
        "text": "New AI-powered risk management tools are allowing financial institutions to better predict and mitigate potential market disruptions through enhanced scenario modeling.",
        "metadata": {"financial_practice": "risk management", "technology": "AI", "sentiment": "positive", "date": "2025-02-28"}
    },
    {
        "text": "Regional banks have shown improved performance in Q1 2025 after implementing cost-cutting measures and digital transformation initiatives to enhance operational efficiency.",
        "metadata": {"industry": "banking", "segment": "regional banks", "metric": "performance", "sentiment": "positive", "date": "2025-04-08"}
    },
    {
        "text": "Microsoft (MSFT) has increased its dividend by 12% for 2025, reflecting strong cash flow generation and commitment to shareholder returns.",
        "metadata": {"company": "Microsoft", "metric": "dividend", "sentiment": "positive", "date": "2025-03-19"}
    },
    {
        "text": "Oil prices have fluctuated between $70-$85 per barrel in Q1 2025 as OPEC+ production adjustments attempt to balance global supply and demand dynamics.",
        "metadata": {"commodity": "oil", "metric": "price", "sentiment": "neutral", "date": "2025-04-05"}
    },
    {
        "text": "The healthcare sector has underperformed broader market indices in early 2025 amid concerns about potential regulatory changes affecting drug pricing.",
        "metadata": {"sector": "healthcare", "metric": "performance", "sentiment": "negative", "date": "2025-03-29"}
    },
    {
        "text": "Corporate bond yields have declined by 25 basis points on average during Q1 2025, reflecting improved credit conditions and strong investor demand for fixed income.",
        "metadata": {"asset": "corporate bonds", "metric": "yields", "sentiment": "positive", "date": "2025-04-01"}
    },
    {
        "text": "The Japanese yen has strengthened against major currencies following the Bank of Japan's decision to gradually normalize its monetary policy stance.",
        "metadata": {"currency": "Japanese yen", "metric": "exchange rate", "sentiment": "positive", "date": "2025-03-26"}
    }
]

# ---------------------------
# Prepare document texts and compute embeddings
# ---------------------------
doc_texts = [doc["text"] for doc in documents]
doc_embeddings = embedding_model.encode(doc_texts, convert_to_numpy=True).astype("float32")
dimension = doc_embeddings.shape[1]
nlist = 10  # Number of clusters; can be tuned based on dataset size

# Create a quantizer for L2 distance (similar to IndexFlatL2)
quantizer = faiss.IndexFlatL2(dimension)
# Use an IVF index for faster approximate nearest neighbor search
index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)

# Train the IVF index on our document embeddings (required before adding vectors)
if not index.is_trained:
    index.train(doc_embeddings)

# Add the embeddings to the index
index.add(doc_embeddings)

# ---------------------------
# Gemini LLM Call Function
# ---------------------------
def call_gemini_llm(prompt, gemini_api_key, model="gemini-2.0-flash"):
    try:
        genai.configure(api_key=gemini_api_key)
        model_obj = genai.GenerativeModel(model)
        response = model_obj.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {e}"

# ---------------------------
# Conversion: Company Name to Ticker Symbol using Gemini LLM
# ---------------------------
def convert_company_to_ticker(company_name, gemini_api_key):
    conversion_prompt = (
        f"Convert the following company name into its stock ticker symbol. "
        f"Only output the ticker symbol without any additional text.\n\n"
        f"Company Name: {company_name}"
    )
    ticker = call_gemini_llm(conversion_prompt, gemini_api_key)
    return ticker.strip().upper()

def extract_stock_symbol(query, gemini_api_key):
    pattern = r"(?:price of|stock price for|quote for)\s+([A-Za-z]+)"
    match = re.search(pattern, query, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        ticker = convert_company_to_ticker(candidate, gemini_api_key)
        return ticker
    return None

# ---------------------------
# Retrieve Context from Vector Database
# ---------------------------
def retrieve_context(query, top_k=2):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    retrieved_texts = [doc_texts[i] for i in indices[0] if i < len(doc_texts)]
    return "\n".join(retrieved_texts)

# ---------------------------
# Alpha Vantage: Fetch Real-Time Stock Data
# ---------------------------
def get_stock_quote_alpha_vantage(symbol, av_api_key):
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": av_api_key
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        if "Global Quote" in data and data["Global Quote"]:
            quote = data["Global Quote"]
            price = quote.get("05. price", "N/A")
            change = quote.get("09. change", "N/A")
            change_percent = quote.get("10. change percent", "N/A")
            return f"Alpha Vantage: The latest quote for {symbol} is ${price}, change: {change} ({change_percent})."
        else:
            return f"Alpha Vantage: Unable to retrieve stock data for {symbol}."
    except Exception as e:
        return f"Error retrieving stock data for {symbol}: {e}"

# ---------------------------
# NewsAPI Integration: Fetch Company News Headlines (English only)
# ---------------------------
def fetch_company_news(company, news_api_key, count=5):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": company,
        "sortBy": "publishedAt",
        "pageSize": count,
        "language": "en",
        "apiKey": news_api_key
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if "articles" in data:
            headlines = [article["title"] for article in data["articles"] if "title" in article]
            return headlines
        else:
            return []
    except Exception as e:
        return []

# ---------------------------
# Sentiment Analysis using News Headlines and Gemini LLM
# ---------------------------
def perform_sentiment_analysis(company, news_api_key, gemini_api_key):
    headlines = fetch_company_news(company, news_api_key, count=5)
    if not headlines:
        return f"No news headlines found for {company} for sentiment analysis."
    prompt = f"Analyze the sentiment of the following news headlines about {company}:\n"
    for headline in headlines:
        prompt += f"- {headline}\n"
    prompt += "\nProvide a summary sentiment analysis (positive, negative, or neutral) along with reasons."
    analysis = call_gemini_llm(prompt, gemini_api_key)
    return analysis

# ---------------------------
# Extract Company Name for Sentiment Analysis
# ---------------------------
def extract_company_name_for_sentiment(query):
    pattern = r"sentiment analysis (?:of|on)\s+([A-Za-z\s]+?)\s+stock"
    match = re.search(pattern, query, re.IGNORECASE)
    if match:
        company = match.group(1).strip()
        return company
    return None

# ---------------------------
# Extract Company Name for News Query with Improved Regex
# ---------------------------
def extract_company_name_for_news(query):
    pattern = r"news\s+(?:related\s+to|about|for)\s+((?:[A-Za-z]+\s*)+)(?:stock)?"
    match = re.search(pattern, query, re.IGNORECASE)
    if match:
        company = match.group(1).strip()
        return company
    return None

# ---------------------------
# Get Stock News Function using NewsAPI
# ---------------------------
def get_stock_news(company, news_api_key, count=5):
    headlines = fetch_company_news(company, news_api_key, count)
    if not headlines:
        return f"No news headlines found for {company}."
    result = f"News headlines for {company}:\n"
    for headline in headlines:
        result += f"- {headline}\n"
    return result

# ---------------------------
# Analyze Uploaded Image/Graph Functionality (Direct Image Upload to Gemini LLM)
# ---------------------------
def analyze_image(file_path, gemini_api_key):
    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        # Updated prompt tailored for financial market charts
        prompt = (
            "You are a financial analyst with expertise in stock market trends and financial charts. "
            "The following image (provided as a base64-encoded string) represents a financial graph—such as an S&P 500 growth chart. "
            "Please analyze the chart and provide detailed insights on trends, key performance metrics, and any notable fluctuations related to the market."
            f"\nImage (base64): {encoded_string}"
        )
        analysis = call_gemini_llm(prompt, gemini_api_key)
        return analysis
    except Exception as e:
        return f"Error processing image: {e}"

# ---------------------------
# Chatbot Integration (with branches for news, image analysis, etc.)
# ---------------------------
def chatbot(gemini_api_key, av_api_key, news_api_key):
    print("Welcome to the Financial Assistant Chatbot (RAG MVP).")
    print("Type 'quit' to exit.\n")
    
    # Extended image trigger phrases for better NLP detection
    image_triggers = ["upload image", "analyze image", "upload graph", "analyze graph", "upload chart", "analyze chart"]
    
    while True:
        user_query = input("User: ").strip()
        if user_query.lower() == "quit":
            print("Exiting chatbot. Goodbye!")
            break

        # Branch: Image/Graph Analysis
        if any(trigger in user_query.lower() for trigger in image_triggers):
            file_path = input("Please enter the file path of the image/graph: ").strip()
            image_analysis = analyze_image(file_path, gemini_api_key)
            print("Chatbot:", image_analysis, "\n")
            continue

        # Branch: Sentiment Analysis if the query mentions "sentiment analysis"
        if "sentiment analysis" in user_query.lower():
            company = extract_company_name_for_sentiment(user_query)
            if company:
                sentiment_result = perform_sentiment_analysis(company, news_api_key, gemini_api_key)
                print("Chatbot:", sentiment_result, "\n")
                continue
            else:
                print("Chatbot: Unable to extract company name for sentiment analysis.\n")
                continue

        # Branch: Fetch news headlines using the NewsAPI if the query mentions "news"
        if "news" in user_query.lower():
            company = extract_company_name_for_news(user_query)
            if company:
                news_result = get_stock_news(company, news_api_key)
                print("Chatbot:", news_result, "\n")
                continue
            else:
                print("Chatbot: Please specify the company name to fetch news.\n")
                continue

        # Regular flow: Retrieve context and check for stock data requests
        context = retrieve_context(user_query, top_k=2)
        ticker = extract_stock_symbol(user_query, gemini_api_key)
        if ticker:
            stock_info = get_stock_quote_alpha_vantage(ticker, av_api_key)
            context += "\n" + stock_info
        combined_prompt = f"Context:\n{context}\n\nUser Query: {user_query}\n\nAnswer:"
        answer = call_gemini_llm(combined_prompt, gemini_api_key)
        print("Chatbot:", answer, "\n")

# ---------------------------
# Main Execution: Replace with your actual API keys
# ---------------------------
gemini_api_key = # Your Gemini API key
av_api_key = # Your Alpha Vantage API key
news_api_key = # your api key

chatbot(gemini_api_key, av_api_key, news_api_key)
