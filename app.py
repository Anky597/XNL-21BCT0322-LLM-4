import gradio as gr
import google.generativeai as genai
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import re
from PIL import Image
import base64
import os

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


# Prepare document texts and compute embeddings
doc_texts = [doc["text"] for doc in documents]
doc_embeddings = embedding_model.encode(doc_texts, convert_to_numpy=True).astype("float32")
dimension = doc_embeddings.shape[1]
nlist = 10  # Number of clusters

# Create a quantizer for L2 distance and build a FAISS IVF index
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
if not index.is_trained:
    index.train(doc_embeddings)
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
# Conversion: Company Name to Ticker Symbol
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
    pattern = r"(?:price of|stock price of|stock price for|quote for)\s+([A-Za-z]+)"
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
# NewsAPI Integration: Fetch Company News Headlines
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

def extract_company_name_for_sentiment(query):
    pattern = r"sentiment analysis (?:of|on)\s+([A-Za-z\s]+?)\s+stock"
    match = re.search(pattern, query, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def extract_company_name_for_news(query):
    pattern = r"news\s+(?:related\s+to|about|for)\s+((?:[A-Za-z]+\s*)+)(?:stock)?"
    match = re.search(pattern, query, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def get_stock_news(company, news_api_key, count=5):
    headlines = fetch_company_news(company, news_api_key, count)
    if not headlines:
        return f"No news headlines found for {company}."
    result = f"News headlines for {company}:\n"
    for headline in headlines:
        result += f"- {headline}\n"
    return result

# ---------------------------
# Analyze Uploaded Image/Graph Functionality
# ---------------------------
def analyze_image(file_path, gemini_api_key):
    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
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
# Process Chat: Main Function with Enhanced UI/UX
# ---------------------------
def process_chat(user_query, image_file):
    # Check if the user is using the Image Analysis tab
    if image_file is not None:
        file_path = image_file.name if hasattr(image_file, "name") else image_file
        return analyze_image(file_path, gemini_api_key)

    # Branch: Sentiment Analysis
    if "sentiment analysis" in user_query.lower():
        company = extract_company_name_for_sentiment(user_query)
        if company:
            return perform_sentiment_analysis(company, news_api_key, gemini_api_key)
        else:
            return "Unable to extract company name for sentiment analysis. Please include the company name clearly."

    # Branch: Fetch News Headlines
    if "news" in user_query.lower():
        company = extract_company_name_for_news(user_query)
        if company:
            return get_stock_news(company, news_api_key)
        else:
            return "Please specify the company name to fetch news."

    # Branch: Direct Stock Price Query
    ticker = extract_stock_symbol(user_query, gemini_api_key)
    if ticker and any(phrase in user_query.lower() for phrase in ["stock price", "price of", "quote for"]):
        return get_stock_quote_alpha_vantage(ticker, av_api_key)

    # Default: Retrieve context and use Gemini LLM for general queries
    context = retrieve_context(user_query, top_k=2)
    combined_prompt = f"Context:\n{context}\n\nUser Query: {user_query}\n\nAnswer:"
    answer = call_gemini_llm(combined_prompt, gemini_api_key)
    return answer

# ---------------------------
# API Keys (set these in your environment)
# ---------------------------
gemini_api_key = os.getenv("GEMINI_API_KEY")
av_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
news_api_key = os.getenv("NEWS_API_KEY")

# ---------------------------
# Enhanced Gradio Interface using Blocks and Tabs
# ---------------------------
with gr.Blocks(css=".gradio-container { max-width: 900px; margin: auto; }") as demo:
    gr.Markdown("# Financial Assistant Chatbot")
    gr.Markdown(
        """
        **Welcome!**  
        This chatbot can help you with a variety of financial queries:
        - **General Query:** Get context-aware answers.
        - **Stock Price:** Retrieve the latest stock quotes.
        - **News:** Fetch the latest financial news headlines.
        - **Sentiment Analysis:** Analyze sentiment based on news headlines.
        - **Image Analysis:** Upload a financial chart for detailed insights.
        """
    )
    
    with gr.Tabs():
        with gr.TabItem("Chat Query"):
            query_input = gr.Textbox(label="Enter your query", placeholder="Type your financial query here...")
            chat_output = gr.Textbox(label="Response")
        with gr.TabItem("Image Analysis"):
            image_input = gr.File(label="Upload image (PNG, JPG, JPEG)")
            image_output = gr.Textbox(label="Image Analysis Output")
    
    # Connect components: if an image is uploaded, use image tab; otherwise, use chat query
    def route_query(query, image):
        # If an image file is provided, process it; else process the text query.
        if image is not None:
            return process_chat(query, image)
        else:
            return process_chat(query, None)
    
    # Using the inputs from the Tabs (simulate a shared process)
    btn = gr.Button("Submit")
    output = gr.Textbox(label="Result", interactive=True)
    
    btn.click(route_query, inputs=[query_input, image_input], outputs=output)

# Launch the enhanced interface
demo.launch()
