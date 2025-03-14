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

## Summary

This project integrates multiple data sources and processing techniques to deliver a responsive financial assistant chatbot.
