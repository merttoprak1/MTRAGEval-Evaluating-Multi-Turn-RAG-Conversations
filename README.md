# Modular RAG Chatbot

A modular Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, LangChain, and FAISS. This application supports multiple LLM providers, including Google Gemini, OpenAI, and local models via LM Studio and Ollama.

## Features

- **Multi-Provider Support**: Switch between Google Gemini, OpenAI, and local LLMs (Ollama/LM Studio).
- **RAG Capabilities**: Upload JSON/JSONL documents to chat with your data.
- **Efficient Vector Store**: Uses FAISS (local file-based) for fast document retrieval and compatibility.
- **Session Management**: Create, rename, and delete chat sessions.
- **Database Inspector**: Built-in tools to inspect chat history and sessions.

## Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.com/) (optional, for local embeddings/inference)
- [LM Studio](https://lmstudio.ai/) (optional, for local inference)

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the application:**

    ```bash
    streamlit run app.py
    ```

2.  **Open your browser:**
    The application will typically run at `http://localhost:8501`.

## Configuration

### Google Gemini (Recommended)

1.  Select **Gemini** as the LLM Provider in the sidebar.
2.  Enter your **Google API Key**.
3.  (Optional) Specify a model name (default: `gemini-flash-latest`).
4.  For Embeddings, select **Gemini** in the Embedding Configuration section (reuses API Key).

### OpenAI

1.  Select **OpenAI** as the LLM Provider in the sidebar.
2.  Enter your **OpenAI API Key**.
3.  (Optional) Specify a model name (default: `gpt-3.5-turbo`).

### Local LLMs

You can run this chatbot with local models using either LM Studio or Ollama.

#### LM Studio

1.  **Download and Install**: Get [LM Studio](https://lmstudio.ai/).
2.  **Load a Model**: Download and load a model (e.g., Llama 3, Mistral).
3.  **Start Server**:
    - Go to the "Local Server" tab (double-headed arrow icon).
    - Ensure the server port is `1234` (default).
    - Click **Start Server**.
4.  **Configure Chatbot**:
    - In the chatbot sidebar, select **Local** as the LLM Provider.
    - Set **Local LLM Base URL** to `http://localhost:1234/v1`.
    - Set **Model Name** to the model you loaded (e.g., `QuantFactory/Meta-Llama-3-8B-Instruct-GGUF`).

#### Ollama

1.  **Download and Install**: Get [Ollama](https://ollama.com/).
2.  **Pull a Model**: Run `ollama pull llama3` (or your preferred model) in your terminal.
3.  **Start Ollama**: Ensure Ollama is running (usually runs in the background).
4.  **Configure Chatbot**:
    - **For Chat**:
        - Select **Local** as the LLM Provider.
        - Set **Local LLM Base URL** to `http://localhost:11434/v1`.
        - Set **Model Name** to your pulled model (e.g., `llama3`).
    - **For Embeddings**:
        - Select **Local (Ollama)** as the Embedding Provider.
        - Set **Embedding Base URL** to `http://localhost:11434`.
        - Set **Embedding Model** to a text embedding model (e.g., `nomic-embed-text` or `mxbai-embed-large`). *Note: You may need to pull these models first via `ollama pull nomic-embed-text`.*

## Data Ingestion

1.  Upload a `.json` or `.jsonl` file in the sidebar.
2.  Click **Process File** to ingest documents into the FAISS vector store.
    - *Note: If using Gemini Embeddings, processing is rate-limited (10 docs / 5s) to respect free tier quotas.*
3.  Start chatting!
