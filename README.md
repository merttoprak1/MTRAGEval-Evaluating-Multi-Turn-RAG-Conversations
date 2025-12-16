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

---

## RAG Playground

The RAG Playground is a comprehensive testing and evaluation environment for RAG pipelines. It provides three task types for different evaluation scenarios.

### Tasks Overview

| Task | Pipeline | Use Case |
|------|----------|----------|
| **Task A** | Query → Retrieval | Test retrieval quality |
| **Task B** | Query → Retrieval → Generation | Test end-to-end RAG |
| **Task C** | Query → Rewrite → Retrieval → Generation | Test full pipeline with query optimization |

### Quick Start

1. **Configure Components** (Sidebar):
   - Select LLM Provider (OpenAI/Gemini/Local) and enter API key
   - Select Embedding Provider and model
   - Choose Vector Database (FAISS/Chroma/Pinecone)
   - Set Top-K retrieval count

2. **Upload Documents**:
   - Upload JSON/JSONL file in sidebar
   - Click "Process File" to ingest into vector store

3. **Run Pipeline** (RAG Playground tab):
   - Select Task (A, B, or C)
   - Configure task-specific options
   - Enter test query
   - Click "▶️ Run Pipeline"

4. **View Results**:
   - Retrieval results with scores and metadata
   - Generated answer (Task B, C)
   - Debug panel with full run details

### Task A: Retrieval Only

**Purpose**: Evaluate retrieval quality in isolation.

**Flow**:
```
Query → Embedding → Vector Search → Retrieved Documents
```

**Configuration**:
- Retrieval Top-K
- Search Type (similarity)

**Output**:
- Retrieved chunks with similarity scores
- Source metadata (ID, source, title)
- Retrieval time

### Task B: Retrieval + Generation

**Purpose**: Test end-to-end RAG with answer generation.

**Flow**:
```
Query → Embedding → Vector Search → Context Building → LLM Generation → Answer
```

**Configuration**:
- All Task A settings
- LLM Model selection
- Prompt template (Default RAG, Concise, Detailed, Custom)
- Temperature & Max Tokens

**Output**:
- Retrieved chunks with scores
- Generated answer
- Debug: Full prompt sent to LLM

### Task C: Full RAG Pipeline

**Purpose**: Complete RAG with query rewriting for improved retrieval.

**Flow**:
```
Query → Query Rewrite → Embedding → Vector Search → Context Building → LLM Generation → Answer
```

**Configuration**:
- All Task B settings
- Rewrite method (LLM-based, Rule-based, Hybrid)
- Custom rewrite prompt (optional)

**Output**:
- Original vs Rewritten query
- Retrieved chunks with scores
- Generated answer
- Pipeline summary with all stages

---

## Evaluation Playground

The Evaluation Playground allows you to evaluate your RAG pipeline performance with various metrics.

### Available Evaluation Scripts

#### Task A - Retrieval Metrics
| Script | Metrics | Description |
|--------|---------|-------------|
| Retrieval Accuracy | Hit Rate, MRR | Measures if relevant docs are retrieved |
| Retrieval Precision | Precision@K, Recall@K | Ratio of relevant docs in results |
| Semantic Similarity | Avg/Min/Max Similarity | Query-document similarity |

#### Task B - Generation Metrics
| Script | Metrics | Description |
|--------|---------|-------------|
| Answer Correctness | Exact Match, F1, BLEU | Compares to ground truth |
| Faithfulness | Faithfulness Score | Grounding in context |
| Answer Relevance | Relevance Score | Answer relevance to question |
| Context Relevance | Context Precision/Recall | Retrieved context quality |

#### Task C - Full Pipeline Metrics
| Script | Metrics | Description |
|--------|---------|-------------|
| End-to-End Accuracy | E2E Accuracy, E2E F1 | Full pipeline accuracy |
| Query Rewrite Quality | Hit Rate Improvement | Rewrite effectiveness |
| RAGAS Score | Composite Score | Comprehensive RAG evaluation |
| Latency Analysis | Stage timings | Performance analysis |

### Running Evaluations

1. **Select Task** to evaluate
2. **Select Evaluation Scripts** (one or more)
3. **Provide Input**:
   - Manual: Enter query + ground truth (optional)
   - Upload: JSON dataset with queries and ground truth
   - Last Run: Use previous pipeline results
4. **Configure**:
   - LLM-as-Judge settings
   - Similarity thresholds
   - Export options
5. **Click "🚀 Run Evaluation"**

### Results Display

- **Overall Score**: Aggregate of all scripts
- **Pass/Fail Status**: Per script
- **Metrics Breakdown**: Individual metric values
- **Explanation**: Human-readable result description
- **Export**: Download results as JSON

---

## Export & Debug Features

### Export Options
- **📥 Export Results**: Full run results as JSON
- **📝 Export Answer**: Generated answer as text
- **⚙️ Export Config**: Configuration snapshot as JSON

### Debug Panel
Each pipeline run includes a debug panel with:
- **Config Tab**: Full configuration snapshot
- **Rewrite Tab**: Query rewrite details (Task C)
- **Retrieval Tab**: Retrieval statistics
- **Generation Tab**: Generation details
- **Raw JSON Tab**: Complete run result

---

## Project Structure

```
├── app.py                 # Main Streamlit application
├── src/
│   ├── ingestion.py       # Document loading and chunking
│   ├── vector_store.py    # FAISS/Chroma/Pinecone setup
│   ├── llm_client.py      # LLM provider integration
│   ├── rag.py             # RAG chain creation
│   ├── query_rewrite.py   # Query rewriting module
│   ├── evaluation.py      # Evaluation scripts
│   └── database.py        # Session management
├── requirements.txt
└── README.md
```

---

## Troubleshooting

### Common Issues

1. **"No vector store loaded"**
   - Upload and process documents first

2. **"API key required"**
   - Enter API key in sidebar for selected provider

3. **Slow retrieval**
   - Reduce Top-K value
   - Use smaller embedding model

4. **Generation timeout**
   - Check API key validity
   - Try smaller model (e.g., gpt-3.5-turbo)

### Logs
Check console output for detailed logs. Set `logging.DEBUG` for verbose output.
