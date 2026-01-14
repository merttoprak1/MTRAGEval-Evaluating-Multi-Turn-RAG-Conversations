Markdown

# 🤖 Modular MT-RAG Workbench

A high-performance, modular RAG (Retrieval-Augmented Generation) workbench designed for developing, testing, and evaluating Multi-Turn RAG systems against the [IBM MT-RAG Benchmark](https://github.com/IBM/MTRAGEval).

This application provides a Streamlit GUI to experiment with different RAG components (Embedding models, Vector DBs, LLMs, Query Rewriting strategies) and run official benchmark evaluations locally or on a high-performance server.

## 📋 Key Features

- **Modular Architecture:** Swap components (OpenAI, Gemini, Local Ollama/HuggingFace) on the fly without restarting.
- **Multi-Turn Support:** Built-in Query Rewriting and History management to handle context-dependent user queries.
- **Benchmark Integration:** seamlessly run official **Task A** (Retrieval), **Task B** (Generation), and **Task C** (End-to-End) evaluations.
- **Local-First Optimization:** Optimized for running with local LLMs (via Ollama) and local Vector DBs (FAISS) to ensure data privacy and zero cost during development.
- **Batch Ingestion:** One-click ingestion for raw JSONL files or standard MT-RAG corpora (`clapnq`, `cloud`, `fiqa`, `govt`) directly from the `corpora/` directory.

---

## 🛠️ Installation

### Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com/)** (Required for Local LLM/Embedding strategy)
- **API Keys** (Optional: OpenAI or Google Gemini if using cloud providers)

### 1. Clone the Repository

```bash
git clone <repository_url>
cd MTRAGEval-Workbench
2. Set up Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

Bash
# MacOS/Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
3. Install Dependencies

Bash
pip install -r requirements.txt
> Note: If you encounter installation errors related to flash_attn or torch, please install PyTorch compatible with your system first, then re-run the command above.

⚙️ Local Model Setup (Recommended)
To achieve high scores using local hardware (e.g., MacBook Pro) before deploying to a server, we recommend using Ollama.

Install Ollama from ollama.com.

Pull the models specified in the architecture:

Bash
# Embedding Model
ollama pull nomic-embed-text

# LLM (Small but capable for local dev)
ollama pull qwen2.5:3b
Ensure the Ollama server is running (ollama serve).

🚀 Usage Guide
Start the application:

Bash
streamlit run app.py
The interface will open at http://localhost:8501.

1. 📚 Knowledge Base (Ingestion)

Before running queries, you must create a Vector Store.

Navigate to the Knowledge Base tab.

Configuration:

LLM Provider: Set to Local (or API provider).

Embedding: Set to Local (Ollama) -> nomic-embed-text.

Vector DB: Select FAISS.

Naming: Enter a suffix in "Knowledge Base Name" (e.g., v1).

This defines the folder name: vectordb/faiss_db_{corpus}_{v1}.

Ingestion:

Select "Choose from MT-RAG Corpora".

Select a specific corpus (e.g., clapnq) or "Full Corpora" to process all 4 datasets at once.

Click Ingest. The system automatically unzips, chunks, and indexes the data into vectordb/.

2. 💬 Interactive Playground

Test your pipeline logic manually.

Navigate to the Interactive Playground tab.

Select a Task:

Task A (Retrieval Only): Upload a query file (e.g., retrieval_taskac_input.jsonl) to see what documents are found.

Task B (Generation): Test the LLM's ability to answer given a specific context.

Task C (End-to-End): Test the full flow: Rewrite Query -> Retrieve -> Generate.

Click Run Pipeline.

Download the generated predictions.jsonl file.

3. 📊 Batch Evaluation

Run the official scoring scripts.

Navigate to the Batch Evaluation tab.

Mode 1: Official Benchmark Run:

Select the corpus and task.

This executes run_mtrag_benchmark.py in the background.

Mode 2: Evaluate Custom Files:

Upload the predictions.jsonl you generated in the Playground.

(For Task A) Select the Ground Truth corpus.

The system calculates metrics like NDCG, Recall, and Faithfulness.

📂 Project Structure
Plaintext
.
├── app.py                 # Main Streamlit Application Entrypoint
├── app.log                # Runtime logs for debugging
├── corpora/               # Zipped Datasets (Source of Truth)
│   ├── document_level/    # Source documents (clapnq, cloud, etc.)
│   └── passage_level/     # Pre-chunked passages
├── src/
│   ├── ingestion.py       # JSONL parsing and document chunking
│   ├── vector_store.py    # FAISS/Chroma logic (saves to /vectordb)
│   ├── embeddings.py      # Wrapper for Ollama/OpenAI embeddings
│   ├── rag.py             # LangChain RAG pipeline construction
│   └── query_rewrite.py   # Multi-turn query rewriting logic
├── vectordb/              # Location where FAISS indexes are saved
├── predictions/           # Output folder for generated results
├── run_mtrag_benchmark.py # CLI Entrypoint for automated benchmarking
└── requirements.txt       # Project dependencies
🧪 Evaluation Metrics
This workbench focuses on three key pillars for the IBM Benchmark:

Faithfulness (F): Answers must be grounded strictly in the retrieved context. Hallucinations are penalized.

Answerability (IDK): The system must correctly identify when the retrieved context is insufficient and output "I do not know" or the specific IDK token.

Contextual Accuracy: For multi-turn conversations, the system must resolve pronouns (e.g., "it", "he") by rewriting the query based on chat history before retrieval.

📝 License
This project is a workbench for the MTRAG Benchmark. Please refer to the original paper and repository for dataset licensing and citation requirements.
```
