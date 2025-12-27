# MTRAGEval - Multi-Turn RAG Evaluation System

A modular Retrieval-Augmented Generation (RAG) system integrated with the **MTRAG Benchmark** for evaluating multi-turn conversational AI. Built with Streamlit, LangChain, and FAISS.

## Features

- **MTRAG Benchmark Integration**: Official multi-turn RAG evaluation from IBM Research
- **Multi-Provider LLM Support**: Google Gemini, OpenAI, and local models (Ollama/LM Studio)
- **BEIR Format Support**: Standard benchmark data format for retrieval tasks
- **Multi-Turn History**: Proper conversation context handling for chat-based evaluation
- **Vector Store Options**: FAISS, Chroma, and Pinecone support
- **Session Management**: Create, rename, and delete chat sessions

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file with your API key
echo "GOOGLE_API_KEY=your_key_here" > .env

# Run Streamlit UI
streamlit run app.py

# Or run CLI benchmark
python run_mtrag_benchmark.py --corpus clapnq --task generation_taskb --limit 10
```

## MTRAG Benchmark

### What is MTRAG?

MTRAG (Multi-Turn RAG) is a comprehensive benchmark dataset from IBM Research for evaluating RAG systems on multi-turn conversations. It includes:

- **4 Corpora**: ClapNQ, Cloud, FiQA, Govt
- **842 Evaluation Tasks**: Human-generated multi-turn conversations
- **3 Task Types**: Retrieval (A), Generation (B), Full RAG (C)

### Running Benchmarks

#### CLI Mode

```bash
# Task A: Retrieval Only
python run_mtrag_benchmark.py --corpus clapnq --task retrieval_taska --top_k 5

# Task B: Generation (with provided contexts)
python run_mtrag_benchmark.py --corpus clapnq --task generation_taskb --limit 10

# Task C: Full RAG Pipeline
python run_mtrag_benchmark.py --corpus clapnq --task rag_taskc --provider Gemini
```

#### UI Mode

1. Open `http://localhost:8501`
2. Enter API key in sidebar
3. Go to **📈 MTRAG Benchmark** tab
4. Select corpus and task type
5. Click **Run MTRAG Benchmark**

### Evaluation Metrics

| Task | Metrics |
|------|---------|
| **Task A** | Recall@K, nDCG@K |
| **Task B/C** | Faithfulness, Appropriateness, Completeness, IDK Accuracy |

## Project Structure

```
├── app.py                     # Streamlit UI
├── run_mtrag_benchmark.py     # CLI benchmark runner
├── src/
│   ├── ingestion.py           # BEIR + JSON document loading
│   ├── rag.py                 # RAG chains + multi-turn history
│   ├── mtrag_evaluator.py     # MTRAG evaluation bridge
│   ├── llm_client.py          # LLM provider factory
│   ├── vector_store.py        # Vector DB setup
│   ├── query_rewrite.py       # Query rewriting
│   └── evaluation/            # MTRAG official scripts
├── .env.example               # API key template
└── requirements.txt
```

## Configuration

### Google Gemini (Recommended)

1. Get API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create `.env` file: `GOOGLE_API_KEY=your_key`
3. Select **Gemini** as provider in sidebar

### Local Models (Ollama)

```bash
# Pull a model
ollama pull llama3

# Configure in app
# Provider: Local
# Base URL: http://localhost:11434/v1
# Model: llama3
```

## UI Tabs

| Tab | Purpose |
|-----|---------|
| 🎯 RAG Playground | Manual RAG testing (Task A/B/C) |
| 📈 MTRAG Benchmark | Official benchmark evaluation |
| 💬 Chat | Interactive conversation |
| 🛠️ Manage Collection | Vector DB document management |
| 🔍 Database Inspector | View chat history database |

## Key Components

### Multi-Turn History Support

The system properly handles MTRAG's multi-turn conversation format:

```python
from src.rag import convert_mtrag_history_to_messages

# MTRAG format → LangChain messages
history, current_question = convert_mtrag_history_to_messages(input_list)
```

### BEIR Format Loading

```python
from src.ingestion import load_beir_corpus, load_beir_queries, load_beir_qrels

# Load MTRAG corpus
docs = load_beir_corpus("path/to/corpus.jsonl")
queries = load_beir_queries("path/to/queries.jsonl")
qrels = load_beir_qrels("path/to/qrels/dev.tsv")
```

## Development

### Adding New Evaluation Metrics

Extend `src/mtrag_evaluator.py` to add custom metrics or integrate additional MTRAG scripts.

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Rate limit errors | Use `--limit` flag or local models |
| "No vector store loaded" | Upload and process documents first |
| Import errors | Run `pip install -r requirements.txt` |

## References

- [MTRAG Benchmark](https://github.com/IBM/mt-rag-benchmark) - Official benchmark repository
- [BEIR Benchmark](https://github.com/beir-cellar/beir) - Information retrieval benchmark format
- [LangChain](https://python.langchain.com/) - LLM orchestration framework

## License

MIT License
