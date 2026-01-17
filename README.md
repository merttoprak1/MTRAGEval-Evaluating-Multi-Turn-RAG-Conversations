# MTRAGEval - Multi-Turn RAG Evaluation System

A modular Retrieval-Augmented Generation (RAG) system integrated with the **MTRAG Benchmark** for evaluating multi-turn conversational AI. Built with Streamlit, LangChain, and FAISS.

## Features

- **MTRAG Benchmark Integration**: Official multi-turn RAG evaluation from IBM Research
- **Multi-Provider LLM Support**: Choose any local model suitable for your machine from Ollama or Google AI Studio
- **BEIR Format Support**: Standard benchmark data format for retrieval tasks
- **Multi-Turn History**: Proper conversation context handling for chat-based evaluation
- **Vector Store Options**: FAISS support (and Qdrant soon)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Pull an embedding model
ollama pull nomic-embed-text

# Pull an LLM model
ollama pull llama3

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

#### CLI Mode (Coming Soon)

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
3. Go to **📚 Knowledge Base** tab
4. Select pre-ingested collection or set up the configurations and select a corpus to ingest
5. Go to **💬 Interactive Playground** tab
6. Select a Task (A, B, or C) to perform
7. Set up Thread and Query Rewrite configurations
8. Upload a input file in the format "retrieval_taskac_input.jsonl" or "retrieval_taskb_input.jsonl"
9. Click on "Run Pipeline" and get your results

### Evaluation Metrics

| Task         | Metrics                                                   |
| ------------ | --------------------------------------------------------- |
| **Task A**   | Recall@K, nDCG@K                                          |
| **Task B/C** | Faithfulness, Appropriateness, Completeness, IDK Accuracy |

## UI Tabs

| Tab                       | Purpose                           |
| ------------------------- | --------------------------------- |
| 💬 Interactive Playground | Manual RAG testing (Task A/B/C)   |
| 📊 Batch Evaluation       | Official benchmark evaluation     |
| 📚 Knowledge Base         | Configuration & collection set-up |
| 📝 Logs & Debugging       | For debugging and troubleshooting |

## Development

### Adding New Evaluation Metrics

Extend `src/mtrag_evaluator.py` to add custom metrics or integrate additional MTRAG scripts.

### Troubleshooting

| Issue                    | Solution                              |
| ------------------------ | ------------------------------------- |
| Rate limit errors        | Use `--limit` flag or local models    |
| "No vector store loaded" | Upload and process documents first    |
| Import errors            | Run `pip install -r requirements.txt` |

## References

- [MTRAG Benchmark](https://github.com/IBM/mt-rag-benchmark) - Official benchmark repository
- [BEIR Benchmark](https://github.com/beir-cellar/beir) - Information retrieval benchmark format
- [LangChain](https://python.langchain.com/) - LLM orchestration framework

## License

MIT License
