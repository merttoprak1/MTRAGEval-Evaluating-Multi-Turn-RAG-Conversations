#!/usr/bin/env python
"""
MTRAG Benchmark Runner

CLI script to run MTRAGEval against the official MTRAG benchmark.
Supports Task A (Retrieval), Task B (Generation), and Task C (Full RAG).

Usage:
    python run_mtrag_benchmark.py --corpus clapnq --task generation_taskb
    python run_mtrag_benchmark.py --corpus clapnq --task retrieval_taska --top_k 5
    python run_mtrag_benchmark.py --corpus clapnq --task rag_taskc --provider gemini --api_key YOUR_KEY
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, skip

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ingestion import load_beir_corpus, chunk_documents
from src.vector_store import setup_vector_store, get_retriever
from src.llm_client import get_llm
from src.rag import convert_mtrag_history_to_messages, create_rag_chain_with_history, run_rag_with_mtrag_input
from src.mtrag_evaluator import (
    validate_predictions,
    run_retrieval_evaluation,
    run_generation_evaluation,
    save_predictions_jsonl
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MTRAG Benchmark paths (relative to this script)
MTRAG_ROOT = Path(__file__).parent.parent
CORPORA_DIR = MTRAG_ROOT / "corpora" / "passage_level"  # Use passage_level corpus
HUMAN_DIR = MTRAG_ROOT / "human"
RETRIEVAL_TASKS_DIR = HUMAN_DIR / "retrieval_tasks"
GENERATION_TASKS_DIR = HUMAN_DIR / "generation_tasks"


def get_corpus_path(corpus_name: str) -> Path:
    """Get path to corpus JSONL file."""
    corpus_file = CORPORA_DIR / f"{corpus_name}.jsonl"
    if corpus_file.exists():
        return corpus_file
    
    # Try unzipping if zip exists
    zip_file = CORPORA_DIR / f"{corpus_name}.jsonl.zip"
    if zip_file.exists():
        import zipfile
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(CORPORA_DIR)
        return corpus_file
    
    raise FileNotFoundError(f"Corpus not found: {corpus_name}")


def get_reference_path(task_type: str, corpus_name: str = None) -> Path:
    """Get path to reference JSONL file."""
    if task_type == "generation_taskb" or task_type == "rag_taskc":
        return GENERATION_TASKS_DIR / "reference.jsonl"
    elif task_type == "retrieval_taska" and corpus_name:
        return RETRIEVAL_TASKS_DIR / corpus_name / f"{corpus_name}_questions.jsonl"
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def get_qrels_path(corpus_name: str) -> Path:
    """Get path to qrels TSV file."""
    return RETRIEVAL_TASKS_DIR / corpus_name / "qrels" / "dev.tsv"


def load_reference_tasks(reference_path: Path, corpus_filter: Optional[str] = None) -> List[Dict]:
    """Load tasks from reference JSONL file.
    
    Handles both MTRAG format (with Collection field) and BEIR format (_id, text).
    """
    tasks = []
    with open(reference_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            
            # For BEIR format (retrieval tasks), no filtering needed
            if "_id" in task and "text" in task:
                tasks.append(task)
                continue
            
            # Filter by corpus if specified (for MTRAG generation tasks)
            if corpus_filter:
                collection = task.get("Collection", "").lower()
                if corpus_filter.lower() not in collection.lower():
                    continue
            
            tasks.append(task)
    
    return tasks


def run_task_a_retrieval(
    tasks: List[Dict],
    retriever,
    top_k: int = 5
) -> List[Dict]:
    """Run Task A: Retrieval only.
    
    Supports both BEIR format (_id, text) and MTRAG format (task_id, input).
    """
    predictions = []
    
    for task in tqdm(tasks, desc="Running retrieval"):
        # Handle BEIR format (retrieval tasks)
        if "_id" in task and "text" in task:
            task_id = task.get("_id", "")
            query_text = task.get("text", "")
            # Extract the last question from multi-turn text
            lines = query_text.split("|user|:")
            current_question = lines[-1].strip() if lines else query_text
        # Handle MTRAG format (generation tasks)
        else:
            task_id = task.get("task_id", "")
            input_list = task.get("input", [])
            _, current_question = convert_mtrag_history_to_messages(input_list)
        
        conversation_id = task_id.split("<::>")[0] if "<::>" in task_id else task_id
        
        # Retrieve documents
        try:
            docs = retriever.get_relevant_documents(current_question)[:top_k]
            contexts = [
                {
                    "document_id": doc.metadata.get("id", ""),
                    "text": doc.page_content,
                    "score": 1.0  # Default score
                }
                for doc in docs
            ]
        except Exception as e:
            logger.error(f"Retrieval failed for {task_id}: {e}")
            contexts = []
        
        predictions.append({
            "task_id": task_id,
            "conversation_id": conversation_id,
            "contexts": contexts
        })
    
    return predictions


def run_task_b_generation(
    tasks: List[Dict],
    llm,
    use_provided_contexts: bool = True
) -> List[Dict]:
    """Run Task B: Generation only (using provided contexts)."""
    predictions = []
    
    for task in tqdm(tasks, desc="Running generation"):
        task_id = task.get("task_id", "")
        conversation_id = task.get("conversation_id", "")
        input_list = task.get("input", [])
        contexts = task.get("contexts", [])
        
        # Get conversation history and current question
        history_messages, current_question = convert_mtrag_history_to_messages(input_list)
        
        # Build context from provided documents
        if use_provided_contexts and contexts:
            context_text = "\n\n".join([
                f"[{i+1}] {ctx.get('text', '')}" 
                for i, ctx in enumerate(contexts)
            ])
        else:
            context_text = ""
        
        # Generate answer
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following retrieved context to answer the question. "
                "If you don't know the answer, say 'I do not know'.\n\n"
                f"Context:\n{context_text}"
            )
            
            messages = [SystemMessage(content=system_prompt)]
            messages.extend(history_messages)
            messages.append(HumanMessage(content=current_question))
            
            response = llm.invoke(messages)
            answer = response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            logger.error(f"Generation failed for {task_id}: {e}")
            answer = "I do not know."
        
        predictions.append({
            "task_id": task_id,
            "conversation_id": conversation_id,
            "predictions": answer
        })
    
    return predictions


def run_task_c_rag(
    tasks: List[Dict],
    llm,
    retriever,
    top_k: int = 5
) -> List[Dict]:
    """Run Task C: Full RAG (Retrieval + Generation)."""
    predictions = []
    
    for task in tqdm(tasks, desc="Running full RAG"):
        task_id = task.get("task_id", "")
        conversation_id = task.get("conversation_id", "")
        input_list = task.get("input", [])
        
        # Run RAG with MTRAG input
        try:
            result = run_rag_with_mtrag_input(
                llm=llm,
                retriever=retriever,
                mtrag_input=input_list,
                use_history=True
            )
            
            answer = result.get("answer", "I do not know.")
            context_docs = result.get("context", [])
            
            contexts = [
                {
                    "document_id": doc.metadata.get("id", ""),
                    "text": doc.page_content,
                    "score": 1.0
                }
                for doc in context_docs
            ] if context_docs else []
            
        except Exception as e:
            logger.error(f"RAG failed for {task_id}: {e}")
            answer = "I do not know."
            contexts = []
        
        predictions.append({
            "task_id": task_id,
            "conversation_id": conversation_id,
            "contexts": contexts,
            "predictions": answer
        })
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Run MTRAG Benchmark")
    parser.add_argument("--corpus", type=str, required=True,
                        choices=["clapnq", "cloud", "fiqa", "govt"],
                        help="Corpus to evaluate on")
    parser.add_argument("--task", type=str, required=True,
                        choices=["retrieval_taska", "generation_taskb", "rag_taskc"],
                        help="Task type to run")
    parser.add_argument("--output", type=str, default="results/predictions.jsonl",
                        help="Output path for predictions")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of documents to retrieve")
    parser.add_argument("--provider", type=str, default="Gemini",
                        choices=["OpenAI", "Gemini", "Local"],
                        help="LLM provider")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key for LLM provider")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name")
    parser.add_argument("--embedding_provider", type=str, default="Gemini",
                        choices=["OpenAI", "Gemini", "Local"],
                        help="Embedding provider")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip evaluation after running predictions")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of tasks to run (for testing)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running MTRAG Benchmark: corpus={args.corpus}, task={args.task}")
    
    # Load corpus and setup vector store (for retrieval tasks)
    retriever = None
    if args.task in ["retrieval_taska", "rag_taskc"]:
        logger.info(f"Loading corpus: {args.corpus}")
        corpus_path = get_corpus_path(args.corpus)
        documents = load_beir_corpus(str(corpus_path))
        
        logger.info(f"Chunking {len(documents)} documents...")
        chunks = chunk_documents(documents)
        
        logger.info(f"Setting up vector store with {len(chunks)} chunks...")
        embedding_config = {
            "provider": args.embedding_provider,
            "api_key": args.api_key,
            "model_name": "models/text-embedding-004" if args.embedding_provider == "Gemini" else "text-embedding-3-small"
        }
        
        vector_store = setup_vector_store(
            chunks,
            embedding_config=embedding_config,
            collection_name=f"mtrag_{args.corpus}",
            db_type="FAISS"
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": args.top_k})
    
    # Setup LLM (for generation tasks)
    llm = None
    if args.task in ["generation_taskb", "rag_taskc"]:
        # Get API key from args or environment
        api_key = args.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("API key required for generation tasks. Set --api_key or GOOGLE_API_KEY env var")
            sys.exit(1)
        
        model_name = args.model or ("gemini-flash-latest" if args.provider == "Gemini" else "gpt-3.5-turbo")
        llm = get_llm(args.provider, api_key, model_name=model_name)
    
    # Load reference tasks
    reference_path = get_reference_path(args.task, args.corpus)
    logger.info(f"Loading tasks from: {reference_path}")
    tasks = load_reference_tasks(reference_path, corpus_filter=args.corpus)
    
    if args.limit:
        tasks = tasks[:args.limit]
    
    logger.info(f"Running {len(tasks)} tasks...")
    
    # Run appropriate task
    if args.task == "retrieval_taska":
        predictions = run_task_a_retrieval(tasks, retriever, args.top_k)
    elif args.task == "generation_taskb":
        predictions = run_task_b_generation(tasks, llm)
    else:  # rag_taskc
        predictions = run_task_c_rag(tasks, llm, retriever, args.top_k)
    
    # Save predictions
    save_predictions_jsonl(predictions, str(output_path), args.task)
    logger.info(f"Saved predictions to: {output_path}")
    
    # Run evaluation
    if not args.skip_eval:
        logger.info("Running MTRAG evaluation...")
        
        # Validate format
        validation = validate_predictions(str(output_path), args.task)
        if validation["valid"]:
            logger.info("✓ Format validation passed")
        else:
            logger.warning(f"✗ Format validation failed: {validation['errors']}")
        
        # Run metrics
        if args.task == "retrieval_taska":
            qrels_path = get_qrels_path(args.corpus)
            if qrels_path.exists():
                results = run_retrieval_evaluation(str(output_path), str(qrels_path))
                logger.info(f"Retrieval Results: {results}")
        
        elif args.task in ["generation_taskb", "rag_taskc"]:
            results = run_generation_evaluation(
                str(output_path),
                str(reference_path),
                llm_provider=args.provider.lower(),
                api_key=args.api_key
            )
            logger.info(f"Generation Results: {results}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
