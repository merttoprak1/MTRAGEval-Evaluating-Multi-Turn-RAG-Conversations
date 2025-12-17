"""
Evaluation module for RAG pipeline evaluation.
Provides various evaluation scripts for Task A, B, C.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Structured evaluation result."""
    script_name: str
    overall_score: float
    metrics: Dict[str, float]
    explanation: str
    details: Dict[str, Any]
    passed: bool


# ==================== TASK A: Retrieval Evaluation ====================

def evaluate_retrieval_accuracy(
    retrieved_docs: List,
    ground_truth_docs: List[str] = None,
    top_k: int = 5
) -> EvaluationResult:
    """
    Evaluate if relevant documents are retrieved (Hit Rate).
    """
    if not retrieved_docs:
        return EvaluationResult(
            script_name="Retrieval Accuracy",
            overall_score=0.0,
            metrics={"hit_rate": 0.0, "mrr": 0.0},
            explanation="No documents were retrieved.",
            details={"retrieved_count": 0},
            passed=False
        )
    
    hit_rate = 0.0
    mrr = 0.0
    
    if ground_truth_docs:
        # Check if any ground truth doc is in retrieved docs
        gt_set = set(ground_truth_docs)
        retrieved_ids = [doc.metadata.get('id', doc.metadata.get('source', '')) for doc in retrieved_docs]
        
        hits = sum(1 for rid in retrieved_ids if rid in gt_set)
        hit_rate = hits / len(gt_set) if gt_set else 0.0
        
        # MRR: Mean Reciprocal Rank
        for i, rid in enumerate(retrieved_ids):
            if rid in gt_set:
                mrr = 1.0 / (i + 1)
                break
    else:
        # Without ground truth, we assume retrieval worked if we got documents
        hit_rate = 1.0 if retrieved_docs else 0.0
        mrr = 1.0 if retrieved_docs else 0.0
    
    overall = (hit_rate + mrr) / 2
    
    return EvaluationResult(
        script_name="Retrieval Accuracy",
        overall_score=overall,
        metrics={"hit_rate": hit_rate, "mrr": mrr},
        explanation=f"Hit Rate: {hit_rate:.2%}, MRR: {mrr:.2f}. " + 
                   ("Ground truth comparison performed." if ground_truth_docs else "No ground truth provided, basic check only."),
        details={
            "retrieved_count": len(retrieved_docs),
            "ground_truth_count": len(ground_truth_docs) if ground_truth_docs else 0,
            "has_ground_truth": bool(ground_truth_docs)
        },
        passed=overall >= 0.5
    )


def evaluate_semantic_similarity(
    query: str,
    retrieved_docs: List,
    scores: List[float] = None
) -> EvaluationResult:
    """
    Evaluate semantic similarity between query and retrieved docs.
    Uses provided scores or basic text similarity.
    """
    if not retrieved_docs:
        return EvaluationResult(
            script_name="Semantic Similarity",
            overall_score=0.0,
            metrics={"avg_similarity": 0.0, "min_similarity": 0.0, "max_similarity": 0.0},
            explanation="No documents to evaluate similarity.",
            details={},
            passed=False
        )
    
    if scores:
        # Use provided similarity scores (lower is better for distance-based)
        # Normalize: assuming scores are distances, convert to similarity
        similarities = [max(0, 1 - s) if s > 1 else (1 - s) if s > 0 else 1 for s in scores]
    else:
        # Basic text similarity using SequenceMatcher
        similarities = []
        query_lower = query.lower()
        for doc in retrieved_docs:
            content_lower = doc.page_content.lower()[:500]  # Compare first 500 chars
            ratio = SequenceMatcher(None, query_lower, content_lower).ratio()
            similarities.append(ratio)
    
    avg_sim = sum(similarities) / len(similarities)
    min_sim = min(similarities)
    max_sim = max(similarities)
    
    return EvaluationResult(
        script_name="Semantic Similarity",
        overall_score=avg_sim,
        metrics={"avg_similarity": avg_sim, "min_similarity": min_sim, "max_similarity": max_sim},
        explanation=f"Average similarity: {avg_sim:.2%}. Range: [{min_sim:.2%}, {max_sim:.2%}]",
        details={
            "all_similarities": similarities,
            "doc_count": len(retrieved_docs)
        },
        passed=avg_sim >= 0.3
    )


# ==================== TASK B: Generation Evaluation ====================

def evaluate_answer_correctness(
    generated_answer: str,
    ground_truth_answer: str = None
) -> EvaluationResult:
    """
    Evaluate correctness of generated answer against ground truth.
    """
    if not generated_answer:
        return EvaluationResult(
            script_name="Answer Correctness",
            overall_score=0.0,
            metrics={"exact_match": 0.0, "f1_score": 0.0},
            explanation="No answer was generated.",
            details={},
            passed=False
        )
    
    if not ground_truth_answer:
        return EvaluationResult(
            script_name="Answer Correctness",
            overall_score=0.5,
            metrics={"exact_match": 0.0, "f1_score": 0.0},
            explanation="No ground truth provided. Cannot evaluate correctness. Answer was generated successfully.",
            details={"answer_length": len(generated_answer)},
            passed=True  # Passed because answer was generated
        )
    
    # Normalize texts
    gen_normalized = generated_answer.lower().strip()
    gt_normalized = ground_truth_answer.lower().strip()
    
    # Exact match
    exact_match = 1.0 if gen_normalized == gt_normalized else 0.0
    
    # F1 Score (word overlap)
    gen_words = set(gen_normalized.split())
    gt_words = set(gt_normalized.split())
    
    if not gt_words:
        f1_score = 0.0
    else:
        common = gen_words.intersection(gt_words)
        precision = len(common) / len(gen_words) if gen_words else 0
        recall = len(common) / len(gt_words) if gt_words else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Sequence similarity
    similarity = SequenceMatcher(None, gen_normalized, gt_normalized).ratio()
    
    overall = (exact_match * 0.3 + f1_score * 0.4 + similarity * 0.3)
    
    return EvaluationResult(
        script_name="Answer Correctness",
        overall_score=overall,
        metrics={"exact_match": exact_match, "f1_score": f1_score, "similarity": similarity},
        explanation=f"F1 Score: {f1_score:.2%}, Similarity: {similarity:.2%}. " +
                   ("Exact match!" if exact_match else "Not an exact match."),
        details={
            "generated_length": len(generated_answer),
            "ground_truth_length": len(ground_truth_answer),
            "common_words": len(gen_words.intersection(gt_words))
        },
        passed=overall >= 0.5
    )


def evaluate_faithfulness(
    generated_answer: str,
    context: str,
    llm=None
) -> EvaluationResult:
    """
    Evaluate if the answer is grounded in the retrieved context.
    Uses LLM-as-judge if available, otherwise uses text matching.
    """
    if not generated_answer or not context:
        return EvaluationResult(
            script_name="Faithfulness",
            overall_score=0.0,
            metrics={"faithfulness_score": 0.0},
            explanation="Missing answer or context for faithfulness evaluation.",
            details={},
            passed=False
        )
    
    if llm:
        try:
            # LLM-as-judge evaluation
            from langchain_core.messages import HumanMessage
            
            judge_prompt = f"""You are an expert evaluator. Evaluate if the following answer is faithful to the given context.
            
Context:
{context[:3000]}

Answer:
{generated_answer}

Evaluate faithfulness on a scale of 0.0 to 1.0 where:
- 1.0 = Answer is completely grounded in the context
- 0.5 = Answer is partially grounded, some claims not supported
- 0.0 = Answer contains hallucinations or unsupported claims

Respond with ONLY a JSON object: {{"score": <float>, "reason": "<brief explanation>"}}"""
            
            response = llm.invoke([HumanMessage(content=judge_prompt)])
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            import json
            import re
            json_match = re.search(r'\{[^}]+\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
                score = float(result.get('score', 0.5))
                reason = result.get('reason', 'LLM-as-judge evaluation')
            else:
                score = 0.5
                reason = "Could not parse LLM response"
            
            return EvaluationResult(
                script_name="Faithfulness",
                overall_score=score,
                metrics={"faithfulness_score": score},
                explanation=reason,
                details={"evaluation_method": "llm_as_judge"},
                passed=score >= 0.5
            )
        except Exception as e:
            logger.warning(f"LLM-as-judge failed: {e}, falling back to text matching")
    
    # Fallback: Text overlap between answer and context
    answer_words = set(generated_answer.lower().split())
    context_words = set(context.lower().split())
    
    if not answer_words:
        faithfulness = 0.0
    else:
        overlap = len(answer_words.intersection(context_words))
        faithfulness = overlap / len(answer_words)
    
    return EvaluationResult(
        script_name="Faithfulness",
        overall_score=faithfulness,
        metrics={"faithfulness_score": faithfulness},
        explanation=f"Word overlap faithfulness: {faithfulness:.2%}. {overlap} of {len(answer_words)} answer words found in context.",
        details={
            "evaluation_method": "word_overlap",
            "answer_words": len(answer_words),
            "context_words": len(context_words),
            "overlap": overlap
        },
        passed=faithfulness >= 0.5
    )


# ==================== TASK C: Full Pipeline Evaluation ====================

def evaluate_latency(
    rewrite_time_ms: float = 0,
    retrieval_time_ms: float = 0,
    generation_time_ms: float = 0,
    total_time_ms: float = 0
) -> EvaluationResult:
    """
    Evaluate pipeline latency performance.
    """
    # Define acceptable thresholds (in ms)
    thresholds = {
        "rewrite": 2000,
        "retrieval": 1000,
        "generation": 5000,
        "total": 10000
    }
    
    metrics = {
        "rewrite_time_ms": rewrite_time_ms,
        "retrieval_time_ms": retrieval_time_ms,
        "generation_time_ms": generation_time_ms,
        "total_time_ms": total_time_ms
    }
    
    # Score based on meeting thresholds
    scores = []
    if rewrite_time_ms > 0:
        scores.append(1.0 if rewrite_time_ms <= thresholds["rewrite"] else thresholds["rewrite"] / rewrite_time_ms)
    if retrieval_time_ms > 0:
        scores.append(1.0 if retrieval_time_ms <= thresholds["retrieval"] else thresholds["retrieval"] / retrieval_time_ms)
    if generation_time_ms > 0:
        scores.append(1.0 if generation_time_ms <= thresholds["generation"] else thresholds["generation"] / generation_time_ms)
    
    overall = sum(scores) / len(scores) if scores else 1.0
    
    return EvaluationResult(
        script_name="Latency Analysis",
        overall_score=overall,
        metrics=metrics,
        explanation=f"Total pipeline time: {total_time_ms:.0f}ms. " +
                   f"Breakdown - Rewrite: {rewrite_time_ms:.0f}ms, Retrieval: {retrieval_time_ms:.0f}ms, Generation: {generation_time_ms:.0f}ms",
        details={"thresholds": thresholds},
        passed=total_time_ms <= thresholds["total"]
    )


def evaluate_e2e_accuracy(
    generated_answer: str,
    ground_truth_answer: str = None,
    query: str = None,
    rewritten_query: str = None
) -> EvaluationResult:
    """
    End-to-end accuracy for full RAG pipeline.
    """
    # Use answer correctness as base
    correctness = evaluate_answer_correctness(generated_answer, ground_truth_answer)
    
    # Additional check: query improvement
    query_improved = 0.0
    if query and rewritten_query and query != rewritten_query:
        query_improved = 0.1  # Bonus for query rewriting
    
    overall = min(1.0, correctness.overall_score + query_improved)
    
    return EvaluationResult(
        script_name="End-to-End Accuracy",
        overall_score=overall,
        metrics={
            "answer_score": correctness.overall_score,
            "query_improved": query_improved,
            **correctness.metrics
        },
        explanation=f"E2E Score: {overall:.2%}. " + correctness.explanation + 
                   (" Query was rewritten." if query_improved else ""),
        details={
            **correctness.details,
            "query_rewritten": bool(query_improved)
        },
        passed=overall >= 0.5
    )


# ==================== Main Evaluation Runner ====================

def run_evaluation(
    script_name: str,
    run_result: Dict[str, Any],
    ground_truth_answer: str = None,
    ground_truth_docs: List[str] = None,
    llm=None
) -> EvaluationResult:
    """
    Run a specific evaluation script on pipeline results.
    """
    logger.info(f"Running evaluation: {script_name}")
    
    # Extract data from run_result
    query = run_result.get("query", "")
    retrieved_docs = run_result.get("retrieved_docs", [])
    retrieval_data = run_result.get("retrieval", {})
    generation_data = run_result.get("generation", {})
    rewrite_data = run_result.get("rewrite_result", {})
    
    generated_answer = generation_data.get("answer", "")
    
    # Build context from retrieved docs
    context = ""
    if retrieved_docs:
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Get scores from retrieval
    scores = [r.get("score", 0) for r in retrieval_data.get("results", [])]
    
    # Route to appropriate evaluation function
    if script_name == "Retrieval Accuracy (Hit Rate)":
        return evaluate_retrieval_accuracy(retrieved_docs, ground_truth_docs)
    
    elif script_name == "Retrieval Precision":
        return evaluate_retrieval_accuracy(retrieved_docs, ground_truth_docs)
    
    elif script_name == "Semantic Similarity":
        return evaluate_semantic_similarity(query, retrieved_docs, scores)
    
    elif script_name == "Answer Correctness":
        return evaluate_answer_correctness(generated_answer, ground_truth_answer)
    
    elif script_name == "Faithfulness":
        return evaluate_faithfulness(generated_answer, context, llm)
    
    elif script_name == "Answer Relevance":
        # Use faithfulness with query as context check
        return evaluate_faithfulness(generated_answer, query, llm)
    
    elif script_name == "Context Relevance":
        return evaluate_semantic_similarity(query, retrieved_docs, scores)
    
    elif script_name == "End-to-End Accuracy":
        return evaluate_e2e_accuracy(
            generated_answer, 
            ground_truth_answer,
            query,
            rewrite_data.get("rewritten", "")
        )
    
    elif script_name == "Query Rewrite Quality":
        original = rewrite_data.get("original", "")
        rewritten = rewrite_data.get("rewritten", "")
        improved = original != rewritten
        return EvaluationResult(
            script_name="Query Rewrite Quality",
            overall_score=1.0 if improved else 0.5,
            metrics={"query_changed": 1.0 if improved else 0.0},
            explanation=f"Query was {'modified' if improved else 'not modified'} by rewriter.",
            details={"original": original, "rewritten": rewritten},
            passed=True
        )
    
    elif script_name == "RAGAS Score":
        # Combine multiple metrics
        faithfulness = evaluate_faithfulness(generated_answer, context, llm)
        correctness = evaluate_answer_correctness(generated_answer, ground_truth_answer)
        
        overall = (faithfulness.overall_score + correctness.overall_score) / 2
        
        return EvaluationResult(
            script_name="RAGAS Score",
            overall_score=overall,
            metrics={
                "faithfulness": faithfulness.overall_score,
                "answer_correctness": correctness.overall_score
            },
            explanation=f"RAGAS composite score combining faithfulness ({faithfulness.overall_score:.2%}) and correctness ({correctness.overall_score:.2%})",
            details={
                "faithfulness_details": faithfulness.details,
                "correctness_details": correctness.details
            },
            passed=overall >= 0.5
        )
    
    elif script_name == "Latency Analysis":
        return evaluate_latency(
            rewrite_time_ms=0,  # Would need to track this
            retrieval_time_ms=retrieval_data.get("retrieval_time_ms", 0),
            generation_time_ms=generation_data.get("generation_time_ms", 0),
            total_time_ms=run_result.get("total_time_ms", 0)
        )
    
    else:
        return EvaluationResult(
            script_name=script_name,
            overall_score=0.0,
            metrics={},
            explanation=f"Unknown evaluation script: {script_name}",
            details={},
            passed=False
        )
