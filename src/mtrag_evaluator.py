"""
MTRAG Evaluator Module

Bridge to MTRAG's official evaluation scripts.
Handles format conversion and script execution.
"""

import subprocess
import json
import logging
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# MTRAG scripts directory (relative to MTRAGEval project root)
MTRAG_BENCHMARK_DIR = Path(__file__).parent.parent.parent.parent
MTRAG_SCRIPTS_DIR = MTRAG_BENCHMARK_DIR / "scripts" / "evaluation"


def get_mtrag_scripts_path() -> Path:
    """Get the path to MTRAG evaluation scripts directory."""
    if MTRAG_SCRIPTS_DIR.exists():
        return MTRAG_SCRIPTS_DIR
    
    # Try alternative paths
    alt_paths = [
        Path("../scripts/evaluation"),
        Path("../../scripts/evaluation"),
        Path("../mt-rag-benchmark-main/scripts/evaluation"),
    ]
    
    for alt in alt_paths:
        if alt.exists():
            return alt.resolve()
    
    raise FileNotFoundError(
        f"MTRAG scripts directory not found. Expected at: {MTRAG_SCRIPTS_DIR}"
    )


def validate_predictions(predictions_path: str, mode: str) -> Dict[str, Any]:
    """
    Validate predictions file using MTRAG format_checker.py.
    
    Args:
        predictions_path: Path to predictions JSONL file
        mode: One of 'retrieval_taska', 'generation_taskb', 'rag_taskc'
        
    Returns:
        Dictionary with validation result:
        {
            "valid": bool,
            "output": str,
            "errors": List[str]
        }
    """
    logger.info(f"Validating predictions: {predictions_path} (mode: {mode})")
    
    try:
        scripts_dir = get_mtrag_scripts_path()
        format_checker = scripts_dir / "format_checker.py"
        
        result = subprocess.run(
            ["python", str(format_checker), "--mode", mode, "--predictions", predictions_path],
            capture_output=True,
            text=True,
            cwd=str(scripts_dir)
        )
        
        return {
            "valid": result.returncode == 0,
            "output": result.stdout,
            "errors": result.stderr.split('\n') if result.stderr else []
        }
        
    except FileNotFoundError as e:
        logger.error(f"MTRAG scripts not found: {e}")
        return {"valid": False, "output": "", "errors": [str(e)]}
    except Exception as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        return {"valid": False, "output": "", "errors": [str(e)]}


def run_retrieval_evaluation(
    predictions_path: str, 
    qrels_path: str
) -> Dict[str, float]:
    """
    Run MTRAG retrieval evaluation using run_retrieval_eval.py.
    
    Args:
        predictions_path: Path to predictions JSONL with 'contexts' field
        qrels_path: Path to ground truth qrels TSV file
        
    Returns:
        Dictionary with retrieval metrics:
        {
            "Recall@1": float,
            "Recall@3": float,
            "Recall@5": float,
            "Recall@10": float,
            "nDCG@1": float,
            "nDCG@3": float,
            "nDCG@5": float,
            "nDCG@10": float
        }
    """
    logger.info(f"Running retrieval evaluation: {predictions_path}")
    
    try:
        scripts_dir = get_mtrag_scripts_path()
        eval_script = scripts_dir / "run_retrieval_eval.py"
        
        result = subprocess.run(
            [
                "python", str(eval_script),
                "--predictions", predictions_path,
                "--reference", qrels_path
            ],
            capture_output=True,
            text=True,
            cwd=str(scripts_dir)
        )
        
        if result.returncode != 0:
            logger.error(f"Retrieval eval failed: {result.stderr}")
            return {"error": result.stderr}
        
        return parse_retrieval_output(result.stdout)
        
    except Exception as e:
        logger.error(f"Retrieval evaluation error: {e}", exc_info=True)
        return {"error": str(e)}


def run_generation_evaluation(
    predictions_path: str,
    reference_path: str,
    llm_provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run MTRAG generation evaluation using run_generation_eval.py.
    
    Uses LLM-as-a-judge for evaluation metrics.
    
    Args:
        predictions_path: Path to predictions JSONL with 'predictions' field
        reference_path: Path to reference JSONL with 'targets' field
        llm_provider: LLM provider for judge ('openai', 'gemini', etc.)
        api_key: API key for the LLM provider
        model_name: Model name to use for judging
        
    Returns:
        Dictionary with generation metrics:
        {
            "faithfulness": float,
            "appropriateness": float,
            "completeness": float,
            "idk_accuracy": float,
            "overall": float
        }
    """
    logger.info(f"Running generation evaluation: {predictions_path}")
    
    try:
        scripts_dir = get_mtrag_scripts_path()
        eval_script = scripts_dir / "run_generation_eval.py"
        
        cmd = [
            "python", str(eval_script),
            "--predictions", predictions_path,
            "--reference", reference_path
        ]
        
        # Add LLM configuration if provided
        if llm_provider:
            cmd.extend(["--llm_provider", llm_provider])
        if api_key:
            cmd.extend(["--api_key", api_key])
        if model_name:
            cmd.extend(["--model_name", model_name])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(scripts_dir)
        )
        
        if result.returncode != 0:
            logger.error(f"Generation eval failed: {result.stderr}")
            return {"error": result.stderr}
        
        return parse_generation_output(result.stdout)
        
    except Exception as e:
        logger.error(f"Generation evaluation error: {e}", exc_info=True)
        return {"error": str(e)}


def parse_retrieval_output(output: str) -> Dict[str, float]:
    """Parse MTRAG retrieval evaluation output."""
    metrics = {}
    
    try:
        # Try to parse as JSON first
        data = json.loads(output)
        return data
    except json.JSONDecodeError:
        pass
    
    # Parse text output
    for line in output.split('\n'):
        line = line.strip()
        if ':' in line:
            key, value = line.split(':', 1)
            try:
                metrics[key.strip()] = float(value.strip())
            except ValueError:
                continue
    
    return metrics


def parse_generation_output(output: str) -> Dict[str, Any]:
    """Parse MTRAG generation evaluation output."""
    try:
        # Try to parse as JSON first
        data = json.loads(output)
        return data
    except json.JSONDecodeError:
        pass
    
    # Parse text output
    metrics = {}
    for line in output.split('\n'):
        line = line.strip()
        if ':' in line:
            key, value = line.split(':', 1)
            try:
                metrics[key.strip()] = float(value.strip())
            except ValueError:
                metrics[key.strip()] = value.strip()
    
    return metrics


def format_predictions_for_mtrag(
    predictions: List[Dict[str, Any]],
    task_type: str
) -> str:
    """
    Format predictions for MTRAG evaluation.
    
    Args:
        predictions: List of prediction dictionaries
        task_type: 'retrieval_taska', 'generation_taskb', or 'rag_taskc'
        
    Returns:
        JSONL string formatted for MTRAG evaluation
    """
    lines = []
    
    for pred in predictions:
        if task_type == "retrieval_taska":
            # Retrieval task: needs 'contexts' field
            formatted = {
                "task_id": pred.get("task_id", ""),
                "conversation_id": pred.get("conversation_id", ""),
                "contexts": pred.get("contexts", [])
            }
        elif task_type == "generation_taskb":
            # Generation task: needs 'predictions' field
            formatted = {
                "task_id": pred.get("task_id", ""),
                "conversation_id": pred.get("conversation_id", ""),
                "predictions": pred.get("predictions", pred.get("answer", ""))
            }
        else:  # rag_taskc
            # Full RAG: needs both contexts and predictions
            formatted = {
                "task_id": pred.get("task_id", ""),
                "conversation_id": pred.get("conversation_id", ""),
                "contexts": pred.get("contexts", []),
                "predictions": pred.get("predictions", pred.get("answer", ""))
            }
        
        lines.append(json.dumps(formatted, ensure_ascii=False))
    
    return '\n'.join(lines)


def save_predictions_jsonl(
    predictions: List[Dict[str, Any]],
    output_path: str,
    task_type: str
) -> str:
    """
    Save predictions to JSONL file in MTRAG format.
    
    Args:
        predictions: List of prediction dictionaries
        output_path: Path to save the JSONL file
        task_type: Task type for formatting
        
    Returns:
        Path to the saved file
    """
    content = format_predictions_for_mtrag(predictions, task_type)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"Saved {len(predictions)} predictions to {output_path}")
    return output_path


def run_full_evaluation(
    predictions: List[Dict[str, Any]],
    task_type: str,
    reference_path: str,
    qrels_path: Optional[str] = None,
    llm_provider: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run full MTRAG evaluation pipeline.
    
    Args:
        predictions: List of prediction dictionaries
        task_type: 'retrieval_taska', 'generation_taskb', or 'rag_taskc'
        reference_path: Path to reference JSONL
        qrels_path: Path to qrels TSV (for retrieval)
        llm_provider: LLM provider for generation eval
        api_key: API key for LLM
        
    Returns:
        Combined evaluation results
    """
    results = {"task_type": task_type}
    
    # Save predictions to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        content = format_predictions_for_mtrag(predictions, task_type)
        f.write(content)
        predictions_path = f.name
    
    try:
        # Validate format first
        validation = validate_predictions(predictions_path, task_type)
        results["validation"] = validation
        
        if not validation["valid"]:
            logger.warning(f"Predictions failed format validation: {validation['errors']}")
        
        # Run appropriate evaluation
        if task_type == "retrieval_taska" and qrels_path:
            results["retrieval"] = run_retrieval_evaluation(predictions_path, qrels_path)
            
        elif task_type == "generation_taskb":
            results["generation"] = run_generation_evaluation(
                predictions_path, reference_path, llm_provider, api_key
            )
            
        elif task_type == "rag_taskc":
            # Run both evaluations
            if qrels_path:
                results["retrieval"] = run_retrieval_evaluation(predictions_path, qrels_path)
            results["generation"] = run_generation_evaluation(
                predictions_path, reference_path, llm_provider, api_key
            )
        
        return results
        
    finally:
        # Cleanup temp file
        os.unlink(predictions_path)
