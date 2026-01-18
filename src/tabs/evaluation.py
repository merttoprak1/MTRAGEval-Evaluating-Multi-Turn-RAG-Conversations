import ast
import streamlit as st
import pandas as pd
import tempfile
import subprocess
import os
import platform
import logging
import sys
import json
import concurrent.futures
import time
from pathlib import Path
from datetime import datetime
from src.ingestion import load_json_documents, load_beir_queries
from src.vector_store import setup_vector_store, get_retriever, add_to_vector_store, delete_from_vector_store
from src.llm_client import get_llm
from src.rag import create_rag_chain
from src.query_rewrite import rewrite_query, DEFAULT_REWRITE_PROMPT, CONTEXTUAL_REWRITE_PROMPT
from src.beir_utils import (
    AVAILABLE_CORPORA, QUERY_TYPES, get_retrieval_task_paths,
    load_qrels, load_queries, calculate_retrieval_metrics
)
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from src.calculate_ranking import append_ranking_score
from src.prepare_doc_to_task_c_evaluation import merge_predictions_with_reference

logger = logging.getLogger(__name__)

def render():
    st.header("📊 Batch Evaluation")
    
    # We split the tab into two modes: Official Benchmark vs. Custom File Eval
    eval_mode = st.radio("Evaluation Mode", [ "📂 Evaluate Custom Files","🚀 Run Official MTRAG Benchmark"], horizontal=True)
    st.divider()

    # # ---------------- MODE 1: OFFICIAL BENCHMARK RUNNER ----------------
    if eval_mode == "🚀 Run Official MTRAG Benchmark":

        st.subheader("UNDER CONSTRUCTION 🚧 PROPOSAL TO BE REMOVED")
    #     st.markdown("""
    #     Run the standard benchmark on specific corpora using the `run_mtrag_benchmark.py` script.
    #     This generates predictions which can then be evaluated.
    #     """)
        
    #     # Configuration
    #     col1, col2 = st.columns(2)
    #     with col1:
    #         mtrag_corpus = st.selectbox(
    #             "Select Corpus",
    #             ["clapnq", "cloud", "fiqa", "govt"],
    #             help="MTRAG corpus to evaluate on"
    #         )
            
    #         mtrag_task = st.selectbox(
    #             "Select Task",
    #             ["generation_taskb", "retrieval_taska", "rag_taskc"],
    #             format_func=lambda x: {
    #                 "retrieval_taska": "Task A: Retrieval Only",
    #                 "generation_taskb": "Task B: Generation",
    #                 "rag_taskc": "Task C: Full RAG Pipeline"
    #             }.get(x, x)
    #         )
        
    #     with col2:
    #         mtrag_limit = st.number_input("Limit Examples", 1, 1000, 10)
    #         skip_eval = st.checkbox("Skip Auto-Evaluation (Generate Only)", value=False)
        
    #     if st.button("▶️ Start Benchmark Run", type="primary"):
    #         if provider != "Local" and not api_key:
    #             st.error("❌ API Key required in Sidebar.")
    #         else:
    #             try:
    #                 import subprocess
    #                 import sys
                    
    #                 # Command Construction
    #                 cmd = [
    #                     sys.executable, "run_mtrag_benchmark.py",
    #                     "--corpus", mtrag_corpus,
    #                     "--task", mtrag_task,
    #                     "--limit", str(mtrag_limit),
    #                     "--provider", provider,
    #                     "--model", model_name
    #                 ]
                    
    #                 if provider == "Local":
    #                     if base_url: cmd.extend(["--base_url", base_url])
    #                 else:
    #                     cmd.extend(["--api_key", api_key])
                        
    #                 if skip_eval:
    #                     cmd.append("--skip_eval")
                        
    #                 # Output file path logic
    #                 output_file = Path("results") / mtrag_task / f"{mtrag_corpus}_predictions.jsonl"
    #                 cmd.extend(["--output", str(output_file)])
                    
    #                 st.info(f"Executing: {' '.join(cmd)}")
                    
    #                 with st.spinner(f"Running benchmark on {mtrag_corpus}..."):
    #                     # Run Process
    #                     process = subprocess.Popen(
    #                         cmd,
    #                         stdout=subprocess.PIPE,
    #                         stderr=subprocess.PIPE,
    #                         text=True,
    #                         encoding='utf-8',
    #                         errors='replace'
    #                     )
                        
    #                     # Live Log Output
    #                     output_container = st.empty()
    #                     logs = []
    #                     while True:
    #                         output = process.stdout.readline()
    #                         if output == '' and process.poll() is not None:
    #                             break
    #                         if output:
    #                             line = output.strip()
    #                             logs.append(line)
    #                             output_container.code("\n".join(logs[-10:]), language="bash")
                        
    #                     if process.poll() != 0:
    #                         st.error(f"Failed with code {process.poll()}")
    #                         st.error(process.stderr.read())
    #                     else:
    #                         st.success("✅ Benchmark Completed!")
                            
    #                         # Show Results
    #                         if output_file.exists():
    #                             st.subheader("📋 Predictions Preview")
    #                             predictions = []
    #                             with open(output_file, 'r', encoding='utf-8') as f:
    #                                 for line in f:
    #                                     predictions.append(json.loads(line))
                                
    #                             results_df = pd.DataFrame([{
    #                                 "Task ID": p.get("task_id", "")[:15],
    #                                 "Prediction": str(p.get("predictions", ""))[:80] + "...",
    #                                 "Contexts": len(p.get("contexts", []))
    #                             } for p in predictions])
                                
    #                             st.dataframe(results_df, use_container_width=True)
                                
    #                             with open(output_file, "r", encoding='utf-8') as f:
    #                                 st.download_button("📥 Download JSONL", f, output_file.name)

    #             except Exception as e:
    #                 st.error(f"Execution Error: {e}")

    # ---------------- MODE 2: CUSTOM FILE EVALUATOR ----------------
    else:
        st.subheader("📂 Evaluate Custom Files")
        st.markdown("Upload your own prediction files to calculate metrics (Recall, NDCG, RAGAS, etc).")
        
        col_sel1, col_sel2 = st.columns([1, 2])
        
        with col_sel1:
            eval_selected_task = st.radio(
                "Task Type",
                ["A: Retrieval Only", "B: Generation", "C: Full RAG"],
                key="custom_eval_task_selector"
            )
            # Map UI selection to internal categories
            category_map = {
                "A: Retrieval Only": "Task A",
                "B: Generation": "Task B",
                "C: Full RAG": "Task C"
            }
            selected_category = category_map[eval_selected_task]

        # Dynamic Input Section
        eval_dataset_path = None
        beir_qrels = None
        beir_queries = None
        
        with col_sel2:
            # --- TASK A INPUTS ---
            if selected_category == "Task A":
                st.info("Task A requires a **Retrieval Predictions File (.jsonl)**. The Ground Truth (Qrels) will be automatically selected based on the 'Collection' field in your file.")
                
                # Upload Predictions
                task_a_file = st.file_uploader("Upload Retrieval Predictions (.jsonl)", type=["json", "jsonl"])
                if task_a_file:
                    # Save to persistent uploads folder
                    # Structure: data/evaluation/task_a/original_files
                    base_eval_dir = Path("data/evaluation") / "task_a"
                    uploads_dir = base_eval_dir / "original_files"
                    uploads_dir.mkdir(parents=True, exist_ok=True)
                    
                    filename = task_a_file.name
                    file_path = uploads_dir / filename
                    eval_dataset_path = str(file_path.absolute())
                    
                    # Check if file is new or changed
                    if "task_a_last_file" not in st.session_state or st.session_state.task_a_last_file != filename:
                        with open(file_path, "wb") as f:
                            f.write(task_a_file.getvalue())
                        
                        st.session_state.task_a_last_file = filename
                        st.success(f"Saved to {eval_dataset_path}")
                    else:
                        st.info(f"File ready: {eval_dataset_path}")

            # --- TASK B/C INPUTS ---
            else:
                st.info("Task B/C requires a **Test Dataset** (Input + Ground Truth).")
                eval_dataset = st.file_uploader("Upload Test Dataset (.jsonl)", type=["json", "jsonl"])
                if eval_dataset:
                    # Save to specific folder based on task
                    task_folder_name = "task_c" if selected_category == "Task C" else "task_b"
                    if selected_category not in ["Task B", "Task C"]:
                         task_folder_name = "others"
                    
                    # Define structure: data/evaluation/{task}/original_files
                    base_eval_dir = Path("data/evaluation") / task_folder_name
                    uploads_dir = base_eval_dir / "original_files"
                    uploads_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Use original filename (no timestamp prefix, as requested)
                    filename = eval_dataset.name
                    file_path = uploads_dir / filename
                    eval_dataset_path = str(file_path.absolute())

                    if "task_bc_last_file" not in st.session_state or st.session_state.task_bc_last_file != filename:
                        with open(file_path, "wb") as f:
                            f.write(eval_dataset.getvalue())
                        st.session_state.task_bc_last_file = filename
                        st.success(f"Saved to {eval_dataset_path}")
                    else:
                        st.info(f"File ready: {eval_dataset_path}")

# ... (inside render function logic later on) ...
                    with st.spinner("Running evaluation..."):
                    eval_start = time.time()
                    logger.info(f"Starting evaluation for {selected_category} with file {eval_dataset_path}")

                
                    # Use the currently running Python interpreter
                    import sys
                    venv_python = sys.executable
                    
                    if selected_category == "Task A":
                        # Task A: Run retrieval evaluation
                        st.subheader("📊 Task A Retrieval Evaluation")
                        
                        st.info(f"""
                        **Evaluation Configuration:**
                        - Corpus: **Auto-Detected**
                        - Query Type: **All Questions**
                        """)
                        
                        task_a_predictions_path = eval_dataset_path 
                        
                        if task_a_predictions_path:
                            output_path = task_a_predictions_path.replace(".jsonl", "_results.json").replace(".json", "_results.json")
                            
                            run_retrieval_eval_command = [
                                venv_python, "src/evaluation/run_retrieval_eval.py",
                                "--input_file", task_a_predictions_path,
                                "--output_file", output_path,
                            ]
                            
                            with st.status("Running Retrieval Evaluation...", expanded=True) as status:
                                st.write("📂 Input file:", task_a_predictions_path)
                                st.write("📊 Running official MTRAG retrieval evaluation...")
                                
                                try:
                                    # Get project root (parent of src/tabs/)
                                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                                    logger.debug(f"Executing Retrieval Eval Command: {run_retrieval_eval_command} in {project_root}")
                                    result = subprocess.run(
                                        run_retrieval_eval_command, 
                                        capture_output=True, 
                                        text=True,
                                        cwd=project_root)                                
                                    if result.returncode == 0:
                                        status.update(label="✅ Retrieval Evaluation Complete!", state="complete")
                                               
                                        aggregate_csv = os.path.splitext(output_path)[0] + "_aggregate.csv"
                                        if os.path.exists(aggregate_csv):
                                            df_agg = pd.read_csv(aggregate_csv)
                                            last_row = df_agg.iloc[-1]
                                            ndcg_values = ast.literal_eval(last_row['nDCG'])
                                            recall_values = ast.literal_eval(last_row['Recall'])
                                            row_data = {
                                                'R@1': recall_values[0],
                                                'R@3': recall_values[1],
                                                'R@5': recall_values[2],
                                                'R@10': recall_values[3],
                                                'nDCG@1': ndcg_values[0],
                                                'nDCG@3': ndcg_values[1],
                                                'nDCG@5': ndcg_values[2],
                                                'nDCG@10': ndcg_values[3]
                                            }
                                            display_df = pd.DataFrame([row_data])
                                            st.table(display_df)
                                            st.subheader("📋 Aggregate Results")
                                            st.dataframe(df_agg)

                                            
                                        if result.stdout:
                                            st.subheader("📈 Retrieval Metrics")
                                            st.code(result.stdout)
                                            logger.info("Task A Retrieval Evaluation completed successfully.")
                                        
                                        if os.path.exists(output_path):
                                            with open(output_path, "rb") as f:
                                                st.download_button(
                                                    label="📥 Download Enriched Results",
                                                    data=f,
                                                    file_name="retrieval_eval_results.jsonl",
                                                    mime="application/json"
                                                )
                                    else:
                                        status.update(label="❌ Evaluation Failed", state="error")
                                        st.error("Evaluation script failed")
                                        if result.stderr:
                                            st.error(result.stderr)
                                            
                                except Exception as e:
                                    status.update(label="❌ Execution Error", state="error")
                                    st.error(f"Failed to run subprocess: {e}")
                                    logger.error(f"Failed to run subprocess for Task A eval: {e}", exc_info=True)

                        else:
                            st.warning("⚠️ Please upload your retrieval predictions file above")

                        eval_time = time.time() - eval_start
                        st.caption(f"⏱️ Completed in {eval_time:.2f}s")
                    else:
                        # --- Task B/C: Generation Eval ---
                        # Determine task folder name
                        task_folder_name = "task_c" if selected_category == "Task C" else "task_b"
                        filename = Path(eval_dataset_path).name
                        
                        # Define evaluation root directory (at project root)
                        eval_root_dir = Path("data/evaluation") / task_folder_name
                        eval_root_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Set main output path (data/evaluation/task_x/filename.jsonl)
                        output_path = str(eval_root_dir / filename)

                        
                        run_gen_eval_command = [
                            venv_python, "src/evaluation/run_generation_eval.py",
                            "-i", eval_dataset_path,
                            "-o", output_path,
                            "-e", "src/evaluation/config.yaml",
                            "--provider", "lmstudio",
                            "--judge_model", judge_provider
                        ]
                        
                        # Initialize session state variables if they don't exist
                        if "gen_eval_file" not in st.session_state:
                            st.session_state.gen_eval_file = None
                        if "gen_eval_score" not in st.session_state:
                            st.session_state.gen_eval_score = None

                        # 1. RUN BUTTON
                        if st.button("🚀 Run Evaluation", type="primary"):
                            with st.status("Evaluating...", expanded=True) as status:
                                try:
                                    # --- Task C PRE-PROCESSING ---
                                    if selected_category == "Task C":
                                        status.write("🔄 Merging reference data (Task C preparation)...")
                                        logger.info("Triggering Task C merge logic...")
                                        
                                        # Define merged files directory
                                        merged_dir = eval_root_dir / "merged_files"
                                        merged_dir.mkdir(parents=True, exist_ok=True)
                                        
                                        merged_path = str(merged_dir / filename)
                                        
                                        # Assuming reference is at fixed path relative to CWD (project root)
                                        reference_path = "human/evaluations/reference.jsonl"
                                        
                                        try:
                                            merge_predictions_with_reference(reference_path, eval_dataset_path, merged_path)
                                            status.write(f"✅ Merge complete. Saved to: {merged_path}")
                                            status.write("Using merged file for evaluation.")
                                            
                                            # Update input file in command
                                            # Index 3 is the input file path (after -i)
                                            run_gen_eval_command[3] = merged_path
                                            
                                        except Exception as merge_err:
                                            logger.error(f"Merge failed: {merge_err}")
                                            status.write(f"⚠️ Merge failed: {merge_err}. Proceeding with original file...")

                                    # Run the script
                                    logger.debug(f"Executing Generation Eval Command: {run_gen_eval_command}")
                                    result = subprocess.run(run_gen_eval_command, capture_output=True, text=True)
                                    
                                    if result.returncode == 0:
                                        status.write("✅ Evaluation script finished. Calculating score...")
                                        status.write(f"Results saved to: {output_path}")
                                        
                                        if os.path.exists(output_path):
                                            # Calculate score and append to file
                                            final_score = append_ranking_score(output_path)
                                            
                                            # Save to Session State (This persists the result!)
                                            st.session_state.gen_eval_file = output_path
                                            st.session_state.gen_eval_score = final_score
                                            
                                            status.update(label="✅ Evaluation Complete!", state="complete")
                                        else:
                                            status.update(label="❌ Output file missing", state="error")
                                            st.error("Script finished but no output file was found.")
                                    else:
                                        status.update(label="❌ Evaluation Failed", state="error")
                                        st.error(result.stderr)
                                        
                                except Exception as e:
                                    status.update(label="❌ Execution Error", state="error")
                                    st.error(f"Subprocess failed: {e}")
                                    logger.error(f"Subprocess failed for Task B/C eval: {e}", exc_info=True)

                        # 2. PERSISTENT DOWNLOAD BUTTON (Outside the Run block)
                        # This checks if a file exists in session state and renders the button.
                        if st.session_state.gen_eval_file and os.path.exists(st.session_state.gen_eval_file):
                            st.divider()
                            st.success(f"Evaluation Ready! Ranking Score: **{st.session_state.gen_eval_score:.4f}**")
                            
                            with open(st.session_state.gen_eval_file, "rb") as f:
                                st.download_button(
                                    label="📥 Download Results (with Ranking Score)",
                                    data=f,
                                    file_name="eval_results_ranked.jsonl",
                                    mime="application/json",
                                    key="persistent_download_btn"  # Unique key is important
                                )
