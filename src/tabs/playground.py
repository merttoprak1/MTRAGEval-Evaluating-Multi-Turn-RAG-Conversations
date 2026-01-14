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
from pathlib import Path
from src.ingestion import load_json_documents, load_beir_queries
from src.vector_store import setup_vector_store, get_retriever, add_to_vector_store, delete_from_vector_store
from src.llm_client import get_llm
from src.rag import create_rag_chain
from src.query_rewrite import rewrite_query, DEFAULT_REWRITE_PROMPT, CONTEXTUAL_REWRITE_PROMPT

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from app import logger
import time

PROMPT_TEMPLATES = """
You are an expert generative AI assistant in a RAG pipeline.

When given a user query:
1. Always prioritize *only information from the retrieved documents* included in the context.
2. Do not hallucinate facts or invent details not supported by those documents.
3. If the retrieved context lacks necessary information, acknowledge it clearly.
4. If there is no answer in the retrieved contents answer like this: I'm sorry, but I don't have the answer to your question.
5. Follow these steps:
   a. Briefly summarize the relevant retrieved content linked to the query.
   b. Integrate that content into a coherent answer.
   c. Cite or reference the source document or snippet when appropriate.
6. Use precise, clear language and format answers for readability.
7. Do not write for the indicating like, answer, source and any others
Answer the user's question now using the retrieved context.

"""

# ============================================================================
# MODULAR TASK FUNCTIONS
# ============================================================================

def run_task_a_retrieval(
    input_file_path: str,
    collections_list: list,
    embedding_config: dict,
    rw_config: dict,
    llm_for_rewrite,
    task_a_top_k: int,
    max_workers: int,
    output_filename_prefix: str = "task_a",
    output_dir: str = "predictions/task_a",
    rewrite_dir: str = "predictions/task_a"
) -> tuple[str, str]:
    """
    Execute Task A retrieval logic.
    
    Returns:
        tuple: (predictions_path, rewrite_path) - paths to saved output files
    """
    # Collection mapping helper
    def map_collection_name(input_collection: str) -> str:
        """Map input Collection field to local collection folder name."""
        input_lower = input_collection.lower()
        for local_col in collections_list:
            local_lower = local_col.lower()
            if 'clapnq' in input_lower and 'clapnq' in local_lower:
                return local_col
            if 'govt' in input_lower and 'govt' in local_lower:
                return local_col
            if 'fiqa' in input_lower and 'fiqa' in local_lower:
                return local_col
            if ('cloud' in input_lower or 'ibmcloud' in input_lower) and ('cloud' in local_lower or 'ibmcloud' in local_lower):
                return local_col
        return collections_list[0] if collections_list else input_collection
    
    # Collection cache for dynamic loading
    collection_cache = {}
    
    def get_vector_store_for_collection(collection_field: str):
        """Get or create vector store for a collection, with caching."""
        local_collection = map_collection_name(collection_field)
        
        if local_collection in collection_cache:
            return collection_cache[local_collection], local_collection
        
        try:
            with open(f"collections/{local_collection}/info.json", "r", encoding="utf-8") as f:
                collection_infos = json.load(f)
            embed_cfg = embedding_config.copy()
            embed_cfg["model_name"] = collection_infos["embedding"]["model_name"]
            vs = setup_vector_store(
                documents=None, 
                embedding_config=embed_cfg, 
                collection_name=local_collection,
                db_config={}
            )
            collection_cache[local_collection] = vs
            return vs, local_collection
        except Exception as e:
            logger.error(f"Failed to load collection {local_collection}: {e}")
            return None, local_collection
    
    # Worker function
    def process_single_item(item):
        try:
            # A. Rewrite
            final_query = item['text']
            if rw_config.get("enabled"):
                rewrite_result = rewrite_query(
                    query=item['text'],
                    method=rw_config.get("method"),
                    llm=llm_for_rewrite,
                    enabled=True,
                    history=item['history'], 
                    custom_prompt=rw_config.get("custom_prompt")
                )
                final_query = rewrite_result['rewritten']

            # B. Get vector store and Retrieve
            vector_store_instance, local_col = get_vector_store_for_collection(item['collection'])
            if vector_store_instance is None:
                # Return error dict
                return {"error": f"No vector store for collection {item['collection']}"}
            
            docs_with_scores = vector_store_instance.similarity_search_with_score(final_query, k=task_a_top_k)
            
            # C. Format Contexts
            contexts = []
            for doc, score in docs_with_scores:
                contexts.append({
                    "document_id": doc.metadata.get("id", "unknown_id"),
                    "score": float(score),
                    "text": doc.page_content,
                    "title": doc.metadata.get("title", "No Title")
                })
            
            # D. Construct Output Object
            output_obj = {
                "conversation_id": item.get('conversation_id'),
                "task_id": item['id'],
                "task_type": item.get('task_type', 'rag'),
                "turn": item.get('turn'),
                "Collection": item['collection'],
                "dataset": item.get('dataset', 'unknown'),
                "contexts": contexts,
                "input": item['original_input_obj'],
                "rewritten_query": final_query,
            }
            return json.dumps(output_obj)
        
        except Exception as e:
            # Capture specific error info
            return {"error": str(e), "type": type(e).__name__}
    
    # Parse input file
    items_to_process = []
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if not line.strip(): continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # Parse MTRAG format only
            if "input" in data and isinstance(data["input"], list) and data["input"]:
                conv = data["input"]
                parsed_item = {
                    "id": data.get("task_id", f"line_{line_idx}"),
                    "conversation_id": data.get("conversation_id", ""),
                    "task_type": data.get("task_type", "rag"),
                    "turn": data.get("turn", line_idx),
                    "dataset": data.get("dataset", "unknown"),
                    "collection": data.get("Collection", ""),
                    "text": conv[-1]['text'],
                    "history": conv[:-1],
                    "original_input_obj": conv
                }
                items_to_process.append(parsed_item)
            else:
                pass # Skip non-MTRAG lines silently or log warning if needed? 
                # Original logged warning, let's keep it clean or just skip.
                # The user verified their file is MTRAG, so clean skip is fine or warning.
                # Original had warning.
                # logger.warning(f"Skipping line {line_idx}: not in MTRAG format")
    
    if not items_to_process:
        logger.warning(f"No valid items found in {input_file_path}")
        return "", ""

    # Parallel execution
    results_buffer = []
    errors_encountered = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(process_single_item, item): item for item in items_to_process}
        
        for future in concurrent.futures.as_completed(future_to_item):
            try:
                result = future.result()
                if isinstance(result, dict) and "error" in result:
                     errors_encountered.append(result["error"])
                elif result:
                    results_buffer.append(result)
            except Exception as e:
                errors_encountered.append(str(e))
                
    # Report errors to UI if significant failed
    if errors_encountered:
        unique_errors = list(set(errors_encountered))
        error_msg = f"Encountered {len(errors_encountered)} errors during processing. Unique errors: {unique_errors[:3]}"
        logger.error(error_msg)
        # We can't use st.error here directly easily if it's running in background, 
        # but since this function is blocked on compatible with st, we can.
        # But for better separation, let's just log it and maybe the caller handles it? 
        # The caller (render) doesn't see this return.
        # Let's print to streamlit here since we are inside the app flow.
        st.error(error_msg)

    # Save predictions
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_filename = f"{output_filename_prefix}_multi_{timestamp}.jsonl"
    save_path = os.path.join(output_dir, save_filename)
    
    if results_buffer:
        final_jsonl = "\n".join(results_buffer)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(final_jsonl)
    else:
        # Save empty file or hint?
        # Better to save an error note or just empty
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("")

    # Save rewrite queries
    os.makedirs(rewrite_dir, exist_ok=True)
    rewrite_filename = f"{output_filename_prefix}_rewrite_{timestamp}.jsonl"
    rewrite_path = os.path.join(rewrite_dir, rewrite_filename)
    
    with open(rewrite_path, "w") as outfile:
        for line in results_buffer:
            try:
                line_data = json.loads(line)
                new_data = {
                    "_id": line_data["task_id"],
                    "text": line_data["rewritten_query"]
                }
                outfile.write(json.dumps(new_data) + "\n")
            except:
                pass
    
    return save_path, rewrite_path


def run_task_b_generation(
    input_file_path: str,
    gen_prompt_template: str,
    llm,
    max_workers: int,
    output_filename_prefix: str = "task_b",
    output_dir: str = "predictions/task_b"
) -> str:
    """
    Execute Task B generation logic.
    
    Returns:
        str: Path to saved output file
    """
    def process_single_record(line: str) -> str | None:
        if not line.strip():
            return None

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        retrieved_docs = data.get("contexts", [])
        conversation_turns = data.get("input", [])

        if not conversation_turns:
            return None

        last_turn = conversation_turns[-1]
        current_query = last_turn.get("text", "")
        history_turns = conversation_turns[:-1]

        messages = [SystemMessage(content=gen_prompt_template)]

        for turn in history_turns:
            speaker = turn.get("speaker")
            text = turn.get("text", "")
            if speaker == "user":
                messages.append(HumanMessage(content=text))
            elif speaker == "agent":
                messages.append(AIMessage(content=text))

        context_blocks = []
        for i, doc in enumerate(retrieved_docs, start=1):
            title = doc.get("title", "Unknown Title")
            text = doc.get("text", "")
            context_blocks.append(
                f"Retrieval Doc {i}\nTitle: {title}\n{text}"
            )

        user_last_message = "\n\n".join(context_blocks)
        user_last_message += f"\n\nUser Question: {current_query}"

        messages.append(HumanMessage(content=user_last_message))

        try:
            ai_response = llm.invoke(messages)
            prediction = ai_response.content
        except Exception:
            prediction = "Error generating response."

        data["predictions"] = [{"text": prediction}]
        return json.dumps(data, ensure_ascii=False)
    
    # Read input file
    with open(input_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Parallel execution
    task_b_output = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_record, line)
            for line in lines
        ]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                task_b_output.append(result)
    
    # Save output
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_filename = f"{output_filename_prefix}_{timestamp}.jsonl"
    save_path = os.path.join(output_dir, save_filename)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(task_b_output))
    
    return save_path


# ============================================================================
# MAIN RENDER FUNCTION
# ============================================================================

def render():
    st.header("🎯 Interactive Playground")
    st.subheader("Global LLM Provider")
    llm_col1, llm_col2, llm_col3 = st.columns(3)
    
    with llm_col1:
        provider = st.selectbox("Select LLM Provider", ["OpenAI", "Gemini", "Local"], index=2, key="llm_type")
    
    with llm_col2:
        api_key = None
        base_url = None
        model_name = "gpt-3.5-turbo"
        if provider == "OpenAI":
            api_key = st.text_input("OpenAI API Key", type="password", key="llm_type_openai")
        elif provider == "Gemini":
            api_key = st.text_input("Google API Key", type="password", key="llm_type_gemini")
        else:
            base_url = st.text_input("Local LLM Base URL", value="http://localhost:1234/v1",key="llm_type_local")
    with llm_col3:
        model_name = st.text_input("Custom Model Name", key="llm_model_name", value="openai/gpt-oss-20b")
    st.divider()
    st.subheader("Global Embedding Provider")
            
    embedding_provider = st.selectbox("Embedding Provider", ["OpenAI", "Gemini", "Local"], index=2)
    if embedding_provider in ("OpenAI", "Gemini"):
        embedding_api_key = st.text_input(f"{embedding_provider} Embedding API Key", type="password")
        embedding_config = {"provider": embedding_provider, "api_key": embedding_api_key}
    else:
        embed_base_url = st.text_input("Embedding Base URL", value="http://localhost:11434")
        embedding_config = {"provider": "Local", "base_url": embed_base_url}
    st.divider()

    # Task Selector
    task_options = {
        None: "-- Select a Task --",
        "A": "Task A: Retrieval Only",
        "B": "Task B: Generation",
        "C": "Task C: Rewrite + Retrieval + Generation"
    }
    
    selected_task = st.selectbox(
        "Select Task",
        options=list(task_options.keys()),
        format_func=lambda x: task_options[x],
        key="task_selector"
    )
    st.session_state.selected_task = selected_task
    st.divider()

    collections_list = os.listdir("collections")

    if selected_task is None:
        st.info("👆 Please select a task to begin.")

    # --- TASK A: BATCH RETRIEVAL ---
    elif selected_task == "A":
        st.subheader("📂 Batch Retrieval (Task A)")
        st.markdown("Upload a Query File. Supports **BEIR format** (e.g., `clapnq_lastturn.jsonl`) or **Task A/C Input format** (e.g., `retrieval_taskac_input.jsonl`).")
        
        # --- Task A Configuration Columns ---
        conf_col1, conf_col2 = st.columns([1, 1])
        
        with conf_col1:
            # Threading Configuration
            max_workers = st.slider("Concurrent Worker Threads", min_value=1, max_value=16, value=4, help="Increase this for higher throughput if your LLM/Server can handle parallel requests.")

        with conf_col2:
                # Retrieval Settings Override
                task_a_top_k = st.number_input("Top-K Documents", min_value=1, max_value=100, value=10)
        
        # Collection mapping: maps input Collection names to local collection folders
        def map_collection_name(input_collection: str) -> str:
            """Map input Collection field to local collection folder name."""
            input_lower = input_collection.lower()
            for local_col in collections_list:
                local_lower = local_col.lower()
                # Check if key parts match (e.g., 'clapnq', 'govt', 'fiqa', 'cloud')
                if 'clapnq' in input_lower and 'clapnq' in local_lower:
                    return local_col
                if 'govt' in input_lower and 'govt' in local_lower:
                    return local_col
                if 'fiqa' in input_lower and 'fiqa' in local_lower:
                    return local_col
                if ('cloud' in input_lower or 'ibmcloud' in input_lower) and ('cloud' in local_lower or 'ibmcloud' in local_lower):
                    return local_col
            # Fallback: return first collection or the input as-is
            return collections_list[0] if collections_list else input_collection
        
        st.info(f"📁 Available collections: {', '.join(collections_list)}")

        # --- Query Rewrite Configuration ---
        with st.expander("✏️ Query Rewrite Configuration", expanded=True):
            rewrite_enabled = st.checkbox("Enable Query Rewriting", value=True, key="rw_enable_a")
            rewrite_method = st.selectbox("Rewrite Method", ["LLM-based", "Rule-based", "Hybrid"], key="rw_method_a")
            
            custom_prompt = None
            if rewrite_method in ["LLM-based", "Hybrid"]:
                prompt_type = st.radio("Prompt Type", ["Default/Contextual", "Custom"], horizontal=True, key="rw_prompt_type_a")
                if prompt_type == "Custom":
                    custom_prompt = st.text_area("Custom Prompt", value=CONTEXTUAL_REWRITE_PROMPT, key="rw_custom_prompt_a")
                else:
                    st.caption("ℹ️ Uses 'Contextual Prompt' if history exists, otherwise 'Default Prompt'.")
            
            st.session_state.selected_components["rewriter_a"] = {
                "enabled": rewrite_enabled, 
                "method": rewrite_method, 
                "custom_prompt": custom_prompt
            }

        uploaded_file = st.file_uploader("Upload Query File (JSONL)", type=["json", "jsonl"], key="task_a_uploader")
        
        # Check if new file uploaded, reset output
        if uploaded_file and (
            "task_a_file_id" not in st.session_state or 
            st.session_state.task_a_file_id != uploaded_file.file_id
        ):
            st.session_state.task_a_file_id = uploaded_file.file_id
            st.session_state.task_a_output = None
        
        # Run button
        run_pipeline = st.button("🚀 Run Task A Pipeline", disabled=(uploaded_file is None))
        
        if uploaded_file and run_pipeline:
            try:
                suffix = ".jsonl" if uploaded_file.name.endswith(".jsonl") else ".json"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                # Prepare Resources
                rw_config = st.session_state.selected_components.get("rewriter_a", {})
                llm_for_rewrite = None
                if rw_config.get("enabled") and rw_config.get("method") in ["LLM-based", "Hybrid"]:
                    llm_for_rewrite = get_llm(provider, api_key, base_url, model_name)

                st.write(f"Processing with {max_workers} threads...")
                progress_bar = st.progress(0)
                start_time = time.time()
                
                # Use modular function
                predictions_path, rewrite_path = run_task_a_retrieval(
                    input_file_path=tmp_file_path,
                    collections_list=collections_list,
                    embedding_config=embedding_config,
                    rw_config=rw_config,
                    llm_for_rewrite=llm_for_rewrite,
                    task_a_top_k=task_a_top_k,
                    max_workers=max_workers,
                    output_filename_prefix="task_a",
                    output_dir="predictions/task_a",
                    rewrite_dir="predictions/task_a"
                )
                
                progress_bar.progress(1.0)
                total_time = time.time() - start_time
                st.success(f"✅ Retrieval complete in {total_time:.2f}s")
                st.success(f"✅ Predictions saved locally to: `{predictions_path}`")
                st.success(f"✅ Rewrite queries saved to: `{rewrite_path}`")

                # Store output for download button
                st.session_state.task_a_output = predictions_path
                
                os.remove(tmp_file_path)

            except Exception as e:
                st.error(f"Error: {e}")
        
        # Download button (shown if output exists)
        if st.session_state.get("task_a_output"):
            try:
                with open(st.session_state.task_a_output, "r", encoding="utf-8") as f:
                    final_jsonl = f.read()

                st.download_button(
                    label="📥 Download Retrieval Predictions",
                    data=final_jsonl,
                    file_name=os.path.basename(st.session_state.task_a_output),
                    mime="application/jsonl",
                    key="download_task_a_predictions"
                )
            except Exception as e:
                st.error(f"Error reading output file: {e}")

    elif selected_task == "B":
        st.subheader("🤖 Generation (Task B)")
        gen_prompt_template = st.text_area(
            "Custom Prompt Template",
            value=PROMPT_TEMPLATES,
            height=200
        )
        max_workers = st.slider("Concurrent Worker Threads", min_value=1, max_value=12, value=4, help="Increase this for higher throughput if your LLM/Server can handle parallel requests.")
        uploaded_file = st.file_uploader("Upload input File", type=["json", "jsonl"], key="task_b_uploader")

        # Check if new file uploaded, reset output
        if uploaded_file and (
            "task_b_file_id" not in st.session_state or 
            st.session_state.task_b_file_id != uploaded_file.file_id
        ):
            st.session_state.task_b_file_id = uploaded_file.file_id
            st.session_state.task_b_output = None
        
        # Run button
        run_pipeline = st.button("🚀 Run Task B Pipeline", disabled=(uploaded_file is None))

        if uploaded_file and run_pipeline:
            try:
                # Save uploaded file to temp file
                suffix = ".jsonl" if uploaded_file.name.endswith(".jsonl") else ".json"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Initialize LLM
                llm = get_llm(provider, api_key, base_url, model_name)
                
                st.write(f"Processing with {max_workers} threads...")
                progress_bar = st.progress(0)
                start_time = time.time()
                
                # Use modular function
                save_path = run_task_b_generation(
                    input_file_path=tmp_file_path,
                    gen_prompt_template=gen_prompt_template,
                    llm=llm,
                    max_workers=max_workers,
                    output_filename_prefix=f"task_b_{uploaded_file.name.split('.')[0]}",
                    output_dir="predictions/task_b"
                )
                
                progress_bar.progress(1.0)
                total_time = time.time() - start_time
                st.success(f"✅ Generation complete in {total_time:.2f}s")
                st.success(f"✅ Predictions saved locally to: `{save_path}`")
                
                # Store output for download button
                st.session_state.task_b_output = save_path
                
                os.remove(tmp_file_path)
                
            except Exception as e:
                logger.error(f"Error processing file: {e}", exc_info=True)
                st.error(f"Error processing file: {e}")
        
        # Download button (shown if output exists)
        if st.session_state.get("task_b_output"):
            try:
                with open(st.session_state.task_b_output, "r", encoding="utf-8") as f:
                    output_data = f.read()
                
                st.download_button(
                    label="📥 Download Generation Predictions",
                    data=output_data,
                    file_name=os.path.basename(st.session_state.task_b_output),
                    mime="application/jsonl",
                    key="download_task_b_predictions"
                )
            except Exception as e:
                st.error(f"Error reading output file: {e}")

    elif selected_task == "C":
        st.subheader("🔄 Rewrite + Retrieval + Generation (Task C)")
        st.markdown("Upload a Query File to run **Task A (Retrieval)** followed by **Task B (Generation)**.")
        
        # --- Task C Configuration ---
        conf_col1, conf_col2 = st.columns([1, 1])
        
        with conf_col1:
            # Threading Configuration for Retrieval
            max_workers_retrieval = st.slider("Retrieval Worker Threads", min_value=1, max_value=16, value=4, help="Concurrent threads for retrieval step.")
        
        with conf_col2:
            # Threading Configuration for Generation
            max_workers_generation = st.slider("Generation Worker Threads", min_value=1, max_value=12, value=4, help="Concurrent threads for generation step.")
        
        # Retrieval Settings
        task_c_top_k = st.number_input("Top-K Documents", min_value=1, max_value=100, value=10)
        
        st.info(f"📁 Available collections: {', '.join(collections_list)}")
        
        # --- Query Rewrite Configuration ---
        with st.expander("✏️ Query Rewrite Configuration", expanded=True):
            rewrite_enabled = st.checkbox("Enable Query Rewriting", value=True, key="rw_enable_c")
            rewrite_method = st.selectbox("Rewrite Method", ["LLM-based", "Rule-based", "Hybrid"], key="rw_method_c")
            
            custom_prompt = None
            if rewrite_method in ["LLM-based", "Hybrid"]:
                prompt_type = st.radio("Prompt Type", ["Default/Contextual", "Custom"], horizontal=True, key="rw_prompt_type_c")
                if prompt_type == "Custom":
                    custom_prompt = st.text_area("Custom Prompt", value=CONTEXTUAL_REWRITE_PROMPT, key="rw_custom_prompt_c")
                else:
                    st.caption("ℹ️ Uses 'Contextual Prompt' if history exists, otherwise 'Default Prompt'.")
            
            st.session_state.selected_components["rewriter_c"] = {
                "enabled": rewrite_enabled, 
                "method": rewrite_method, 
                "custom_prompt": custom_prompt
            }
        
        # --- Generation Prompt Configuration ---
        with st.expander("🤖 Generation Prompt Configuration", expanded=True):
            gen_prompt_template = st.text_area(
                "Custom Prompt Template",
                value=PROMPT_TEMPLATES,
                height=200,
                key="gen_prompt_c"
            )
        
        # File Upload
        uploaded_file = st.file_uploader("Upload Query File (JSONL)", type=["json", "jsonl"], key="task_c_uploader")
        
        # Check if new file uploaded, reset output path
        if uploaded_file and (
            "task_c_file_id" not in st.session_state or 
            st.session_state.task_c_file_id != uploaded_file.file_id
        ):
            st.session_state.task_c_file_id = uploaded_file.file_id
            st.session_state.task_c_output = None
        
        # Run button
        run_pipeline = st.button("🚀 Run Task C Pipeline", disabled=(uploaded_file is None))
        
        if uploaded_file and run_pipeline:
            try:
                suffix = ".jsonl" if uploaded_file.name.endswith(".jsonl") else ".json"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # ===== STEP 1: RETRIEVAL (Task A) =====
                st.write("### Step 1: Retrieval")
                st.write(f"Processing retrieval with {max_workers_retrieval} threads...")
                
                # Prepare Resources
                rw_config = st.session_state.selected_components.get("rewriter_c", {})
                llm_for_rewrite = None
                if rw_config.get("enabled") and rw_config.get("method") in ["LLM-based", "Hybrid"]:
                    llm_for_rewrite = get_llm(provider, api_key, base_url, model_name)
                
                progress_bar_retrieval = st.progress(0)
                start_time_retrieval = time.time()
                
                # Run Task A
                predictions_path, rewrite_path = run_task_a_retrieval(
                    input_file_path=tmp_file_path,
                    collections_list=collections_list,
                    embedding_config=embedding_config,
                    rw_config=rw_config,
                    llm_for_rewrite=llm_for_rewrite,
                    task_a_top_k=task_c_top_k,
                    max_workers=max_workers_retrieval,
                    output_filename_prefix="task_c_retrieval",
                    output_dir="predictions/task_c/retrieval",
                    rewrite_dir="predictions/task_c/retrieval"
                )
                
                progress_bar_retrieval.progress(1.0)
                total_time_retrieval = time.time() - start_time_retrieval
                st.success(f"✅ Retrieval complete in {total_time_retrieval:.2f}s")
                st.info(f"📄 Retrieval output saved to: `{predictions_path}`")
                
                # ===== STEP 2: GENERATION (Task B) =====
                st.write("### Step 2: Generation")
                st.write(f"Processing generation with {max_workers_generation} threads...")
                
                # Initialize LLM for generation
                llm = get_llm(provider, api_key, base_url, model_name)
                
                progress_bar_generation = st.progress(0)
                start_time_generation = time.time()
                
                # Run Task B using Task A output
                generation_output_path = run_task_b_generation(
                    input_file_path=predictions_path,
                    gen_prompt_template=gen_prompt_template,
                    llm=llm,
                    max_workers=max_workers_generation,
                    output_filename_prefix=f"task_c_{uploaded_file.name.split('.')[0]}",
                    output_dir="predictions/task_c/generation"
                )
                
                progress_bar_generation.progress(1.0)
                total_time_generation = time.time() - start_time_generation
                st.success(f"✅ Generation complete in {total_time_generation:.2f}s")
                
                # Summary
                st.write("---")
                st.write("### Task C Summary")
                total_time = total_time_retrieval + total_time_generation
                st.success(f"✅ Task C complete in {total_time:.2f}s (Retrieval: {total_time_retrieval:.2f}s, Generation: {total_time_generation:.2f}s)")
                st.success(f"✅ Final predictions saved to: `{generation_output_path}`")
                st.info(f"📄 Intermediate retrieval output: `{predictions_path}`")
                st.info(f"📄 Rewrite queries: `{rewrite_path}`")
                
                # Store output path for download button
                st.session_state.task_c_output = generation_output_path
                
                # Cleanup temp file
                os.remove(tmp_file_path)
                
            except Exception as e:
                logger.error(f"Error in Task C: {e}", exc_info=True)
                st.error(f"Error in Task C: {e}")
        
        # Download button (shown if output exists)
        if st.session_state.get("task_c_output"):
            try:
                with open(st.session_state.task_c_output, "r", encoding="utf-8") as f:
                    final_output = f.read()
                
                st.download_button(
                    label="📥 Download Task C Final Predictions",
                    data=final_output,
                    file_name=os.path.basename(st.session_state.task_c_output),
                    mime="application/jsonl",
                    key="download_task_c_final"
                )
            except Exception as e:
                st.error(f"Error reading output file: {e}")
