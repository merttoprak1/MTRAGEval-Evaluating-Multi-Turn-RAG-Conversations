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
from src.vector_store_retrieval import get_vector_store
from src.llm_client import get_llm
from src.rag import create_rag_chain
from src.query_rewrite import rewrite_query, DEFAULT_REWRITE_PROMPT, CONTEXTUAL_REWRITE_PROMPT
from src.file_manager import FileManager
from src.filename_utils import generate_prediction_filename, generate_taskc_generation_filename

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from app import logger
from src.reranker import get_reranker, RERANKER_TYPES, FLASHRANK_MODELS, BGE_MODELS
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

def run_task_a_retrieval(
    input_file_path: str,
    active_collections: list,
    embedding_config: dict,
    rw_config: dict,
    llm_for_rewrite,
    task_a_top_k: int,
    max_workers: int,
    output_filename_prefix: str = "task_a",
    output_dir: str = "predictions/task_a",
    rewrite_dir: str = "predictions/task_a",
    rerank_enabled: bool = False,
    rerank_top_k: int = 5,
    reranker_type: str = "flashrank",
    reranker_model: str = None,
    vector_db_type: str = None,
    embedding_model: str = None,
    global_llm_name: str = None
) -> str:
    """
    Execute Task A retrieval logic.
    
    Returns:
        str: Path to saved predictions file
    """
    # Collection mapping helper
    def map_collection_target(input_collection_name: str) -> dict:
        """
        Map input 'Collection' string to one of the user-selected active collections.
        Returns a dict: {'name': str, 'db': str, 'model': str}
        """
        input_lower = input_collection_name.lower()
        
        # 1. Exact or Partial Match on Name against active collections
        for col_info in active_collections:
            local_name = col_info['name'].lower()
            
            # Specific dataset keywords common in benchmarks
            if 'clapnq' in input_lower and 'clapnq' in local_name: return col_info
            if 'govt' in input_lower and 'govt' in local_name: return col_info
            if 'fiqa' in input_lower and 'fiqa' in local_name: return col_info
            if ('cloud' in input_lower or 'ibmcloud' in input_lower) and ('cloud' in local_name or 'ibmcloud' in local_name): return col_info
            
            # General fallback: is the input name inside the local name?
            if input_lower in local_name:
                return col_info

        # 2. Fallback: Return the first selected collection if no match found
        return active_collections[0] if active_collections else None

    # Collection cache for dynamic loading
    collection_cache = {}
    
    def get_vector_store_for_collection(collection_field: str):
        """Get or create vector store for a collection, with caching."""
        
        # Determine which local collection to use
        target_info = map_collection_target(collection_field)
        
        if not target_info:
            return None, "No Collection Selected"

        # Unique cache key based on path components
        cache_key = f"{target_info['db']}_{target_info['model']}_{target_info['name']}"
        
        if cache_key in collection_cache:
            return collection_cache[cache_key], target_info['name']
        
        try:
            # Use FileManager to get the correct path
            col_path = FileManager.get_collection_path(
                target_info['db'], 
                target_info['model'], 
                target_info['name']
            )
            
            # Load the Bridge Info
            info_path = os.path.join(col_path, "info.json")
            if not os.path.exists(info_path):
                 logger.error(f"info.json missing at {info_path}")
                 return None, target_info['name']

            with open(info_path, "r", encoding="utf-8") as f:
                collection_infos = json.load(f)
            
            # 1. Setup Embedding Config
            embed_cfg = embedding_config.copy()
            if "embedding" in collection_infos:
                embed_cfg["model_name"] = collection_infos["embedding"].get("model_name", embed_cfg.get("model_name"))
                if "dimension" in collection_infos["embedding"]:
                    embed_cfg["dimension"] = collection_infos["embedding"]["dimension"]

            # 2. Setup DB Config
            db_info = collection_infos.get("collection", {})
            db_type = db_info.get("vector_db_type", "FAISS")
            
            # 3. Initialize Correct Store
            # 3. Initialize Correct Store
            vs = get_vector_store(
                embedding_config=embed_cfg, 
                collection_name=target_info['name'],
                db_type=db_type,
                db_config=db_info
            )
            
            collection_cache[cache_key] = vs
            return vs, target_info['name']
            
        except Exception as e:
            logger.error(f"Failed to load collection {target_info['name']}: {e}")
            return None, target_info['name']
    
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
            
            # D. Reranking (if enabled)
            if rerank_enabled and contexts:
                try:
                    reranker = get_reranker(reranker_type=reranker_type, model_name=reranker_model)
                    logger.info(f"Reranking {len(contexts)} documents with {reranker_type}:{reranker_model}, top_k={rerank_top_k}")
                    contexts = reranker.rerank(final_query, contexts, top_k=rerank_top_k)
                    # Log sample rerank score to verify it was added
                    if contexts and 'rerank_score' in contexts[0]:
                        logger.info(f"Reranking successful. Sample score: {contexts[0].get('rerank_score')}")
                    else:
                        logger.warning("Reranking completed but no rerank_score found in contexts")
                except Exception as e:
                    logger.warning(f"Reranking failed, using original order: {e}")
            
            # E. Construct Output Object
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
                "rerank_enabled": rerank_enabled
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
        st.error(error_msg)

    # Save predictions
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename_base = generate_prediction_filename(
        task_type=output_filename_prefix,
        timestamp=timestamp,
        vector_db_type=vector_db_type,
        embedding_model=embedding_model,
        top_k=task_a_top_k,
        has_reranker=rerank_enabled,
        reranker_model=reranker_model if rerank_enabled else None,
        top_k_reranker=rerank_top_k if rerank_enabled else None,
        global_llm_name=global_llm_name,
        query_rewritten=rw_config.get("enabled", False)
    )
    save_filename = f"{filename_base}.jsonl"
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
    
    return save_path


def run_task_b_generation(
    input_file_path: str,
    gen_prompt_template: str,
    llm,
    max_workers: int,
    output_filename_prefix: str = "task_b",
    output_dir: str = "predictions/task_b",
    global_llm_name: str = None,
    # Optional retrieval params for Task C
    vector_db_type: str = None,
    embedding_model: str = None,
    top_k: int = None,
    has_reranker: bool = False,
    reranker_model: str = None,
    top_k_reranker: int = None,
    query_rewritten: bool = False
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
    
    # Use Task C format if retrieval params are provided, otherwise Task B format
    if vector_db_type or embedding_model or top_k is not None:
        # Task C generation - use shorter notation with retrieval params
        filename_base = generate_taskc_generation_filename(
            task_type=output_filename_prefix,
            timestamp=timestamp,
            vector_db_type=vector_db_type,
            embedding_model=embedding_model,
            top_k=top_k,
            has_reranker=has_reranker,
            reranker_model=reranker_model if has_reranker else None,
            top_k_reranker=top_k_reranker if has_reranker else None,
            global_llm_name=global_llm_name,
            query_rewritten=query_rewritten
        )
    else:
        # Task B generation - simple format
        filename_base = generate_prediction_filename(
            task_type=output_filename_prefix,
            timestamp=timestamp,
            global_llm_name=global_llm_name
        )
    
    save_filename = f"{filename_base}.jsonl"
    save_path = os.path.join(output_dir, save_filename)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(task_b_output))
    
    return save_path

# MAIN RENDER FUNCTION

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
        "C": "Task C: Retrieval + Generation"
    }
    
    selected_task = st.selectbox(
        "Select Task",
        options=list(task_options.keys()),
        format_func=lambda x: task_options[x],
        key="task_selector"
    )
    st.session_state.selected_task = selected_task
    st.divider()

    # --- COLLECTION SELECTION UI ---
    active_collections = []
    
    # Only show collection selector for Retrieval tasks
    if selected_task in ["A", "C"]:
        st.subheader("🗂️ Active Collection Selection")
        
        # 1. DB Type
        avail_dbs = []
        if os.path.exists(FileManager.BASE_COLLECTIONS_DIR):
            avail_dbs = [d for d in os.listdir(FileManager.BASE_COLLECTIONS_DIR) if os.path.isdir(os.path.join(FileManager.BASE_COLLECTIONS_DIR, d))]
        
        sel_db = st.selectbox("1. Select Vector DB", avail_dbs if avail_dbs else ["No DB Found"])
        
        # 2. Embedding Model
        avail_models = []
        if sel_db and sel_db != "No DB Found":
             db_path = os.path.join(FileManager.BASE_COLLECTIONS_DIR, sel_db)
             avail_models = [d for d in os.listdir(db_path) if os.path.isdir(os.path.join(db_path, d))]
        
        sel_model = st.selectbox("2. Select Embedding Model", avail_models if avail_models else ["No Models Found"])
        
        # 3. Collections (Multi-select)
        avail_cols = []
        if sel_model and sel_model != "No Models Found":
            avail_cols = FileManager.list_collections(sel_db, sel_model)
            
        selected_col_names = st.multiselect("3. Select Collections", avail_cols, default=avail_cols[:1] if avail_cols else None)
        
        # Construct the active_collections list of dicts
        for name in selected_col_names:
            active_collections.append({
                "name": name,
                "db": sel_db,
                "model": sel_model
            })
            
        if not active_collections:
            st.warning("⚠️ No collections selected. Retrieval will fail.")
        else:
            st.info(f"✅ Active collections: {', '.join([c['name'] for c in active_collections])}")

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
        
        # --- Reranker Configuration ---
        with st.expander("🔄 Reranker Configuration", expanded=False):
            rerank_enabled_a = st.checkbox("Enable Reranking", value=False, key="rerank_enable_a")
            
            col1, col2 = st.columns(2)
            with col1:
                reranker_type_a = st.selectbox(
                    "Reranker Type",
                    options=list(RERANKER_TYPES.keys()),
                    format_func=lambda x: RERANKER_TYPES[x],
                    key="reranker_type_a",
                    disabled=not rerank_enabled_a
                )
            with col2:
                # Show model options based on selected reranker type
                if reranker_type_a == "flashrank":
                    model_options = FLASHRANK_MODELS
                else:
                    model_options = BGE_MODELS
                
                reranker_model_a = st.selectbox(
                    "Reranker Model",
                    options=list(model_options.keys()),
                    format_func=lambda x: model_options[x],
                    key="reranker_model_a",
                    disabled=not rerank_enabled_a
                )
            
            rerank_top_k_a = st.number_input("Rerank Top-K", min_value=1, max_value=20, value=5, key="rerank_top_k_a", disabled=not rerank_enabled_a)
            
            if reranker_type_a == "bge":
                st.warning("⚠️ BGE requires sentence-transformers and PyTorch. Make sure they are installed.")
            else:
                st.caption("ℹ️ FlashRank is lightweight and fast, uses ONNX for CPU inference.")

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
                predictions_path = run_task_a_retrieval(
                    input_file_path=tmp_file_path,
                    active_collections=active_collections,
                    embedding_config=embedding_config,
                    rw_config=rw_config,
                    llm_for_rewrite=llm_for_rewrite,
                    task_a_top_k=task_a_top_k,
                    max_workers=max_workers,
                    output_filename_prefix="task_a",
                    output_dir="predictions/task_a",
                    rewrite_dir="predictions/task_a",
                    rerank_enabled=rerank_enabled_a,
                    rerank_top_k=rerank_top_k_a,
                    reranker_type=reranker_type_a,
                    reranker_model=reranker_model_a,
                    vector_db_type=sel_db if sel_db != "No DB Found" else None,
                    embedding_model=sel_model if sel_model != "No Models Found" else None,
                    global_llm_name=model_name
                )
                
                progress_bar.progress(1.0)
                total_time = time.time() - start_time
                st.success(f"✅ Retrieval complete in {total_time:.2f}s")
                st.success(f"✅ Predictions saved locally to: `{predictions_path}`")

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
                    output_filename_prefix="task_b",
                    output_dir="predictions/task_b",
                    global_llm_name=model_name
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
        
        # --- Reranker Configuration ---
        with st.expander("🔄 Reranker Configuration", expanded=False):
            rerank_enabled_c = st.checkbox("Enable Reranking", value=False, key="rerank_enable_c")
            
            col1, col2 = st.columns(2)
            with col1:
                reranker_type_c = st.selectbox(
                    "Reranker Type",
                    options=list(RERANKER_TYPES.keys()),
                    format_func=lambda x: RERANKER_TYPES[x],
                    key="reranker_type_c",
                    disabled=not rerank_enabled_c
                )
            with col2:
                # Show model options based on selected reranker type
                if reranker_type_c == "flashrank":
                    model_options_c = FLASHRANK_MODELS
                else:
                    model_options_c = BGE_MODELS
                
                reranker_model_c = st.selectbox(
                    "Reranker Model",
                    options=list(model_options_c.keys()),
                    format_func=lambda x: model_options_c[x],
                    key="reranker_model_c",
                    disabled=not rerank_enabled_c
                )
            
            rerank_top_k_c = st.number_input("Rerank Top-K", min_value=1, max_value=20, value=5, key="rerank_top_k_c", disabled=not rerank_enabled_c)
            
            if reranker_type_c == "bge":
                st.warning("⚠️ BGE requires sentence-transformers and PyTorch. Make sure they are installed.")
            else:
                st.caption("ℹ️ FlashRank is lightweight and fast, uses ONNX for CPU inference.")
        
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
                predictions_path = run_task_a_retrieval(
                    input_file_path=tmp_file_path,
                    active_collections=active_collections,
                    embedding_config=embedding_config,
                    rw_config=rw_config,
                    llm_for_rewrite=llm_for_rewrite,
                    task_a_top_k=task_c_top_k,
                    max_workers=max_workers_retrieval,
                    output_filename_prefix="task_c",
                    output_dir="predictions/task_c/retrieval",
                    rewrite_dir="predictions/task_c/retrieval",
                    rerank_enabled=rerank_enabled_c,
                    rerank_top_k=rerank_top_k_c,
                    reranker_type=reranker_type_c,
                    reranker_model=reranker_model_c,
                    vector_db_type=sel_db if sel_db != "No DB Found" else None,
                    embedding_model=sel_model if sel_model != "No Models Found" else None,
                    global_llm_name=model_name
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
                    output_filename_prefix="task_c",
                    output_dir="predictions/task_c/generation",
                    global_llm_name=model_name,
                    # Pass retrieval params for Task C filename
                    vector_db_type=sel_db if sel_db != "No DB Found" else None,
                    embedding_model=sel_model if sel_model != "No Models Found" else None,
                    top_k=task_c_top_k,
                    has_reranker=rerank_enabled_c,
                    reranker_model=reranker_model_c if rerank_enabled_c else None,
                    top_k_reranker=rerank_top_k_c if rerank_enabled_c else None,
                    query_rewritten=rw_config.get("enabled", False)
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
