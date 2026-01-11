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
from src.beir_utils import (
    AVAILABLE_CORPORA, QUERY_TYPES, get_retrieval_task_paths,
    load_qrels, load_queries, calculate_retrieval_metrics
)
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from app import logger

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
            
        collection_name = st.selectbox("Collections", collections_list, key="task_a_collections")

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
        
        if uploaded_file:
            try:
                    import time
                    suffix = ".jsonl" if uploaded_file.name.endswith(".jsonl") else ".json"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name

                    # Prepare Resources
                    rw_config = st.session_state.selected_components.get("rewriter_a", {})
                    llm_for_rewrite = None
                    if rw_config.get("enabled") and rw_config.get("method") in ["LLM-based", "Hybrid"]:
                        llm_for_rewrite = get_llm(provider, api_key, base_url, model_name)

                    # CRITICAL: Capture vector_store locally for threads
                    with open(f"collections/{collection_name}/info.json", "r", encoding="utf-8") as f:
                        collection_infos = json.load(f)
                    embedding_config["model_name"] = collection_infos["embedding"]["model_name"]
                    selected_collections = setup_vector_store(
                        documents=None, 
                        embedding_config=embedding_config, 
                        collection_name=collection_name,
                        db_config={}
                    )
                    vector_store_instance = selected_collections

                    # --- DEFINING THE WORKER FUNCTION ---
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

                            # B. Retrieve
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
                            
                            # D. Construct Output Object (Matches Task B Input Format)
                            output_obj = {
                                "conversation_id": item.get('conversation_id'), # Added
                                "task_id": item['id'],
                                "task_type": item.get('task_type', 'rag'),      # Added
                                "turn": item.get('turn'),                       # Added
                                "Collection": item['collection'],
                                "dataset": item.get('dataset', 'unknown'),      # Added
                                "contexts": contexts,                           # The new data
                                "input": item['original_input_obj'],            # Preserved input
                                
                                # Debugging metadata (ignored by evaluator)
                                "rewritten_query": final_query, 
                            }
                            return json.dumps(output_obj)
                        
                        except Exception as e:
                            logger.error(f"Error processing {item.get('id', 'unknown')}: {e}")
                            return None

                    # --- MAIN PROCESSING LOOP ---
                    st.write(f"Processing with {max_workers} threads...")
                    progress_bar = st.progress(0)
                    
                    items_to_process = []
                    
                    # 1. Parse File
                    with open(tmp_file_path, 'r', encoding='utf-8') as f:
                        for line_idx, line in enumerate(f):
                            if not line.strip(): continue
                            data = json.loads(line)
                            
                            # Parse MTRAG vs BEIR
                            parsed_item = {}
                            
                            # CASE 1: MTRAG (Has 'input' list - Rich Metadata)
                            if "input" in data and isinstance(data["input"], list):
                                conv = data["input"]
                                if not conv: continue
                                
                                parsed_item = {
                                    "id": data.get("task_id", f"line_{line_idx}"),
                                    "conversation_id": data.get("conversation_id", ""),
                                    "task_type": data.get("task_type", "rag"),
                                    "turn": data.get("turn", line_idx),
                                    "dataset": data.get("dataset", "unknown"),
                                    "collection": data.get("Collection", collection_name),
                                    "text": conv[-1]['text'], # Current query
                                    "history": conv[:-1],     # History
                                    "original_input_obj": conv
                                }
                            
                            # CASE 2: BEIR (Has 'text' and '_id' - Minimal Metadata)
                            elif "text" in data and "_id" in data:
                                parsed_item = {
                                    "id": data["_id"],
                                    "conversation_id": data["_id"], # Fallback
                                    "task_type": "rag",
                                    "turn": 1,
                                    "dataset": "BEIR",
                                    "collection": collection_name,
                                    "text": data["text"].replace("|user|:", "").replace("|agent|:", "").strip(),
                                    "history": [], 
                                    "original_input_obj": [{"speaker": "user", "text": data["text"]}]
                                }
                            else:
                                continue 
                            
                            items_to_process.append(parsed_item)

                    # 2. Parallel Execution
                    results_buffer = []
                    start_time = time.time()
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_item = {executor.submit(process_single_item, item): item for item in items_to_process}
                        
                        completed_count = 0
                        for future in concurrent.futures.as_completed(future_to_item):
                            result = future.result()
                            if result:
                                results_buffer.append(result)
                            
                            completed_count += 1
                            progress_bar.progress(completed_count / len(items_to_process))

                    total_time = time.time() - start_time
                    st.success(f"✅ Retrieval complete in {total_time:.2f}s")
                    
                    # 3. Save to Disk & Download
                    final_jsonl = "\n".join(results_buffer)
                    
                    predictions_dir = "predictions"
                    os.makedirs(predictions_dir, exist_ok=True)
                    
                    # Create a timestamped filename to avoid overwriting
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    save_filename = f"task_a_{collection_name}_{timestamp}.jsonl"
                    save_path = os.path.join(predictions_dir, save_filename)
                    
                    with open(save_path, "w", encoding="utf-8") as f:
                        f.write(final_jsonl)
                    
                    rewrite_dir = "query_rewrite_files"
                    os.makedirs(rewrite_dir , exist_ok=True)
                    # Create a timestamped filename to avoid overwriting
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    save_filename = f"task_a_{collection_name}_{uploaded_file.name.split('.')[0]}_{timestamp}.jsonl"
                    save_path = os.path.join(rewrite_dir, save_filename)



                    with open(save_path, "w") as outfile:
                        for line in results_buffer:
                            line = json.loads(line)
                            # select keys and rename them
                            new_data = {
                                "_id": line["task_id"],
                                "text": line["rewritten_query"]
                            }
                            outfile.write(json.dumps(new_data) + "\n")


                    st.success(f"✅ Predictions saved locally to: `{save_path}`")
                    # ----------------------

                    st.download_button(
                        label="📥 Download Retrieval Predictions",
                        data=final_jsonl,
                        file_name=save_filename,
                        mime="application/jsonl"
                    )
                    os.remove(tmp_file_path)

            except Exception as e:
                st.error(f"Error: {e}")

    # --- TASK B & C ---
    else:
        # get reference.jsonl
        uploaded_file = st.file_uploader("Upload input File", type=["json", "jsonl"])

        if uploaded_file:
            try:
                # Save uploaded file to temp file
                suffix = ".jsonl" if uploaded_file.name.endswith(".jsonl") else ".json"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                    st.session_state.test_file_path = tmp_file_path
            except Exception as e:
                logger.error(f"Error processing file: {e}", exc_info=True)
                st.error(f"Error processing file: {e}")
                        
        with st.expander("🤖 Generation Configuration", expanded=True):
            # LLM Model Lists
            OPENAI_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
            GEMINI_MODELS = ["gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-pro"]
            LOCAL_MODELS = ["QuantFactory/Meta-Llama-3-8B-Instruct-GGUF", "mistral-7b-instruct", "custom"]
            
            # Prompt Templates
            PROMPT_TEMPLATES = {
                "Default RAG": """"You are a helpful assistant. You must answer the user's question strictly using ONLY the information provided in the 'Reference Passages' section below. Rules: 1. If the 'Reference Passages' section is empty or does not contain the answer, you must strictly output: 'I do not know'. 2. Do not use your own internal knowledge. 3. Do not make up facts.""",
                "Concise": """Based on the context below, provide a brief, direct answer to the question.""",
                "Detailed": """You are a knowledgeable assistant. Analyze the provided context thoroughly and give a comprehensive, well-structured answer to the question. Include relevant details and explanations.""",
                "Custom": ""
            }
            
            gen_col1, gen_col2 = st.columns(2)
            
            with gen_col1:
                # LLM Provider is already selected in sidebar, show model selector
                st.markdown(f"**LLM Provider:** {provider}")
                
                if provider == "OpenAI":
                    gen_model = st.selectbox("Model", OPENAI_MODELS, key="gen_model_openai")
                elif provider == "Gemini":
                    gen_model = st.selectbox("Model", GEMINI_MODELS, key="gen_model_gemini")
                else:
                    gen_model = st.selectbox("Model", LOCAL_MODELS, key="gen_model_local")
                    if gen_model == "custom":
                        gen_model = st.text_input("Custom Model Name", value=model_name)
                
                temperature = st.slider("Temperature", 0.0, 2.0, 0.1, 0.1)
                max_tokens = st.slider("Max Tokens", 100, 4096, 1024, 100)
            
            with gen_col2:
                # Prompt Template Selection
                prompt_template_name = st.selectbox(
                    "Prompt Template",
                    list(PROMPT_TEMPLATES.keys()),
                    help="Select a pre-defined prompt template or create custom"
                )
                
                if prompt_template_name == "Custom":
                    gen_prompt_template = st.text_area(
                        "Custom Prompt Template",
                        value=PROMPT_TEMPLATES["Default RAG"],
                        height=200,
                        help="Use {context} and {question} placeholders"
                    )
                else:
                    gen_prompt_template = PROMPT_TEMPLATES[prompt_template_name]
                    with st.expander("📄 View Prompt Template"):
                        st.code(gen_prompt_template, language=None)
            
            st.session_state.selected_components["generator"] = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "provider": provider,
                "model": gen_model,
                "prompt_template_name": prompt_template_name,
                "prompt_template": gen_prompt_template
            }
    
        st.divider()

        if st.button("▶️ Run Pipeline", type="primary"):
            if selected_task == "C" and st.session_state.vector_store is None:
                st.error("❌ No vector store loaded. Please go to 'Knowledge Base'.")
            else:
                progress_container = st.empty()
                status_container = st.empty()
                
                try:
                    import time
                    from datetime import datetime
                    
                    start_time = time.time()
                    
                    # Capture config snapshot
                    config_snapshot = {
                        "timestamp": datetime.now().isoformat(),
                        "task": selected_task,
                        # "query": test_query,
                        "llm_provider": provider,
                        "llm_model": model_name,
                        "embedding_provider": embedding_provider,
                        "embedding_model": embedding_config.get("model_name", "default"),
                        "vector_db": vector_db_type,
                        "collection": collection_name,
                        "top_k": retrieval_top_k,
                        "components": st.session_state.selected_components
                    }
                    
                    run_result = {
                        "task": selected_task,
                        # "query": test_query,
                        "config_snapshot": config_snapshot,
                        "components": st.session_state.selected_components,
                        "status": "running",
                        "errors": []
                    }
                    
                    # Progress tracking
                    total_steps = 1  # Retrieval# 
                    if selected_task == "C":
                        total_steps += 1  # Rewrite
                    if selected_task in ["B", "C"]:
                        total_steps += 1  # Generation
                    
                    current_step = 0
                    
                    # Determine the query to use for retrieval
                    # retrieval_query = test_query 
                    
                    # Task C: Execute Query Rewrite first
                    if selected_task == "C":
                        rewriter_config = st.session_state.selected_components.get("rewriter", {})
                        rewrite_enabled = rewriter_config.get("enabled", True)
                        rewrite_method = rewriter_config.get("method", "LLM-based")
                        custom_prompt = rewriter_config.get("custom_prompt", None)
                        
                        # Get LLM for rewriting if needed
                        rewrite_llm = None
                        if rewrite_method in ["LLM-based", "Hybrid"]:
                            try:
                                rewrite_llm = get_llm(provider, api_key, base_url, model_name)
                            except Exception as e:
                                st.warning(f"Could not initialize LLM for rewriting: {e}")
                        
                        # Execute rewrite
                        rewrite_result = rewrite_query(
                            # query=test_query,
                            method=rewrite_method,
                            llm=rewrite_llm,
                            enabled=rewrite_enabled,
                            custom_prompt=custom_prompt
                        )
                        run_result["rewrite_result"] = rewrite_result
                        retrieval_query = rewrite_result["rewritten"]
                        
                        # Display rewrite results
                        st.subheader("✏️ Query Rewrite Result")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Original Query:**")
                            st.info(rewrite_result["original"])
                        with col2:
                            st.markdown("**Rewritten Query:**")
                            if rewrite_enabled:
                                st.success(rewrite_result["rewritten"])
                            else:
                                st.warning(f"{rewrite_result['rewritten']} (rewrite disabled)")
                        
                        st.caption(f"Method: {rewrite_result['method']} | Enabled: {rewrite_result['enabled']}")
                    
                    # ==================== TASK C: RETRIEVAL ====================
                    if selected_task in ["C"]:
                        retriever_config = st.session_state.selected_components.get("retriever", {})
                        top_k = retriever_config.get("top_k", 5)
                        search_type = retriever_config.get("search_type", "similarity")
                        
                        st.subheader("🔍 Retrieval Results")
                        
                        try:
                            # Execute retrieval with scores
                            retrieval_start = time.time()
                            
                            # Use similarity_search_with_score to get scores
                            docs_with_scores = st.session_state.vector_store.similarity_search_with_score(
                                retrieval_query, 
                                k=top_k
                            )
                            retrieval_time = time.time() - retrieval_start
                            
                            # Prepare results for storage
                            retrieval_results = []
                            for doc, score in docs_with_scores:
                                retrieval_results.append({
                                    "content": doc.page_content[:500],  # Truncate for storage
                                    "score": float(score),
                                    "metadata": doc.metadata
                                })
                            
                            run_result["retrieval"] = {
                                "query_used": retrieval_query,
                                "top_k": top_k,
                                "search_type": search_type,
                                "num_results": len(docs_with_scores),
                                "retrieval_time_ms": round(retrieval_time * 1000, 2),
                                "results": retrieval_results
                            }
                            
                            # Store retrieved docs for potential generation use
                            run_result["retrieved_docs"] = [doc for doc, _ in docs_with_scores]
                            
                            # Display retrieval stats
                            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                            with stat_col1:
                                st.metric("📄 Results", len(docs_with_scores))
                            with stat_col2:
                                st.metric("⏱️ Time", f"{round(retrieval_time * 1000, 2)} ms")
                            with stat_col3:
                                st.metric("🎯 Top-K", top_k)
                            with stat_col4:
                                if docs_with_scores:
                                    avg_score = sum(s for _, s in docs_with_scores) / len(docs_with_scores)
                                    st.metric("📊 Avg Score", f"{avg_score:.4f}")
                            
                            # Display retrieved documents with scores
                            if docs_with_scores:
                                st.markdown("---")
                                st.markdown("### 📋 Retrieved Chunks")
                                
                                for i, (doc, score) in enumerate(docs_with_scores):
                                    # Create a card-like container for each result
                                    with st.container():
                                        # Header row with rank, score, and source
                                        header_col1, header_col2, header_col3 = st.columns([1, 2, 3])
                                        
                                        with header_col1:
                                            st.markdown(f"### #{i+1}")
                                        
                                        with header_col2:
                                            # Score with progress bar visualization
                                            # Note: Lower score = better match for some DBs (distance), higher = better for others (similarity)
                                            st.markdown(f"**Score:** `{score:.4f}`")
                                            # Normalize score for progress bar (assuming similarity, higher is better)
                                            normalized = min(1.0, max(0.0, 1 - score if score > 1 else score))
                                            st.progress(normalized)
                                        
                                        with header_col3:
                                            # Source metadata summary
                                            source = doc.metadata.get('source', doc.metadata.get('title', 'Unknown'))
                                            doc_id = doc.metadata.get('id', 'N/A')
                                            st.markdown(f"**Source:** {source}")
                                            st.caption(f"ID: {doc_id}")
                                    
                                    # Chunk text in expander
                                    with st.expander(f"📄 View Chunk Text ({len(doc.page_content)} chars)", expanded=(i==0)):
                                        # Chunk content
                                        st.markdown("**Chunk Text:**")
                                        st.code(doc.page_content, language=None)
                                        
                                        # Full metadata
                                        st.markdown("**Source Metadata:**")
                                        meta_cols = st.columns(2)
                                        with meta_cols[0]:
                                            for key in ['id', 'source', 'title', 'chunk_id']:
                                                if key in doc.metadata:
                                                    st.write(f"• **{key}:** {doc.metadata[key]}")
                                        with meta_cols[1]:
                                            for key in doc.metadata:
                                                if key not in ['id', 'source', 'title', 'chunk_id']:
                                                    st.write(f"• **{key}:** {doc.metadata[key]}")
                                        
                                        # Raw JSON
                                        with st.expander("📋 Raw Metadata JSON"):
                                            st.json(doc.metadata)
                                    
                                    st.markdown("---")
                            else:
                                st.warning("No documents retrieved. Try adjusting your query or Top-K value.")
                                
                        except Exception as e:
                            st.error(f"❌ Retrieval failed: {e}")
                            logger.error(f"Retrieval error: {e}", exc_info=True)
                            run_result["retrieval"] = {"error": str(e)}

                    # ==================== TASK B ====================
                    if selected_task in ["B"]:
                        st.subheader("🤖 Generation")
                        
                        # Check if we have retrieved docs
                        retrieved_docs = run_result.get("retrieved_docs", [])
                        # TODO: fix if not retrieved_docs
                        if retrieved_docs:
                            st.warning("⚠️ No retrieved documents to use as context. Skipping generation.")
                        else:
                            try:
                                gen_config = st.session_state.selected_components.get("generator", {})
                                gen_model = gen_config.get("model", model_name)
                                gen_temperature = gen_config.get("temperature", 0.1)
                                gen_max_tokens = gen_config.get("max_tokens", 1024)
                                prompt_template = gen_config.get("prompt_template", "")
                                prompt_template_name = gen_config.get("prompt_template_name", "Default RAG")
                                
                                task_b_output = []
                                
                                # Initialize LLM
                                gen_start = time.time()
                                llm = get_llm(provider, api_key, base_url, gen_model)
                                
                                with open(st.session_state.get("test_file_path", ""), 'r', encoding='utf-8') as f:
                                    for line_number, line in enumerate(f):
                                        if not line.strip():
                                            continue
                                        
                                        # 1. Parse the JSON line
                                        data = json.loads(line)
                                        retrieved_docs = data.get('contexts', [])
                                        context_parts = []
                                        for i, doc in enumerate(retrieved_docs):
                                            title = doc.get('title', 'Unknown Title')
                                            text = doc.get('text', '')
                                            context_parts.append(f"Document [{i+1}] (Title: {title}):\n{text}")
                                        
                                        full_context_str = "\n\n".join(context_parts)
                                        
                                        # 'input' contains the conversation history
                                        # The structure is a list of dictionaries with "speaker" and "text"
                                        conversation_turns = data.get('input', [])
                                        
                                        if not conversation_turns:
                                            continue

                                        # The last item in 'input' is the current user query
                                        last_turn = conversation_turns[-1]
                                        current_query = last_turn['text']

                                        # Everything before the last item is history
                                        history_turns = conversation_turns[:-1]
                                        # --- BUILD PROMPT ---
                
                                        system_instruction = prompt_template
                                        # Create the message list for the LLM
                                        # 1. System instruction with the Documents (Context)
                                        messages = [
                                            SystemMessage(content=f"{system_instruction}\n\n### REFERENCE PASSAGES:\n{full_context_str}")
                                        ]
                                        
                                        # 2. Add Conversation History (Crucial for Multi-Turn understanding)
                                        for turn in history_turns:
                                            speaker = turn.get('speaker')
                                            text = turn.get('text')
                                            if speaker == 'user':
                                                messages.append(HumanMessage(content=text))
                                            elif speaker == 'agent':
                                                messages.append(AIMessage(content=text))
                                        
                                        # 3. Add the final User Query
                                        messages.append(HumanMessage(content=current_query))
                                        
                                        # --- GENERATE RESPONSE ---
                                        
                                        # invoke the LLM
                                        # Note: Ensure your 'llm' object is initialized before running this
                                        try:
                                            ai_response = llm.invoke(messages, temperature=gen_temperature)
                                            prediction = ai_response.content

                                        except Exception as e:
                                            prediction = "Error generating response."
                                        
                                        # Generate response
                                        # response = llm.invoke([HumanMessage(content=formatted_prompt)])
                                        gen_time = time.time() - gen_start
                                        
                                        # Extract answer
                                        # answer = response.content if hasattr(response, 'content') else str(response)
                                        # print("\nanswer: ", answer)
                                        # Store generation result
                                        run_result["generation"] = {
                                            "model": gen_model,
                                            "provider": provider,
                                            "temperature": gen_temperature,
                                            "max_tokens": gen_max_tokens,
                                            "prompt_template_name": prompt_template_name,
                                            "context_length": len(full_context_str),
                                            "num_context_docs": len(retrieved_docs),
                                            "answer": prediction,
                                            "generation_time_ms": round(gen_time * 1000, 2)
                                        }
                                        
                                        data["predictions"] = [
                                            {
                                                "text": prediction
                                            }
                                        ]
                                        task_b_output.append(json.dumps(data, ensure_ascii=False))
                                            
                            except Exception as e:
                                st.error(f"❌ Generation failed: {e}")
                                logger.error(f"Generation error: {e}", exc_info=True)
                                run_result["generation"] = {"error": str(e)}

                    final_jsonl_content = "\n".join(task_b_output)
                    st.session_state.gen_result_file_ready = True
                    st.session_state.gen_result_final_content = final_jsonl_content
                    
                    # Calculate total time
                    total_time = time.time() - start_time
                    run_result["total_time_ms"] = round(total_time * 1000, 2)
                    
                    # ==================== TASK C: FULL PIPELINE SUMMARY ====================
                    if selected_task == "C":
                        st.markdown("---")
                        st.subheader("📊 Full RAG Pipeline Summary (Task C)")
                        
                        # Pipeline visualization
                        st.markdown("""
                        ```
                        ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
                        │   REWRITE   │ → │  RETRIEVAL  │ → │ GENERATION  │
                        └─────────────┘    └─────────────┘    └─────────────┘
                        ```
                        """)
                        
                        # Summary cards
                        summary_col1, summary_col2, summary_col3 = st.columns(3)
                        
                        with summary_col1:
                            st.markdown("#### ✏️ Rewrite Stage")
                            rewrite_data = run_result.get("rewrite_result", {})
                            st.write(f"**Method:** {rewrite_data.get('method', 'N/A')}")
                            st.write(f"**Enabled:** {'✅' if rewrite_data.get('enabled') else '❌'}")
                            st.write(f"**Query Changed:** {'✅' if rewrite_data.get('original') != rewrite_data.get('rewritten') else '❌'}")
                        
                        with summary_col2:
                            st.markdown("#### 🔍 Retrieval Stage")
                            retrieval_data = run_result.get("retrieval", {})
                            st.write(f"**Docs Retrieved:** {retrieval_data.get('num_results', 0)}")
                            st.write(f"**Time:** {retrieval_data.get('retrieval_time_ms', 0)} ms")
                            st.write(f"**Top-K:** {retrieval_data.get('top_k', 0)}")
                        
                        with summary_col3:
                            st.markdown("#### 🤖 Generation Stage")
                            gen_data = run_result.get("generation", {})
                            st.write(f"**Model:** {gen_data.get('model', 'N/A')[:15]}...")
                            st.write(f"**Time:** {gen_data.get('generation_time_ms', 0)} ms")
                            st.write(f"**Answer Length:** {len(gen_data.get('answer', ''))} chars")
                        
                        # Intermediate outputs expander
                        with st.expander("📋 All Intermediate Outputs", expanded=False):
                            st.markdown("#### 1️⃣ Rewritten Query")
                            rewrite_data = run_result.get("rewrite_result", {})
                            st.code(rewrite_data.get("rewritten", test_query), language=None)
                            
                            st.markdown("---")
                            st.markdown("#### 2️⃣ Retrieved Documents")
                            retrieval_data = run_result.get("retrieval", {})
                            results = retrieval_data.get("results", [])
                            for i, res in enumerate(results[:3]):  # Show first 3
                                st.markdown(f"**[{i+1}] Score: {res.get('score', 0):.4f}**")
                                st.text(res.get("content", "")[:200] + "...")
                            if len(results) > 3:
                                st.caption(f"... and {len(results) - 3} more documents")
                            
                            st.markdown("---")
                            st.markdown("#### 3️⃣ Final Answer")
                            gen_data = run_result.get("generation", {})
                            st.markdown(gen_data.get("answer", "No answer generated"))
                    
                    # Update status
                    run_result["status"] = "completed"
                    st.session_state.run_result = run_result
                    st.success(f"✅ Pipeline executed for Task {selected_task} in {round(total_time * 1000, 2)} ms!")
                
                except Exception as e:
                    st.error(f"❌ Pipeline execution failed: {str(e)}")
                    logger.error(f"Pipeline error: {e}", exc_info=True)
                    with st.expander("🐛 Error Details"):
                        import traceback
                        st.code(traceback.format_exc(), language="python")
        if st.session_state.gen_result_file_ready:
            st.download_button(
                            label="📥 Download Task B Predictions",
                            data=st.session_state.gen_result_final_content,
                            file_name="predictions.jsonl",
                            mime="application/jsonl"
                        )
