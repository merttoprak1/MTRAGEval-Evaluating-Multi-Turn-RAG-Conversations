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
Answer the user’s question now using the retrieved context.

"""
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

    elif selected_task == "B":
        gen_prompt_template = st.text_area(
            "Custom Prompt Template",
            value=PROMPT_TEMPLATES,
            height=200
        )
        max_workers = st.slider("Concurrent Worker Threads", min_value=1, max_value=12, value=8, help="Increase this for higher throughput if your LLM/Server can handle parallel requests.")
        uploaded_file = st.file_uploader("Upload input File", type=["json", "jsonl"])

        if uploaded_file:
            try:
                # Save uploaded file to temp file
                suffix = ".jsonl" if uploaded_file.name.endswith(".jsonl") else ".json"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
            except Exception as e:
                logger.error(f"Error processing file: {e}", exc_info=True)
                st.error(f"Error processing file: {e}")

            st.subheader("🤖 Generation")

            import concurrent.futures
            

            def process_single_record(
                line: str,
                gen_prompt_template: str,
                llm
            ) -> str | None:
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
            # Initialize LLM ONCE
            llm = get_llm(provider, api_key, base_url, model_name)

            with open(tmp_file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            task_b_output = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        process_single_record,
                        line,
                        gen_prompt_template,
                        llm
                    )
                    for line in lines
                ]

                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        task_b_output.append(result)
            predictions_dir = "predictions/generation"
            os.makedirs(predictions_dir, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_filename = f"task_b_{uploaded_file.name.split('.')[0]}_{timestamp}.jsonl"
            save_path = os.path.join(predictions_dir, save_filename)

            with open(save_path, "w", encoding="utf-8") as f:
                f.write("\n".join(task_b_output))
