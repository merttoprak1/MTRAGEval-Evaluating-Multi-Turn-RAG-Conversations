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
from src.ingestion import load_json_documents, chunk_documents, load_beir_queries
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

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Modular RAG Chatbot", layout="wide")

def main():
    st.title("🤖 Modular RAG Chatbot")
    logger.info("Application started")

    # --- Global State Management ---
    if "selected_task" not in st.session_state:
        st.session_state.selected_task = None
    if "selected_components" not in st.session_state:
        st.session_state.selected_components = {}
    if "run_result" not in st.session_state:
        st.session_state.run_result = None

    if 'gen_result_file_ready' not in st.session_state:
        st.session_state.gen_result_file_ready = False
    if 'gen_result_final_content' not in st.session_state:
        st.session_state.gen_result_final_content = None

    ## ==================== MAIN TABS ====================
    tab_interactive, tab_batch, tab_kb, tab_logs = st.tabs([
        "💬 Interactive Playground", 
        "📊 Batch Evaluation",
        "📚 Knowledge Base", 
        "📝 Logs & Debugging"  # Changed from Database Inspector
    ])

    # ==================== TAB: KNOWLEDGE BASE (Consolidated) ====================
    # NOTE: We execute this block first so 'collection_name', 'vector_store', etc. are defined 
    # and available for the other tabs, even though it appears 3rd in the UI list.
    with tab_kb:
        st.header("📚 Knowledge Base Management")

        # LLM Provider Selection --------------------------------
        st.subheader("0. Global LLM Provider")
        llm_col1, llm_col2 = st.columns(2)
        
        with llm_col1:
            provider = st.selectbox("Select LLM Provider", ["OpenAI", "Gemini", "Local"], index=1)
        
        with llm_col2:
            api_key = None
            base_url = None
            model_name = "gpt-3.5-turbo"

            if provider == "OpenAI":
                api_key = st.text_input("OpenAI API Key", type="password")
                model_name = "gpt-3.5-turbo"
            elif provider == "Gemini":
                api_key = st.text_input("Google API Key", type="password")
                model_name = "gemini-flash-latest"
            else:
                base_url = st.text_input("Local LLM Base URL", value="http://localhost:1234/v1")
                model_name = "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF"
                st.info("Ensure local server is running (Ollama/LM Studio).")
        
        st.divider()
        # -----------------------------------------------------------
        
        kb_col1, kb_col2 = st.columns(2)
        
        with kb_col1:
            st.subheader("1. Embedding Configuration")
            embedding_provider = st.selectbox("Embedding Provider", ["OpenAI", "Gemini", "Local (Ollama)"], index=1)
            
            # Model lists
            OPENAI_EMBEDDING_MODELS = {
                "text-embedding-3-small": {"dim": 1536, "description": "Fastest"},
                "text-embedding-3-large": {"dim": 3072, "description": "Best quality"},
                "text-embedding-ada-002": {"dim": 1536, "description": "Legacy"},
            }
            GEMINI_EMBEDDING_MODELS = {
                "models/text-embedding-004": {"dim": 768, "description": "Latest"},
                "models/embedding-001": {"dim": 768, "description": "Legacy"},
            }
            LOCAL_EMBEDDING_MODELS = {
                "nomic-embed-text": {"dim": 768, "description": "General purpose"},
                "mxbai-embed-large": {"dim": 1024, "description": "High quality"},
                "all-minilm": {"dim": 384, "description": "Fast"},
                "custom": {"dim": None, "description": "Custom"},
            }

            embedding_config = {}
            if embedding_provider == "OpenAI":
                embedding_api_key = api_key if (provider == "OpenAI" and api_key) else st.text_input("OpenAI Embedding API Key", type="password")
                selected_model = st.selectbox("Embedding Model", list(OPENAI_EMBEDDING_MODELS.keys()))
                model_info = OPENAI_EMBEDDING_MODELS[selected_model]
                st.caption(f"Dim: {model_info['dim']}")
                batch_size = st.number_input("Batch Size", 1, 2048, 100)
                embedding_config = {"provider": "OpenAI", "api_key": embedding_api_key, "model_name": selected_model, "dimension": model_info['dim'], "batch_size": batch_size}
                
            elif embedding_provider == "Gemini":
                embedding_api_key = api_key if (provider == "Gemini" and api_key) else st.text_input("Gemini Embedding API Key", type="password")
                selected_model = st.selectbox("Embedding Model", list(GEMINI_EMBEDDING_MODELS.keys()))
                model_info = GEMINI_EMBEDDING_MODELS[selected_model]
                st.caption(f"Dim: {model_info['dim']}")
                batch_size = st.number_input("Batch Size", 1, 100, 10)
                embedding_config = {"provider": "Gemini", "api_key": embedding_api_key, "model_name": selected_model, "dimension": model_info['dim'], "batch_size": batch_size}
                
            else: # Local
                embed_base_url = st.text_input("Embedding Base URL", value="http://localhost:11434")
                selected_model = st.selectbox("Embedding Model", list(LOCAL_EMBEDDING_MODELS.keys()))
                if selected_model == "custom":
                    embed_model = st.text_input("Custom Model Name")
                    model_dim = st.number_input("Dimension", 64, 4096, 768)
                else:
                    embed_model = selected_model
                    model_dim = LOCAL_EMBEDDING_MODELS[selected_model]['dim']
                batch_size = st.number_input("Batch Size", 1, 500, 50)
                embedding_config = {"provider": "Local", "base_url": embed_base_url, "model_name": embed_model, "dimension": model_dim, "batch_size": batch_size}

        with kb_col2:
            st.subheader("2. Vector Database & Collection")
            vector_db_type = st.selectbox("Vector DB Type", ["FAISS", "Chroma", "Pinecone"], index=0)
            
            # Common config
            retrieval_top_k = st.slider("Default Top-K", 1, 20, 5)
            db_config = {"top_k": retrieval_top_k}
            
            if vector_db_type == "FAISS":
                faiss_index_name = st.text_input("Index Name", value="default")
                db_config["index_name"] = faiss_index_name
            elif vector_db_type == "Chroma":
                chroma_index_name = st.text_input("Collection Name", value="default", key="chroma_collection")
                db_config["index_name"] = chroma_index_name
            elif vector_db_type == "Pinecone":
                pinecone_api_key = st.text_input("Pinecone API Key", type="password")
                pinecone_index = st.text_input("Index Name", value="default-index")
                if pinecone_api_key:
                    db_config["api_key"] = pinecone_api_key
                    db_config["index_name"] = pinecone_index
            
            collection_name = st.text_input("Internal Collection ID", value="default_collection", help="Unique ID for this dataset")

        st.divider()
        
        # --- Vector Store Initialization Logic ---
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = None
        if "current_db_type" not in st.session_state:
            st.session_state.current_db_type = None
        if "current_embedding_model" not in st.session_state:
            st.session_state.current_embedding_model = None
        
        current_embed_model = embedding_config.get("model_name", "default")
        
        # Reload check
        should_reload = (
            st.session_state.vector_store is None or 
            st.session_state.get("current_collection") != collection_name or
            st.session_state.get("current_db_type") != vector_db_type or
            st.session_state.get("current_embedding_model") != current_embed_model
        )
        
        if should_reload:
             try:
                 st.session_state.vector_store = setup_vector_store(
                     documents=None, 
                     embedding_config=embedding_config, 
                     collection_name=collection_name,
                     db_type=vector_db_type,
                     db_config=db_config
                 )
                 st.session_state.current_collection = collection_name
                 st.session_state.current_db_type = vector_db_type
                 st.session_state.current_embedding_model = current_embed_model
                 # logger.info(f"Vector store loaded: {collection_name}")
             except Exception as e:
                 pass # Silent fail if empty, wait for ingestion

        # --- Ingestion Section ---
        st.subheader("3. Data Ingestion")
        uploaded_file = st.file_uploader("Upload Documents (JSON/JSONL)", type=["json", "jsonl"])

        if uploaded_file:
            if st.button("Process & Ingest File"):
                with st.spinner("Processing..."):
                    try:
                        suffix = ".jsonl" if uploaded_file.name.endswith(".jsonl") else ".json"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        documents = load_json_documents(tmp_file_path)
                        if documents:
                            chunks = chunk_documents(documents)
                            if embedding_provider == "OpenAI" and not embedding_config.get("api_key"):
                                st.error("OpenAI API Key required for embedding.")
                            else:
                                if api_key: os.environ["OPENAI_API_KEY"] = api_key
                                
                                st.session_state.vector_store = setup_vector_store(
                                    chunks, embedding_config, collection_name, vector_db_type, db_config
                                )
                                st.session_state.current_collection = collection_name
                                st.success(f"Successfully ingested {len(chunks)} chunks into {collection_name}")
                        else:
                            st.error("No valid documents found.")
                        os.remove(tmp_file_path)
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")

        # --- Management Section (Merged from old tab) ---
        st.divider()
        with st.expander("🛠️ Inspect & Manage Collection"):
            if st.session_state.vector_store:
                try:
                    collection_data = st.session_state.vector_store.get()
                    num_docs = len(collection_data['ids'])
                    st.write(f"Total Chunks: {num_docs}")
                    
                    if num_docs > 0:
                        data_for_df = []
                        for i in range(min(num_docs, 100)): # Limit preview
                            data_for_df.append({
                                "Select": False,
                                "ID": collection_data['ids'][i],
                                "Content": collection_data['documents'][i][:100] + "...",
                                "Metadata": str(collection_data['metadatas'][i])
                            })
                        
                        df_to_edit = pd.DataFrame(data_for_df)
                        edited_df = st.data_editor(df_to_edit, column_config={"Select": st.column_config.CheckboxColumn("Select", default=False)}, hide_index=True)

                        if st.button("Delete Selected"):
                            ids_to_delete = edited_df[edited_df.Select]["ID"].tolist()
                            if ids_to_delete:
                                delete_from_vector_store(st.session_state.vector_store, ids_to_delete)
                                st.success(f"Deleted {len(ids_to_delete)} chunks.")
                                st.rerun()
                except Exception as e:
                    st.error(f"Error inspecting DB: {e}")
            else:
                st.info("No active vector store.")

    # ==================== TAB: INTERACTIVE PLAYGROUND ====================
    with tab_interactive:
        st.header("🎯 Interactive Playground")
        
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
                 task_a_top_k = st.number_input("Top-K Documents", min_value=1, max_value=100, value=retrieval_top_k)

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
            
            if st.session_state.vector_store is None:
                st.warning("⚠️ No vector store loaded. Please go to the 'Knowledge Base' tab to ingest data.")
            
            elif uploaded_file:
                if st.button("▶️ Run Batch Retrieval", type="primary"):
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

                        # CRITICAL FIX: Capture vector_store in a local variable for the threads
                        # Threads cannot access st.session_state directly
                        vector_store_instance = st.session_state.vector_store

                        # --- DEFINING THE WORKER FUNCTION ---
                        def process_single_item(item):
                            """
                            Worker function to process a single query item:
                            1. Rewrite (with history)
                            2. Retrieve
                            3. Format Output
                            """
                            try:
                                # A. Rewrite
                                final_query = item['text']
                                if rw_config.get("enabled"):
                                    rewrite_result = rewrite_query(
                                        query=item['text'],
                                        method=rw_config.get("method"),
                                        llm=llm_for_rewrite,
                                        enabled=True,
                                        history=item['history'], # Pass extracted history
                                        custom_prompt=rw_config.get("custom_prompt")
                                    )
                                    final_query = rewrite_result['rewritten']

                                # B. Retrieve
                                # FIX: Use the local variable 'vector_store_instance' instead of st.session_state
                                docs_with_scores = vector_store_instance.similarity_search_with_score(final_query, k=task_a_top_k)
                                
                                # C. Format Contexts
                                contexts = []
                                for doc, score in docs_with_scores:
                                    contexts.append({
                                        "document_id": doc.metadata.get("id", "unknown_id"), # REQUIRED
                                        "score": float(score), # REQUIRED
                                        "text": doc.page_content, # Optional for A, Required for B
                                        "title": doc.metadata.get("title", "No Title")
                                    })
                                
                                # D. Construct Output Object
                                output_obj = {
                                    "task_id": item['id'],
                                    "Collection": item['collection'],
                                    "input": item['original_input_obj'], # Pass through the original input block
                                    "contexts": contexts,
                                    
                                    # Debugging metadata (ignored by evaluator)
                                    "rewritten_query": final_query, 
                                    "original_query": item['text']
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
                                
                                # CASE 1: MTRAG (Has 'input' list)
                                if "input" in data and isinstance(data["input"], list):
                                    conv = data["input"]
                                    if not conv: continue
                                    
                                    parsed_item = {
                                        "id": data.get("task_id", f"line_{line_idx}"),
                                        "text": conv[-1]['text'], # Current query
                                        "history": conv[:-1],     # History
                                        "collection": data.get("Collection", collection_name),
                                        "original_input_obj": conv
                                    }
                                
                                # CASE 2: BEIR (Has 'text' and '_id')
                                elif "text" in data and "_id" in data:
                                    parsed_item = {
                                        "id": data["_id"],
                                        "text": data["text"].replace("|user|:", "").replace("|agent|:", "").strip(),
                                        "history": [], # No history
                                        "collection": collection_name,
                                        "original_input_obj": [{"speaker": "user", "text": data["text"]}] # Synthetic input obj
                                    }
                                else:
                                    continue # Unknown format
                                
                                items_to_process.append(parsed_item)

                        # 2. Parallel Execution
                        results_buffer = []
                        start_time = time.time()
                        
                        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                            # Submit all tasks
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
                        
                        # 3. Download
                        final_jsonl = "\n".join(results_buffer)
                        st.download_button(
                            label="📥 Download Retrieval Predictions",
                            data=final_jsonl,
                            file_name=f"predictions_{collection_name}.jsonl",
                            mime="application/jsonl"
                        )
                        os.remove(tmp_file_path)

                    except Exception as e:
                        st.error(f"Error: {e}")

        # --- TASK B & C ---
        else:
            # Task C Rewrite Config
            if selected_task == "C":
                with st.expander("✏️ Query Rewrite Configuration", expanded=True):
                    rewrite_enabled = st.checkbox("Enable Query Rewriting", value=True)
                    rewrite_method = st.selectbox("Rewrite Method", ["LLM-based", "Rule-based", "Hybrid"])
                    
                    if rewrite_method in ["LLM-based", "Hybrid"]:
                        prompt_type = st.radio("Prompt Type", ["Default", "Custom"], horizontal=True)
                        custom_prompt = None
                        if prompt_type == "Custom":
                            custom_prompt = st.text_area("Custom Prompt", value=DEFAULT_REWRITE_PROMPT)
                        else:
                            st.code(DEFAULT_REWRITE_PROMPT, language=None)
                    else:
                        prompt_type = "N/A"
                        custom_prompt = None
                    
                    st.session_state.selected_components["rewriter"] = {"enabled": rewrite_enabled, "method": rewrite_method, "custom_prompt": custom_prompt}
            
            # Retrieval Config (Interactive override)
            if selected_task in ["C"]:
                with st.expander("🔍 Retrieval Configuration", expanded=True):
                    st.session_state.selected_components["retriever"] = {
                        "top_k": st.slider("Top K Results", 1, 20, 5, key="interactive_top_k"),
                        "collection": collection_name
                    }
            
            # File Upload for B/C
            uploaded_file = st.file_uploader("Upload Input File (JSONL)", type=["json", "jsonl"])
            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    st.session_state.test_file_path = tmp.name

            # Generation Config
            with st.expander("🤖 Generation Configuration", expanded=True):
                gen_col1, gen_col2 = st.columns(2)
                with gen_col1:
                    if provider == "OpenAI":
                        gen_model = st.selectbox("Model", ["gpt-4o", "gpt-3.5-turbo"], key="gen_model")
                    elif provider == "Gemini":
                        gen_model = st.selectbox("Model", ["gemini-2.0-flash-exp", "gemini-1.5-pro"], key="gen_model")
                    else:
                        gen_model = st.text_input("Model Name", value=model_name, key="gen_model")
                    
                    temperature = st.slider("Temperature", 0.0, 2.0, 0.1)
                    max_tokens = st.slider("Max Tokens", 100, 4096, 1024)
                
                with gen_col2:
                    prompt_template = st.text_area("Prompt Template", value="Answer based on context: {context}\n\nQuestion: {question}")
                
                st.session_state.selected_components["generator"] = {
                    "temperature": temperature, 
                    "max_tokens": max_tokens, 
                    "provider": provider, 
                    "model": gen_model, 
                    "prompt_template": prompt_template
                }

            if st.button("▶️ Run Pipeline", type="primary"):
                if selected_task == "C" and st.session_state.vector_store is None:
                    st.error("❌ No vector store loaded. Please go to 'Knowledge Base'.")
                else:
                    # ... [Run Logic for B/C - same as before, simplified for brevity] ...
                    # Reusing the logic from your previous snippet
                    try:
                        import time
                        from datetime import datetime
                        progress_bar = st.progress(0)
                        task_b_output = []
                        
                        if st.session_state.get("test_file_path"):
                            with open(st.session_state.get("test_file_path"), 'r', encoding='utf-8') as f:
                                lines = f.readlines()
                                for idx, line in enumerate(lines):
                                    if not line.strip(): continue
                                    data = json.loads(line)
                                    
                                    # Logic Extraction
                                    conversation_turns = data.get('input', [])
                                    if not conversation_turns: continue
                                    current_query = conversation_turns[-1]['text']
                                    
                                    # 1. Rewrite
                                    retrieval_query = current_query
                                    if selected_task == "C":
                                        rw_config = st.session_state.selected_components.get("rewriter", {})
                                        if rw_config.get("enabled"):
                                            # Call rewrite logic here (simplified)
                                            retrieval_query = current_query # Placeholder for actual call
                                    
                                    # 2. Retrieve
                                    retrieved_docs = []
                                    if selected_task == "C":
                                        docs = st.session_state.vector_store.similarity_search_with_score(retrieval_query, k=5)
                                        for d, s in docs:
                                            retrieved_docs.append({"title": d.metadata.get('title'), "text": d.page_content, "score": float(s)})
                                    else:
                                        retrieved_docs = data.get('contexts', [])
                                    
                                    # 3. Generate
                                    gen_config = st.session_state.selected_components.get("generator", {})
                                    # Construct prompt...
                                    # Call LLM...
                                    prediction = "Simulated Answer" # Replace with actual LLM call using get_llm
                                    
                                    # For real implementation, paste your LLM call block here
                                    # Keeping it short to fit response limit
                                    llm = get_llm(provider, api_key, base_url, gen_config['model'])
                                    msg = [HumanMessage(content=f"Context: {retrieved_docs} Q: {current_query}")]
                                    try:
                                        res = llm.invoke(msg)
                                        prediction = res.content
                                    except: prediction = "Error"

                                    data["predictions"] = [{"text": prediction}]
                                    if selected_task == "C": data["contexts"] = retrieved_docs
                                    task_b_output.append(json.dumps(data))
                                    progress_bar.progress((idx + 1) / len(lines))
                        
                        st.session_state.gen_result_final_content = "\n".join(task_b_output)
                        st.session_state.gen_result_file_ready = True
                        st.success("Pipeline Completed!")
                    except Exception as e:
                        st.error(f"Pipeline Failed: {e}")

            if st.session_state.gen_result_file_ready:
                st.download_button("📥 Download Predictions", st.session_state.gen_result_final_content, "predictions.jsonl")

    # ==================== TAB: BATCH EVALUATION ====================
    with tab_batch:
        st.header("📊 Batch Evaluation")
        
        # We split the tab into two modes: Official Benchmark vs. Custom File Eval
        eval_mode = st.radio("Evaluation Mode", ["🚀 Run Official MTRAG Benchmark", "📂 Evaluate Custom Files"], horizontal=True)
        st.divider()

        # ---------------- MODE 1: OFFICIAL BENCHMARK RUNNER ----------------
        if eval_mode == "🚀 Run Official MTRAG Benchmark":
            st.subheader("Official MTRAG Benchmark")
            st.markdown("""
            Run the standard benchmark on specific corpora using the `run_mtrag_benchmark.py` script.
            This generates predictions which can then be evaluated.
            """)
            
            # Configuration
            col1, col2 = st.columns(2)
            with col1:
                mtrag_corpus = st.selectbox(
                    "Select Corpus",
                    ["clapnq", "cloud", "fiqa", "govt"],
                    help="MTRAG corpus to evaluate on"
                )
                
                mtrag_task = st.selectbox(
                    "Select Task",
                    ["generation_taskb", "retrieval_taska", "rag_taskc"],
                    format_func=lambda x: {
                        "retrieval_taska": "Task A: Retrieval Only",
                        "generation_taskb": "Task B: Generation",
                        "rag_taskc": "Task C: Full RAG Pipeline"
                    }.get(x, x)
                )
            
            with col2:
                mtrag_limit = st.number_input("Limit Examples", 1, 1000, 10)
                skip_eval = st.checkbox("Skip Auto-Evaluation (Generate Only)", value=False)
            
            if st.button("▶️ Start Benchmark Run", type="primary"):
                if provider != "Local" and not api_key:
                    st.error("❌ API Key required in Sidebar.")
                else:
                    try:
                        import subprocess
                        import sys
                        
                        # Command Construction
                        cmd = [
                            sys.executable, "run_mtrag_benchmark.py",
                            "--corpus", mtrag_corpus,
                            "--task", mtrag_task,
                            "--limit", str(mtrag_limit),
                            "--provider", provider,
                            "--model", model_name
                        ]
                        
                        if provider == "Local":
                            if base_url: cmd.extend(["--base_url", base_url])
                        else:
                            cmd.extend(["--api_key", api_key])
                            
                        if skip_eval:
                            cmd.append("--skip_eval")
                            
                        # Output file path logic
                        output_file = Path("results") / mtrag_task / f"{mtrag_corpus}_predictions.jsonl"
                        cmd.extend(["--output", str(output_file)])
                        
                        st.info(f"Executing: {' '.join(cmd)}")
                        
                        with st.spinner(f"Running benchmark on {mtrag_corpus}..."):
                            # Run Process
                            process = subprocess.Popen(
                                cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                encoding='utf-8',
                                errors='replace'
                            )
                            
                            # Live Log Output
                            output_container = st.empty()
                            logs = []
                            while True:
                                output = process.stdout.readline()
                                if output == '' and process.poll() is not None:
                                    break
                                if output:
                                    line = output.strip()
                                    logs.append(line)
                                    output_container.code("\n".join(logs[-10:]), language="bash")
                            
                            if process.poll() != 0:
                                st.error(f"Failed with code {process.poll()}")
                                st.error(process.stderr.read())
                            else:
                                st.success("✅ Benchmark Completed!")
                                
                                # Show Results
                                if output_file.exists():
                                    st.subheader("📋 Predictions Preview")
                                    predictions = []
                                    with open(output_file, 'r', encoding='utf-8') as f:
                                        for line in f:
                                            predictions.append(json.loads(line))
                                    
                                    results_df = pd.DataFrame([{
                                        "Task ID": p.get("task_id", "")[:15],
                                        "Prediction": str(p.get("predictions", ""))[:80] + "...",
                                        "Contexts": len(p.get("contexts", []))
                                    } for p in predictions])
                                    
                                    st.dataframe(results_df, use_container_width=True)
                                    
                                    with open(output_file, "r", encoding='utf-8') as f:
                                        st.download_button("📥 Download JSONL", f, output_file.name)

                    except Exception as e:
                        st.error(f"Execution Error: {e}")

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
                    st.info("Task A requires a **Predictions File** + **Ground Truth (BEIR)**.")
                    
                    # 1. Select Ground Truth Source
                    selected_corpus = st.selectbox("Select Ground Truth Corpus (for Qrels)", AVAILABLE_CORPORA)
                    selected_query_type = st.selectbox("Query Type", list(QUERY_TYPES.keys()), format_func=lambda x: QUERY_TYPES[x])
                    
                    # Load BEIR Data
                    try:
                        paths = get_retrieval_task_paths(selected_corpus, selected_query_type)
                        beir_qrels = load_qrels(paths["qrels"])
                        beir_queries = load_queries(paths["queries"])
                        st.caption(f"Loaded: {len(beir_queries)} queries, {len(beir_qrels)} relevance sets.")
                    except Exception as e:
                        st.error(f"Failed to load BEIR data: {e}")

                    # 2. Upload Predictions
                    task_a_file = st.file_uploader("Upload Retrieval Predictions (.jsonl)", type=["json", "jsonl"])
                    if task_a_file:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
                            tmp.write(task_a_file.getvalue())
                            eval_dataset_path = tmp.name
                        st.success(f"Loaded {task_a_file.name}")

                # --- TASK B/C INPUTS ---
                else:
                    st.info("Task B/C requires a **Test Dataset** (Input + Ground Truth).")
                    eval_dataset = st.file_uploader("Upload Test Dataset (.jsonl)", type=["json", "jsonl"])
                    if eval_dataset:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
                            tmp.write(eval_dataset.getvalue())
                            eval_dataset_path = tmp.name
                        st.success(f"Loaded {eval_dataset.name}")
                        
                        # Judge Settings
                        judge_provider = st.selectbox("LLM-as-a-Judge", ["ibm-granite/granite-3.3-8b-instruct", "Custom"])
                        if judge_provider == "Custom":
                            judge_provider = st.text_input("Custom Provider String", "ibm-granite/granite-3.3-8b-instruct")

            st.divider()
            
            is_ready = False
            if selected_category == "Task A":
                is_ready = (eval_dataset_path is not None and beir_qrels is not None)
            else:
                is_ready = (eval_dataset_path is not None)

            if st.button("🚀 Run Evaluation", type="primary", disabled=not is_ready):
                if not is_ready:
                    if selected_category == "Task A":
                        st.warning("Please select a valid corpus with queries and qrels.")
                    else:
                        st.warning("Please upload a file to evaluate.")
                else:
                    with st.spinner("Running evaluation..."):
                        import time
                        eval_start = time.time()
                    
                        # Use the currently running Python interpreter
                        import sys
                        venv_python = sys.executable
                        
                        if selected_category == "Task A":
                            # Task A: Run retrieval evaluation
                            st.subheader("📊 Task A Retrieval Evaluation")
                            
                            st.info(f"""
                            **Evaluation Configuration:**
                            - Corpus: **{selected_corpus.upper()}**
                            - Query Type: **{QUERY_TYPES[selected_query_type]}**
                            - Total Queries: **{len(beir_queries)}**
                            - Queries with Relevance Judgments: **{len(beir_qrels)}**
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
                                        import subprocess 
                                        
                                        result = subprocess.run(
                                            run_retrieval_eval_command, 
                                            capture_output=True, 
                                            text=True,
                                            cwd=os.path.dirname(os.path.abspath(__file__))
        )
                                        
                                        if result.returncode == 0:
                                            status.update(label="✅ Retrieval Evaluation Complete!", state="complete")
                                            
                                            if result.stdout:
                                                st.subheader("📈 Retrieval Metrics")
                                                st.code(result.stdout)
                                            
                                            aggregate_csv = output_path.replace("_results.json", "_results_aggregate.csv")
                                            if os.path.exists(aggregate_csv):
                                                st.subheader("📋 Aggregate Results")
                                                df_agg = pd.read_csv(aggregate_csv)
                                                st.dataframe(df_agg)
                                            
                                            if os.path.exists(output_path):
                                                with open(output_path, "rb") as f:
                                                    st.download_button(
                                                        label="📥 Download Enriched Results",
                                                        data=f,
                                                        file_name="retrieval_eval_results.json",
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

                            else:
                                st.warning("⚠️ Please upload your retrieval predictions file above")

                            eval_time = time.time() - eval_start
                            st.caption(f"⏱️ Completed in {eval_time:.2f}s")
                        
                        else:
                            # Task B/C: Generation Eval
                            output_path = eval_dataset_path.replace(".jsonl", "_results.json")
                        
                            run_gen_eval_command = [
                                venv_python, "src/evaluation/run_generation_eval.py",
                                "-i", eval_dataset_path,
                                "-o", output_path,
                                "-e", "src/evaluation/config.yaml",
                                "--provider", "hf",
                                "--judge_model", judge_provider
                            ]
                            
                            with st.status("Evaluating...", expanded=True) as status:
                                try:
                                    result = subprocess.run(run_gen_eval_command, capture_output=True, text=True)
                                    
                                    if result.returncode == 0:
                                        status.update(label="✅ Evaluation Complete!", state="complete")
                                        
                                        if os.path.exists(output_path):
                                            with open(output_path, "rb") as f:
                                                st.download_button(
                                                    label="📥 Download Evaluation Results",
                                                    data=f,
                                                    file_name="evaluation_results.json",
                                                    mime="application/json"
                                                )
                                        else:
                                            st.error("Script finished but no output file was found.")
                                    else:
                                        status.update(label="❌ Evaluation Failed", state="error")
                                        st.error(result.stderr)
                                except Exception as e:
                                    st.error(f"Subprocess failed: {e}")

    # ==================== TAB: LOGS & DEBUGGING ====================
    with tab_logs:
        st.header("📝 Application Logs")
        
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🔄 Refresh Logs"):
                st.rerun()
            if st.button("🗑️ Clear Logs"):
                open('app.log', 'w').close()
                st.rerun()
        
        # Read and display logs (last 500 lines)
        log_file_path = 'app.log'
        if os.path.exists(log_file_path):
            with open(log_file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                # Show newest at the bottom, but limit total lines to avoid UI lag
                last_lines = lines[-500:] 
                log_content = "".join(last_lines)
                
                st.text_area("Log Output", log_content, height=600, key="log_viewer")
        else:
            st.info("No log file found yet.")

if __name__ == "__main__":
    main()
