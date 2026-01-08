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
                        vector_store_instance = st.session_state.vector_store

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
                                            cwd=os.path.dirname(os.path.abspath(__file__)))                                
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
            with open(log_file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
                # Show newest at the bottom, but limit total lines to avoid UI lag
                last_lines = lines[-500:] 
                log_content = "".join(last_lines)
                
                st.text_area("Log Output", log_content, height=600, key="log_viewer")
        else:
            st.info("No log file found yet.")

if __name__ == "__main__":
    main()
