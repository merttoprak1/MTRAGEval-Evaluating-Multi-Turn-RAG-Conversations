import streamlit as st
import pandas as pd
import tempfile
import subprocess
import os
import platform
import logging
import sys
import json
from pathlib import Path
from src.ingestion import load_json_documents, chunk_documents, load_beir_corpus
from src.vector_store import setup_vector_store, get_retriever, add_to_vector_store, delete_from_vector_store
from src.llm_client import get_llm
from src.rag import create_rag_chain
from src.query_rewrite import rewrite_query, DEFAULT_REWRITE_PROMPT
from src.database import init_db, create_session, get_sessions, save_message, load_session_history, delete_session, rename_session
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
    
    # Initialize DB
    init_db()

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

    # --- Sidebar Configuration ---
    st.sidebar.header("Configuration")
    
    # Session Management
    st.sidebar.subheader("Chat Sessions")
    
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None

    sessions = get_sessions()
    session_options = {s['id']: f"{s['name']} ({s['created_at'][:16]})" for s in sessions}
    
    # Session Management Callbacks
    def create_session_click():
        new_id = create_session()
        if new_id:
            st.session_state.current_session_id = new_id
            st.session_state.session_selector = new_id

    def delete_session_click(sess_id):
        delete_session(sess_id)
        st.session_state.current_session_id = None
        st.session_state.session_selector = "new_session"

    # Add "New Session" option
    # Use a key to control the widget state
    if "session_selector" not in st.session_state:
        st.session_state.session_selector = "new_session"

    selected_session_id = st.sidebar.selectbox(
        "Select Session", 
        options=["new_session"] + list(session_options.keys()),
        format_func=lambda x: "➕ New Session" if x == "new_session" else session_options.get(x, "Unknown"),
        key="session_selector"
    )
    
    if selected_session_id == "new_session":
        st.sidebar.button("Create Session", on_click=create_session_click)
    else:
        st.session_state.current_session_id = selected_session_id
        
        # Rename Session
        # Get current name safely
        current_name = session_options.get(selected_session_id, "Unknown").split(" (")[0]
        new_session_name = st.sidebar.text_input("Rename Session", value=current_name)
        if st.sidebar.button("Update Name"):
            rename_session(selected_session_id, new_session_name)
            st.rerun()

        st.sidebar.button("Delete Session", on_click=delete_session_click, args=(selected_session_id,))

    st.sidebar.divider()
    
    # Provider Selection
    st.sidebar.subheader("LLM Configuration")
    # Default to Local (index 1)
    # Default to Gemini (index 1) as requested
    provider = st.sidebar.selectbox("Select LLM Provider", ["OpenAI", "Gemini", "Local"], index=1)
    
    api_key = None
    base_url = None
    model_name = "gpt-3.5-turbo"

    if provider == "OpenAI":
        api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        model_name = st.sidebar.text_input("Model Name", value="gpt-3.5-turbo")
    elif provider == "Gemini":
        api_key = st.sidebar.text_input("Google API Key", type="password")
        model_name = st.sidebar.text_input("Model Name", value="gemini-flash-latest")
    else:
        base_url = st.sidebar.text_input("Local LLM Base URL", value="http://localhost:1234/v1")
        model_name = st.sidebar.text_input("Model Name", value="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF")
        st.sidebar.info("Ensure your local server is running and compatible with OpenAI API format (e.g., Ollama, LM Studio).")

    # Embedding Configuration
    st.sidebar.subheader("Embedding Configuration")
    # Default to Gemini (index 1)
    embedding_provider = st.sidebar.selectbox("Select Embedding Provider", ["OpenAI", "Gemini", "Local (Ollama)"], index=1)
    embedding_config = {}
    
    # Model lists for each provider
    OPENAI_EMBEDDING_MODELS = {
        "text-embedding-3-small": {"dim": 1536, "description": "Fastest, lowest cost"},
        "text-embedding-3-large": {"dim": 3072, "description": "Best quality"},
        "text-embedding-ada-002": {"dim": 1536, "description": "Legacy model"},
    }
    
    GEMINI_EMBEDDING_MODELS = {
        "models/text-embedding-004": {"dim": 768, "description": "Latest, recommended"},
        "models/embedding-001": {"dim": 768, "description": "Legacy model"},
    }
    
    LOCAL_EMBEDDING_MODELS = {
        "nomic-embed-text": {"dim": 768, "description": "Good general purpose"},
        "mxbai-embed-large": {"dim": 1024, "description": "High quality"},
        "all-minilm": {"dim": 384, "description": "Fast, lightweight"},
        "shaw/dmeta-embedding-zh-small-q4": {"dim": 512, "description": "Multilingual"},
        "custom": {"dim": None, "description": "Enter custom model name"},
    }

    if embedding_provider == "OpenAI":
        # API Key
        if not api_key:
             embedding_api_key = st.sidebar.text_input("OpenAI Embedding API Key", type="password")
        else:
             embedding_api_key = api_key
        
        # Model selection
        selected_model = st.sidebar.selectbox(
            "Embedding Model",
            list(OPENAI_EMBEDDING_MODELS.keys()),
            format_func=lambda x: f"{x} ({OPENAI_EMBEDDING_MODELS[x]['description']})"
        )
        model_info = OPENAI_EMBEDDING_MODELS[selected_model]
        
        # Show dimension (read-only info)
        st.sidebar.caption(f"📐 Dimension: {model_info['dim']}")
        
        # Batch size (optional)
        batch_size = st.sidebar.number_input("Batch Size", min_value=1, max_value=2048, value=100, help="Documents per batch for embedding")
        
        embedding_config = {
            "provider": "OpenAI",
            "api_key": embedding_api_key,
            "model_name": selected_model,
            "dimension": model_info['dim'],
            "batch_size": batch_size
        }
        
    elif embedding_provider == "Gemini":
        # API Key
        if provider == "Gemini" and api_key:
            embedding_api_key = api_key
        else:
            embedding_api_key = st.sidebar.text_input("Gemini Embedding API Key", type="password")
        
        # Model selection
        selected_model = st.sidebar.selectbox(
            "Embedding Model",
            list(GEMINI_EMBEDDING_MODELS.keys()),
            format_func=lambda x: f"{x.split('/')[-1]} ({GEMINI_EMBEDDING_MODELS[x]['description']})"
        )
        model_info = GEMINI_EMBEDDING_MODELS[selected_model]
        
        # Show dimension
        st.sidebar.caption(f"📐 Dimension: {model_info['dim']}")
        
        # Batch size - Gemini has strict rate limits
        batch_size = st.sidebar.number_input("Batch Size", min_value=1, max_value=100, value=1, help="Documents per batch (keep low for Gemini rate limits)")
        
        embedding_config = {
             "provider": "Gemini",
             "api_key": embedding_api_key,
             "model_name": selected_model,
             "dimension": model_info['dim'],
             "batch_size": batch_size
        }
        
    else:
        # Local Ollama
        embed_base_url = st.sidebar.text_input("Embedding Base URL", value="http://localhost:11434")
        
        # Model selection
        selected_model = st.sidebar.selectbox(
            "Embedding Model",
            list(LOCAL_EMBEDDING_MODELS.keys()),
            format_func=lambda x: f"{x} ({LOCAL_EMBEDDING_MODELS[x]['description']})"
        )
        
        # Custom model input
        if selected_model == "custom":
            custom_model_name = st.sidebar.text_input("Custom Model Name", value="")
            embed_model = custom_model_name
            custom_dim = st.sidebar.number_input("Model Dimension", min_value=64, max_value=4096, value=768, help="Embedding dimension of your custom model")
            model_dim = custom_dim
        else:
            embed_model = selected_model
            model_info = LOCAL_EMBEDDING_MODELS[selected_model]
            model_dim = model_info['dim']
            st.sidebar.caption(f"📐 Dimension: {model_dim}")
        
        # Batch size
        batch_size = st.sidebar.number_input("Batch Size", min_value=1, max_value=500, value=50, help="Documents per batch for local embedding")
        
        embedding_config = {
            "provider": "Local",
            "base_url": embed_base_url,
            "model_name": embed_model,
            "dimension": model_dim,
            "batch_size": batch_size
        }

    # Vector DB Configuration
    st.sidebar.subheader("Vector Database")
    vector_db_type = st.sidebar.selectbox(
        "Select Vector DB",
        ["FAISS", "Chroma", "Pinecone"],
        index=0,
        help="Choose the vector database backend for storing embeddings"
    )
    
    # Common config
    retrieval_top_k = st.sidebar.slider("Top-K Results", min_value=1, max_value=20, value=5, help="Number of documents to retrieve")
    
    # DB-specific configuration
    db_config = {"top_k": retrieval_top_k}
    
    if vector_db_type == "FAISS":
        st.sidebar.caption("💾 FAISS - Local file-based vector store")
        faiss_index_name = st.sidebar.text_input("Index Name", value="default", help="Name for the FAISS index file")
        db_config["index_name"] = faiss_index_name
        
    elif vector_db_type == "Chroma":
        st.sidebar.caption("🎨 Chroma - Local persistent vector store")
        chroma_index_name = st.sidebar.text_input("Collection Name", value="default", key="chroma_collection", help="Chroma collection name")
        chroma_namespace = st.sidebar.text_input("Namespace", value="", key="chroma_namespace", help="Optional namespace for organization")
        db_config["index_name"] = chroma_index_name
        db_config["namespace"] = chroma_namespace if chroma_namespace else None
        
    elif vector_db_type == "Pinecone":
        st.sidebar.caption("🌲 Pinecone - Cloud vector database")
        pinecone_api_key = st.sidebar.text_input("Pinecone API Key", type="password")
        pinecone_index = st.sidebar.text_input("Index Name", value="default-index", key="pinecone_index", help="Pinecone index name")
        pinecone_namespace = st.sidebar.text_input("Namespace", value="", key="pinecone_namespace", help="Optional namespace within the index")
        
        if pinecone_api_key:
            db_config["api_key"] = pinecone_api_key
            db_config["index_name"] = pinecone_index
            db_config["namespace"] = pinecone_namespace if pinecone_namespace else None
        else:
            st.sidebar.warning("⚠️ Pinecone API key required")

    # Collection Configuration (for internal naming)
    st.sidebar.subheader("Collection Management")
    collection_name = st.sidebar.text_input("Internal Collection Name", value="default_collection", help="Internal name for organizing your data")
    
    # Initialize Vector Store with selected collection (load existing data)
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "current_db_type" not in st.session_state:
        st.session_state.current_db_type = None
    if "current_embedding_model" not in st.session_state:
        st.session_state.current_embedding_model = None
    
    # Get current embedding model identifier
    current_embed_model = embedding_config.get("model_name", "default")
    
    # Reload if collection, db_type, or embedding model changed
    should_reload = (
        st.session_state.vector_store is None or 
        st.session_state.get("current_collection") != collection_name or
        st.session_state.get("current_db_type") != vector_db_type or
        st.session_state.get("current_embedding_model") != current_embed_model
    )
    
    if should_reload:
         # Try to load existing collection
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
             logger.info(f"Vector store loaded with embedding model: {current_embed_model}")
         except Exception as e:
             logger.warning(f"Could not load collection {collection_name}: {e}")

    # File Upload
    st.sidebar.subheader("Data Ingestion")
    uploaded_file = st.sidebar.file_uploader("Upload JSON/JSONL File", type=["json", "jsonl"])

    if uploaded_file:
        if st.sidebar.button("Process File"):
            logger.info(f"Processing uploaded file: {uploaded_file.name}")
            with st.spinner("Processing..."):
                try:
                    # Save uploaded file to temp file
                    suffix = ".jsonl" if uploaded_file.name.endswith(".jsonl") else ".json"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                        
                    
                    # Ingest and Chunk
                    documents = load_json_documents(tmp_file_path)
                    st.sidebar.write(f"Loaded {len(documents)} documents.")
                    
                    if not documents:
                        logger.warning("No documents found in uploaded file")
                        st.error("No documents found. Please check the file format.")
                    else:
                        chunks = chunk_documents(documents)
                        st.sidebar.write(f"Created {len(chunks)} chunks.")
                        
                        # Setup Vector Store (Ingest)
                        if embedding_provider == "OpenAI" and not embedding_config.get("api_key"):
                            st.error("Please provide an OpenAI API Key for embedding generation.")
                            return

                        if api_key:
                            os.environ["OPENAI_API_KEY"] = api_key
                        
                        # This will add to the existing collection or create new one
                        st.session_state.vector_store = setup_vector_store(
                            chunks, 
                            embedding_config=embedding_config, 
                            collection_name=collection_name,
                            db_type=vector_db_type,
                            db_config=db_config
                        )
                        st.session_state.current_collection = collection_name
                        st.session_state.current_db_type = vector_db_type
                        st.sidebar.success(f"Ingested into {vector_db_type}: {collection_name}")
                        logger.info("File processing completed successfully")
                    
                    # Cleanup temp file
                    os.remove(tmp_file_path)
                    
                except Exception as e:
                    logger.error(f"Error processing file: {e}", exc_info=True)
                    st.error(f"Error processing file: {e}")

    # --- Main Content ---
    
    # Tabs: 2 new + 3 existing
    tab_rag, tab_eval, tab_chat, tab_manage, tab_db = st.tabs([
        "🎯 RAG Playground", 
        "📊 Evaluation Playground",
        "💬 Chat", 
        "🛠️ Manage Collection", 
        "🔍 Database Inspector"
    ])

    # ==================== TAB: Evaluation Playground (MTRAG Benchmark merged here) ====================
    with tab_eval:
        st.header("📊 Evaluation Playground")
        st.markdown("""
        Run official MTRAG benchmark evaluation on your RAG system.
        This uses the multi-turn conversation dataset from IBM Research.
        """)
        
        # MTRAG Configuration
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
                    "generation_taskb": "Task B: Generation (with provided contexts)",
                    "rag_taskc": "Task C: Full RAG Pipeline"
                }.get(x, x),
                help="MTRAG task type"
            )
        
        with col2:
            mtrag_limit = st.number_input(
                "Number of Tasks",
                min_value=1,
                max_value=100,
                value=10,
                help="Limit number of tasks to run (for faster testing)"
            )
            
            skip_eval = st.checkbox(
                "Skip Evaluation",
                value=False,
                help="Skip MTRAG evaluation after generating predictions"
            )
        
        st.divider()
        
        # Run Benchmark Button
        # Run Benchmark Button
        if st.button("▶️ Run MTRAG Benchmark", type="primary", key="run_mtrag_btn"):
            if provider != "Local" and not api_key:
                st.error("❌ Please provide API key in the sidebar for non-local providers")
            else:
                try:
                    import subprocess
                    import sys
                    
                    # Prepare command
                    cmd = [
                        sys.executable, "run_mtrag_benchmark.py",
                        "--corpus", mtrag_corpus,
                        "--task", mtrag_task,
                        "--limit", str(mtrag_limit),
                        "--provider", provider,
                        "--model", model_name
                    ]
                    
                    if provider == "Local":
                        if base_url:
                            cmd.extend(["--base_url", base_url])
                    else:
                        cmd.extend(["--api_key", api_key])
                        
                    if skip_eval:
                        cmd.append("--skip_eval")
                        
                    # Output file path (must match what run_mtrag_benchmark.py uses)
                    # It creates files like: results/{task}/{corpus}_predictions.jsonl
                    # We can use the --output arg to be sure
                    output_file = Path("results") / mtrag_task / f"{mtrag_corpus}_predictions.jsonl"
                    cmd.extend(["--output", str(output_file)])
                    
                    st.info(f"Executing: {' '.join(cmd)}")
                    
                    with st.spinner(f"Running MTRAG benchmark on {mtrag_corpus}..."):
                        # Run command
                        process = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            encoding='utf-8',
                            errors='replace' # Handle encoding errors gracefully
                        )
                        
                        # Real-time output container
                        output_container = st.empty()
                        logs = []
                        
                        # Read logs in real-time
                        while True:
                            output = process.stdout.readline()
                            if output == '' and process.poll() is not None:
                                break
                            if output:
                                line = output.strip()
                                logs.append(line)
                                # Show last few lines of log
                                output_container.code("\n".join(logs[-10:]), language="bash")
                        
                        # Check exit code
                        return_code = process.poll()
                        if return_code != 0:
                            stderr = process.stderr.read()
                            st.error(f"Benchmark failed with code {return_code}")
                            st.error(stderr)
                        else:
                            st.success("✅ Benchmark completed successfully!")
                            
                            # Load and display results
                            if output_file.exists():
                                st.subheader("📋 Predictions")
                                predictions = []
                                with open(output_file, 'r', encoding='utf-8') as f:
                                    for line in f:
                                        predictions.append(json.loads(line))
                                
                                # Display table
                                results_df = pd.DataFrame([
                                    {
                                        "Task ID": p.get("task_id", "")[:20] + "...",
                                        "Conversation ID": p.get("conversation_id", ""),
                                        "Prediction": str(p.get("predictions", ""))[:100] + "...",
                                        "Contexts": len(p.get("contexts", []))
                                    }
                                    for p in predictions
                                ])
                                st.dataframe(results_df, use_container_width=True)
                                
                                # Download button
                                with open(output_file, "r", encoding='utf-8') as f:
                                    jsonl_content = f.read()
                                
                                st.download_button(
                                    "📥 Download Predictions (JSONL)",
                                    data=jsonl_content,
                                    file_name=output_file.name,
                                    mime="application/jsonl"
                                )
                            else:
                                st.warning(f"Output file not found at {output_file}")
                                
                except Exception as e:
                    st.error(f"❌ Execution error: {e}")
                    logger.error(f"Benchmark execution error: {e}", exc_info=True)

    # ==================== TAB 1: RAG Playground ====================
    with tab_rag:
        st.header("🎯 RAG Playground")
        
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
        
        # Conditional UI based on task selection
        if selected_task is None:
            st.info("👆 Please select a task to begin.")
        else:
            # Task C: Rewrite component
            if selected_task == "C":
                with st.expander("✏️ Query Rewrite Configuration", expanded=True):
                    rewrite_enabled = st.checkbox("Enable Query Rewriting", value=True)
                    rewrite_method = st.selectbox("Rewrite Method", ["LLM-based", "Rule-based", "Hybrid"])
                    
                    # Prompt selection (only relevant for LLM-based and Hybrid)
                    if rewrite_method in ["LLM-based", "Hybrid"]:
                        prompt_type = st.radio(
                            "Prompt Type",
                            ["Default", "Custom"],
                            horizontal=True,
                            help="Choose between the default rewrite prompt or provide your own custom prompt."
                        )
                        
                        custom_prompt = None
                        if prompt_type == "Default":
                            with st.expander("📄 View Default Prompt"):
                                st.code(DEFAULT_REWRITE_PROMPT, language=None)
                        else:
                            custom_prompt = st.text_area(
                                "Custom Rewrite Prompt",
                                value=DEFAULT_REWRITE_PROMPT,
                                height=200,
                                help="Enter your custom system prompt for query rewriting. The query will be provided as {query}."
                            )
                    else:
                        prompt_type = "N/A (Rule-based)"
                        custom_prompt = None
                    
                    st.session_state.selected_components["rewriter"] = {
                        "enabled": rewrite_enabled,
                        "method": rewrite_method,
                        "prompt_type": prompt_type,
                        "custom_prompt": custom_prompt
                    }
            
            # Task A, B, C: Retrieval component
            if selected_task in ["A", "C"]:
                with st.expander("🔍 Retrieval Configuration", expanded=True):
                    st.session_state.selected_components["retriever"] = {
                        "top_k": st.slider("Top K Results", 1, 20, 5),
                        "search_type": st.selectbox("Search Type", ["similarity", "mmr", "similarity_score_threshold"]),
                        "collection": collection_name
                    }
            
            # Task B, C: Generation component
            if selected_task in ["B", "C"]:
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
            
            # # Test Query Input
            # test_query = st.text_input(
            #     "🔤 Test Query", 
            #     placeholder="Enter a query to test the pipeline...",
            #     key="test_query_input"
            # )
            
            # Run Button
            if st.button("▶️ Run Pipeline", type="primary"):
                # if not test_query:
                #     st.warning("⚠️ Please enter a test query.")
                if st.session_state.vector_store is not None:
                    st.error("❌ No vector store loaded. Please upload and process documents first.")
                # elif selected_task in ["B", "C"] and not api_key:
                #     st.error("❌ API key required for generation. Please provide it in the sidebar.")
                else:
                    # Create progress container
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
                        
                        # ==================== TASK A, C: RETRIEVAL ====================
                        if selected_task in ["A", "C"]:
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

                        # ==================== TASK B, C: GENERATION ====================
                        if selected_task in ["B", "C"]:
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
                                    
                                    # # Build context from retrieved docs
                                    # context_parts = []
                                    # for i, doc in enumerate(retrieved_docs):
                                    #     source = doc.metadata.get('source', doc.metadata.get('title', f'Document {i+1}'))
                                    #     context_parts.append(f"[{i+1}] {source}:\n{doc.page_content}")
                                    
                                    # context = "\n\n".join(context_parts)
                                    
                                    # Format the prompt
                                    # formatted_prompt = prompt_template.format(
                                    #     context=context,
                                    #     question=retrieval_query
                                    # )
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

                                            # # Display generation stats
                                            # gen_stat_col1, gen_stat_col2, gen_stat_col3 = st.columns(3)
                                            # with gen_stat_col1:
                                            #     st.metric("🤖 Model", gen_model[:20] + "..." if len(gen_model) > 20 else gen_model)
                                            # with gen_stat_col2:
                                            #     st.metric("⏱️ Gen Time", f"{round(gen_time * 1000, 2)} ms")
                                            # with gen_stat_col3:
                                            #     st.metric("📝 Answer Length", f"{len(prediction)} chars")
                                            
                                            # # Display the generated answer
                                            # st.markdown("---")
                                            # st.markdown("### 💬 Generated Answer")
                                            # st.markdown(prediction)
                                            
                                            # # Debug Panel for Generation
                                            # with st.expander("🐛 Debug: Generation Details", expanded=False):
                                            #     st.markdown("#### Configuration")
                                            #     debug_gen_col1, debug_gen_col2 = st.columns(2)
                                            #     with debug_gen_col1:
                                            #         st.write(f"• **Provider:** {provider}")
                                            #         st.write(f"• **Model:** {gen_model}")
                                            #         st.write(f"• **Temperature:** {gen_temperature}")
                                            #     with debug_gen_col2:
                                            #         st.write(f"• **Max Tokens:** {gen_max_tokens}")
                                            #         st.write(f"• **Prompt Template:** {prompt_template_name}")
                                            #         st.write(f"• **Context Docs:** {len(retrieved_docs)}")
                                                
                                            #     st.markdown("---")
                                            #     st.markdown("#### 📄 Context Used")
                                            #     st.text_area(
                                            #         "Combined Context",
                                            #         full_context_str,
                                            #         height=200,
                                            #         disabled=True,
                                            #         key="debug_context"
                                            #     )
                                                
                                            #     st.markdown("---")
                                            #     st.markdown("#### 📝 Full Prompt Sent to LLM")
                                            #     st.code(full_context_str, language=None)
                                                
                                            #     st.markdown("---")
                                            #     st.markdown("#### 📊 Token Estimates")
                                            #     # Rough estimate: 1 token ≈ 4 chars
                                            #     prompt_tokens_est = len(full_context_str) // 4
                                            #     answer_tokens_est = len(prediction) // 4
                                            #     st.write(f"• **Prompt Tokens (est):** ~{prompt_tokens_est}")
                                            #     st.write(f"• **Answer Tokens (est):** ~{answer_tokens_est}")
                                            #     st.write(f"• **Total Tokens (est):** ~{prompt_tokens_est + answer_tokens_est}")
                                                
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
                        
                        # Export / Copy Buttons
                        # st.markdown("---")
                        # export_col1, export_col2, export_col3, export_col4 = st.columns(4)
                        
                        # with export_col1:
                        #     # Export full results as JSON
                        #     export_data = {
                        #         "task": run_result.get("task"),
                        #         "query": run_result.get("query"),
                        #         "config_snapshot": run_result.get("config_snapshot", {}),
                        #         "total_time_ms": run_result.get("total_time_ms"),
                        #         "retrieval": {
                        #             k: v for k, v in run_result.get("retrieval", {}).items() 
                        #             if k != "results"
                        #         },
                        #         "generation": run_result.get("generation", {})
                        #     }
                        #     st.download_button(
                        #         "📥 Export Results",
                        #         data=json.dumps(export_data, indent=2, default=str),
                        #         file_name=f"rag_run_{selected_task}_{int(time.time())}.json",
                        #         mime="application/json",
                        #         help="Download full run results as JSON"
                        #     )
                        
                        # with export_col2:
                        #     # Export just the answer
                        #     answer_text = run_result.get("generation", {}).get("answer", "")
                        #     if answer_text:
                        #         st.download_button(
                        #             "📝 Export Answer",
                        #             data=answer_text,
                        #             file_name=f"answer_{int(time.time())}.txt",
                        #             mime="text/plain",
                        #             help="Download generated answer as text"
                        #         )
                        #     else:
                        #         st.button("📝 Export Answer", disabled=True, help="No answer to export")
                        
                        # with export_col3:
                        #     # Export config snapshot
                        #     config_data = run_result.get("config_snapshot", {})
                        #     st.download_button(
                        #         "⚙️ Export Config",
                        #         data=json.dumps(config_data, indent=2, default=str),
                        #         file_name=f"config_{int(time.time())}.json",
                        #         mime="application/json",
                        #         help="Download configuration snapshot"
                        #     )
                        
                        # with export_col4:
                        #     # Copy answer to clipboard (using text area workaround)
                        #     if run_result.get("generation", {}).get("answer"):
                        #         st.text_input(
                        #             "📋 Copy Answer",
                        #             value=run_result["generation"]["answer"][:100] + "...",
                        #             key="copy_answer_field",
                        #             help="Select and copy this text",
                        #             disabled=True
                        #         )
                        
                        # Debug Panel with Config Snapshot
                        # with st.expander("🐛 Debug Panel - Full Run Details", expanded=False):
                        #     # Structured view
                        #     st.markdown("### 📦 Complete Run Result")
                            
                        #     tab_config, tab_rewrite, tab_retrieval, tab_generation, tab_json = st.tabs([
                        #         "⚙️ Config", "✏️ Rewrite", "🔍 Retrieval", "🤖 Generation", "📋 Raw JSON"
                        #     ])
                            
                        #     with tab_config:
                        #         st.markdown("#### 📸 Configuration Snapshot")
                        #         config_data = run_result.get("config_snapshot", {})
                        #         if config_data:
                        #             config_col1, config_col2 = st.columns(2)
                        #             with config_col1:
                        #                 st.write(f"**Timestamp:** {config_data.get('timestamp', 'N/A')}")
                        #                 st.write(f"**Task:** {config_data.get('task', 'N/A')}")
                        #                 st.write(f"**LLM Provider:** {config_data.get('llm_provider', 'N/A')}")
                        #                 st.write(f"**LLM Model:** {config_data.get('llm_model', 'N/A')}")
                        #             with config_col2:
                        #                 st.write(f"**Embedding Provider:** {config_data.get('embedding_provider', 'N/A')}")
                        #                 st.write(f"**Embedding Model:** {config_data.get('embedding_model', 'N/A')}")
                        #                 st.write(f"**Vector DB:** {config_data.get('vector_db', 'N/A')}")
                        #                 st.write(f"**Top-K:** {config_data.get('top_k', 'N/A')}")
                        #             st.markdown("**Full Config JSON:**")
                        #             st.json(config_data)
                        #         else:
                        #             st.info("No config snapshot available")
                            
                        #     with tab_rewrite:
                        #         rewrite_data = run_result.get("rewrite_result", {})
                        #         if rewrite_data:
                        #             st.json(rewrite_data)
                        #         else:
                        #             st.info("No rewrite data (Task A or B)")
                            
                        #     with tab_retrieval:
                        #         retrieval_data = run_result.get("retrieval", {})
                        #         if retrieval_data:
                        #             # Don't show full content in JSON view
                        #             display_data = {k: v for k, v in retrieval_data.items() if k != "results"}
                        #             display_data["results_count"] = len(retrieval_data.get("results", []))
                        #             st.json(display_data)
                        #         else:
                        #             st.info("No retrieval data")
                            
                        #     with tab_generation:
                        #         gen_data = run_result.get("generation", {})
                        #         if gen_data:
                        #             st.json(gen_data)
                        #         else:
                        #             st.info("No generation data (Task A)")
                            
                        #     with tab_json:
                        #         # Full JSON but without large content fields
                        #         display_result = {
                        #             "task": run_result.get("task"),
                        #             "query": run_result.get("query"),
                        #             "status": run_result.get("status"),
                        #             "total_time_ms": run_result.get("total_time_ms"),
                        #             "errors": run_result.get("errors", []),
                        #             "has_config": "config_snapshot" in run_result,
                        #             "has_rewrite": "rewrite_result" in run_result,
                        #             "has_retrieval": "retrieval" in run_result,
                        #             "has_generation": "generation" in run_result
                        #         }
                        #         st.json(display_result)
                    
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
    # ==================== TAB 2: Evaluation Playground ====================
    with tab_eval:
        st.header("📊 Evaluation Playground")
        st.markdown("Evaluate your RAG pipeline performance with various metrics and benchmarks.")
        
        # # Evaluation Scripts Definition
        # EVALUATION_SCRIPTS = {
        #     "Task A - Retrieval": {
        #         "Retrieval Accuracy (Hit Rate)": {
        #             "description": "Measures if relevant documents are retrieved in top-k results",
        #             "metrics": ["Hit Rate@K", "MRR", "NDCG"],
        #             "requires": ["ground_truth_docs"]
        #         },
        #         "Retrieval Precision": {
        #             "description": "Ratio of relevant documents among retrieved documents",
        #             "metrics": ["Precision@K", "Recall@K", "F1@K"],
        #             "requires": ["ground_truth_docs"]
        #         },
        #         "Semantic Similarity": {
        #             "description": "Cosine similarity between query and retrieved documents",
        #             "metrics": ["Avg Similarity", "Min Similarity", "Max Similarity"],
        #             "requires": []
        #         }
        #     },
        #     "Task B - Generation": {
        #         "Answer Correctness": {
        #             "description": "Measures correctness of generated answer against ground truth",
        #             "metrics": ["Exact Match", "F1 Score", "BLEU"],
        #             "requires": ["ground_truth_answers"]
        #         },
        #         "Faithfulness": {
        #             "description": "Measures if the answer is grounded in retrieved context",
        #             "metrics": ["Faithfulness Score", "Hallucination Rate"],
        #             "requires": ["llm_as_judge"]
        #         },
        #         "Answer Relevance": {
        #             "description": "Measures how relevant the answer is to the question",
        #             "metrics": ["Relevance Score"],
        #             "requires": ["llm_as_judge"]
        #         },
        #         "Context Relevance": {
        #             "description": "Measures relevance of retrieved context to the question",
        #             "metrics": ["Context Precision", "Context Recall"],
        #             "requires": ["ground_truth_docs"]
        #         }
        #     },
        #     "Task C - Full RAG (Rewrite + Retrieval + Generation)": {
        #         "End-to-End Accuracy": {
        #             "description": "Full pipeline accuracy from query to final answer",
        #             "metrics": ["E2E Accuracy", "E2E F1"],
        #             "requires": ["ground_truth_answers"]
        #         },
        #         "Query Rewrite Quality": {
        #             "description": "Evaluates if rewritten query improves retrieval",
        #             "metrics": ["Rewrite Hit Rate Improvement", "Semantic Preservation"],
        #             "requires": ["llm_as_judge"]
        #         },
        #         "RAGAS Score": {
        #             "description": "Comprehensive RAG evaluation using RAGAS framework",
        #             "metrics": ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"],
        #             "requires": ["llm_as_judge", "ground_truth_answers"]
        #         },
        #         "Latency Analysis": {
        #             "description": "Performance timing for each pipeline stage",
        #             "metrics": ["Rewrite Time", "Retrieval Time", "Generation Time", "Total Time"],
        #             "requires": []
        #         }
        #     }
        # }
        
        # Task Selector
        eval_col1, eval_col2 = st.columns([1, 2])
        
        with eval_col1:
            st.subheader("🎯 Select Task to Evaluate")
            eval_task_options = {
                "A": "Task A: Retrieval Only",
                "B": "Task B: Generation",
                "B": "Task B: Generation",
                "C": "Task C: Full RAG Pipeline"
            }
            
            eval_selected_task = st.radio(
                "Task Type",
                options=list(eval_task_options.keys()),
                format_func=lambda x: eval_task_options[x],
                key="eval_task_selector"
            )
            
            # Map to script category
            task_to_category = {
                "A": "Task A - Retrieval",
                "B": "Task B - Generation",
                "C": "Task C - Full RAG (Rewrite + Retrieval + Generation)"
            }
            selected_category = task_to_category[eval_selected_task]
        
        # with eval_col2:
        #     st.subheader("📋 Select Evaluation Scripts")
            
        #     available_scripts = EVALUATION_SCRIPTS.get(selected_category, {})
            
        #     selected_scripts = []
        #     for script_name, script_info in available_scripts.items():
        #         col_check, col_info = st.columns([1, 4])
        #         with col_check:
        #             if st.checkbox(script_name, key=f"eval_script_{script_name}"):
        #                 selected_scripts.append(script_name)
        #         with col_info:
        #             st.caption(script_info["description"])
        #             metrics_str = ", ".join(script_info["metrics"])
        #             st.caption(f"📊 Metrics: {metrics_str}")
        #             if script_info["requires"]:
        #                 req_str = ", ".join(script_info["requires"])
        #                 st.caption(f"⚠️ Requires: {req_str}")
        
        # Input Configuration
        st.subheader("📝 Evaluation Input")
        
        eval_dataset = None
        eval_dataset_path = None
        beir_qrels = None
        beir_queries = None
        
        # Task A: BEIR Format Configuration
        if eval_selected_task == "A":
            st.info("📋 **Task A uses BEIR format** with pre-loaded queries and relevance judgments (qrels)")
            st.caption("ℹ️ *Corpus seçimi şuanlık sadece önizleme amaçlıdır. Evaluation sırasında predictions dosyasındaki `Collection` alanı kullanılır.*")
            
            beir_col1, beir_col2 = st.columns(2)
            
            with beir_col1:
                st.markdown("**Select Corpus**")
                selected_corpus = st.selectbox(
                    "Corpus",
                    options=AVAILABLE_CORPORA,
                    format_func=lambda x: {
                        "clapnq": "ClapNQ (Wikipedia)",
                        "cloud": "Cloud (Technical Docs)",
                        "fiqa": "FiQA (Finance)",
                        "govt": "Govt (Government)"
                    }.get(x, x),
                    key="beir_corpus_selector",
                    help="Select the document corpus for retrieval evaluation"
                )
                
            with beir_col2:
                st.markdown("**Select Query Type**")
                selected_query_type = st.selectbox(
                    "Query Type",
                    options=list(QUERY_TYPES.keys()),
                    format_func=lambda x: QUERY_TYPES[x],
                    key="beir_query_type_selector",
                    help="Choose query format: full conversation, last turn only, or rewritten queries"
                )
            
            # Load and display BEIR data
            try:
                paths = get_retrieval_task_paths(selected_corpus, selected_query_type)
                beir_qrels = load_qrels(paths["qrels"])
                beir_queries = load_queries(paths["queries"])
                
                # Display stats
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                with stats_col1:
                    st.metric("📄 Queries", len(beir_queries))
                with stats_col2:
                    st.metric("📋 Qrels (Query-Doc Pairs)", sum(len(v) for v in beir_qrels.values()))
                with stats_col3:
                    st.metric("🎯 Queries with Relevance", len(beir_qrels))
                
                # Preview sample queries
                with st.expander("👁️ Preview Sample Queries", expanded=False):
                    sample_queries = list(beir_queries.items())[:5]
                    for qid, qtext in sample_queries:
                        st.markdown(f"**{qid}**")
                        st.caption(qtext[:200] + "..." if len(qtext) > 200 else qtext)
                        st.divider()
                
                # File uploader for Task A predictions
                st.markdown("---")
                st.markdown("**📤 Upload Your Retrieval Predictions**")
                task_a_predictions = st.file_uploader(
                    "Upload JSONL file with your retrieval results",
                    type=["jsonl", "json"],
                    key="task_a_predictions_upload",
                    help="File should contain: task_id, Collection, contexts (with document_id and score)"
                )
                task_a_predictions_path = None
                if task_a_predictions is not None:
                    suffix = f".{task_a_predictions.name.split('.')[-1]}"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                        temp_file.write(task_a_predictions.getvalue())
                        task_a_predictions_path = temp_file.name
                    st.success(f"✅ Loaded: {task_a_predictions.name}")
                        
            except Exception as e:
                st.error(f"❌ Failed to load BEIR data: {e}")
                beir_qrels = None
                beir_queries = None
                task_a_predictions_path = None
        
        # Task B/C: File Upload
        # Task B/C: File Upload
        else:
            dataset_col1, dataset_col2 = st.columns(2)
            
            with dataset_col1:
                st.markdown("**Upload Test Dataset**")
                eval_dataset = st.file_uploader(
                    "Upload JSON/JSONL with test queries and ground truth",
                    type=["json", "jsonl"],
                    key="eval_dataset_upload",
                    help="File should contain: query, ground_truth_answer (optional), ground_truth_docs (optional)"
                )
                if eval_dataset is not None:
                    suffix = f".{eval_dataset.name.split('.')[-1]}"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                        temp_file.write(eval_dataset.getvalue())
                        eval_dataset_path = temp_file.name
                    st.success(f"✅ Loaded: {eval_dataset.name}")
            
            # Expected formats for scripts (Task B/C only)
            EXPECTED_FORMATS = {
                "Task B - Generation": '''
                {
                    "conversation_id": "...",
                    "task_id": "...",
                    "contexts": [{"document_id": "...", "text": "..."}],
                    "predictions": [{"text": "..."}]
                }''',
                "Task C - Full RAG (Rewrite + Retrieval + Generation)": '''
                {
                    "conversation_id": "...",
                    "task_id": "...",
                    "contexts": [{"document_id": "...", "text": "..."}],
                    "predictions": [{"text": "..."}]
                }'''
            }
            
            with dataset_col2:
                st.markdown("**Expected Format:**")
                st.code(EXPECTED_FORMATS.get(selected_category, "Upload a valid JSONL file"), language="json")
        
        # else:  # Use Last Run
        #     if st.session_state.run_result:
        #         use_last_run = True
        #         run = st.session_state.run_result
        #         st.success(f"✅ Using last run: Task {run.get('task', 'N/A')}")
                
        #         result_col1, result_col2 = st.columns(2)
        #         with result_col1:
        #             st.markdown("**Query from last run:**")
        #             st.info(run.get('query', 'N/A'))
        #             eval_query = run.get('query')
                
        #         with result_col2:
        #             st.markdown("**Provide Ground Truth (Optional):**")
        #             eval_ground_truth_answer = st.text_area(
        #                 "Expected Answer for this query",
        #                 placeholder="Enter the correct answer to compare against...",
        #                 height=100,
        #                 key="eval_last_run_gt"
        #             )
        #     else:
        #         st.warning("⚠️ No pipeline run available. Run a query in RAG Playground first.")
        
        st.divider()
        
        # Evaluation Configuration
        st.subheader("⚙️ Evaluation Configuration")
        
        config_col1, config_col2 = st.columns(2)
        judge_provider = "ibm-granite/granite-3.3-8b-instruct"
        
        if eval_selected_task in ["B", "C"]:
            with config_col1:
                st.markdown("**LLM-as-Judge Settings**")
                judge_provider = st.selectbox(
                    "ibm-granite/granite-3.3-8b-instruct",
                    ["Same as Generation", "Custom"],
                    key="eval_judge_provider"
                )
                if judge_provider == "Custom":
                    judge_provider = st.text_input("Custom Judge LLM Provider", placeholder=
                        "eg. ibm-granite/granite-3.3-8b-instruct",
                        value="ibm-granite/granite-3.3-8b-instruct",
                        key="custom_judge_llm_provider"
                    )
                
        
        with config_col2:
            st.markdown("**Output Options**")
            show_detailed = st.checkbox("Show detailed results", value=True, key="eval_detailed")
            export_results = st.checkbox("Export results to JSON", value=True, key="eval_export")
        
        st.divider()
        
        # Run Evaluation Button
        # Determine if ready to evaluate
        if eval_selected_task == "A":
            # Task A requires BEIR data + predictions file
            eval_ready = (beir_qrels is not None and beir_queries is not None and 
                         'task_a_predictions_path' in dir() and task_a_predictions_path is not None)
        else:
            eval_ready = eval_dataset is not None
        
        if st.button("🚀 Run Evaluation", type="primary", disabled=not eval_ready):
            if not eval_ready:
                if eval_selected_task == "A":
                    st.warning("Please select a valid corpus with queries and qrels.")
                else:
                    st.warning("Please upload a file to evaluate.")
            else:
                with st.spinner("Running evaluation..."):
                    import time
                    eval_start = time.time()
                    
                    # Set up venv python path
                    if platform.system() == "Windows":
                        venv_python = os.path.abspath("src/evaluation/venv/Scripts/python.exe")
                    else:
                        venv_python = os.path.abspath("src/evaluation/venv/bin/python")
                    
                    if eval_selected_task == "A":
                        # Task A: Run retrieval evaluation with official script
                        st.subheader("📊 Task A Retrieval Evaluation")
                        
                        # Show corpus and query info
                        st.info(f"""
                        **Evaluation Configuration:**
                        - Corpus: **{selected_corpus.upper()}**
                        - Query Type: **{QUERY_TYPES[selected_query_type]}**
                        - Total Queries: **{len(beir_queries)}**
                        - Queries with Relevance Judgments: **{len(beir_qrels)}**
                        """)
                        
                        # Check if predictions file was uploaded
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
                                
                                result = subprocess.run(
                                    run_retrieval_eval_command, 
                                    capture_output=True, 
                                    text=True,
                                    cwd=os.path.dirname(os.path.abspath(__file__))
                                )
                                
                                if result.returncode == 0:
                                    status.update(label="✅ Retrieval Evaluation Complete!", state="complete")
                                    
                                    # Show stdout output (contains metrics)
                                    if result.stdout:
                                        st.subheader("📈 Retrieval Metrics")
                                        st.code(result.stdout)
                                    
                                    # Check for aggregate CSV file
                                    aggregate_csv = output_path.replace("_results.json", "_results_aggregate.csv")
                                    if os.path.exists(aggregate_csv):
                                        st.subheader("📋 Aggregate Results")
                                        import pandas as pd
                                        df = pd.read_csv(aggregate_csv)
                                        st.dataframe(df)
                                    
                                    # Download enriched results
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
                                        st.code(result.stderr)
                        else:
                            # No predictions file - show sample format
                            st.warning("⚠️ Please upload your retrieval predictions file above")
                            with st.expander("📋 Expected Input Format", expanded=True):
                                st.code('''
{
  "task_id": "dd6b6ffd177f2b311abe676261279d2f<::>2",
  "Collection": "mt-rag-clapnq-elser-512-100-20240503",
  "contexts": [
    {"document_id": "822086267_7384-8758-0-1374", "score": 27.759},
    {"document_id": "123456789_1234-5678", "score": 25.123}
  ]
}''', language="json")
                        
                        eval_time = time.time() - eval_start
                        st.caption(f"⏱️ Completed in {eval_time:.2f}s")
                    
                    else:
                        # Task B/C: Use existing evaluation scripts
                        run_data = None
                        if platform.system() == "Windows":
                            venv_python = os.path.abspath("src/evaluation/venv/Scripts/python.exe")
                        else:
                            # Linux and macOS
                            venv_python = os.path.abspath("src/evaluation/venv/bin/python")
                        
                        suffix = f".{eval_dataset.name.split('.')[-1]}" if eval_dataset else ".jsonl"
                        output_path = eval_dataset_path.replace(suffix, "_results.json")
                    
                        run_retrieval_eval_command = [
                            venv_python, "src/evaluation/run_retrieval_eval.py",
                            "--input_file", eval_dataset_path,
                            "--output_file", output_path,
                        ]
                        
                        run_gen_eval_command = [
                            venv_python, "src/evaluation/run_generation_eval.py",
                            "-i", eval_dataset_path,
                            "-o", output_path,
                            "-e", "src/evaluation/config.yaml",
                            "--provider", "hf",
                            "--judge_model", judge_provider
                        ]
                        selected_script = run_gen_eval_command
                        
                        with st.status("Evaluating...", expanded=True) as status:
                            result = subprocess.run(selected_script, capture_output=True, text=True)
                            
                            if result.returncode == 0:
                                status.update(label="✅ Evaluation Complete!", state="complete")
                                time.sleep(2)
                                # Provide the download button
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
                    # if use_last_run and st.session_state.run_result:
                    #     # Use existing run result
                    #     run_data = st.session_state.run_result
                    #     st.info(f"📋 Using last run result (Task {run_data.get('task', 'N/A')})")
                    
                    # elif eval_query:
                    #     # Need to run the pipeline first
                    #     st.info(f"🔄 Running pipeline for: {eval_query[:50]}...")
                        
                    #     if st.session_state.vector_store is None:
                    #         st.error("❌ No vector store loaded. Please upload documents first.")
                    #     else:
                    #         # Run retrieval
                    #         try:
                    #             retriever_config = st.session_state.selected_components.get("retriever", {})
                    #             top_k = retriever_config.get("top_k", eval_top_k)
                                
                    #             docs_with_scores = st.session_state.vector_store.similarity_search_with_score(
                    #                 eval_query, k=top_k
                    #             )
                                
                    #             retrieval_results = []
                    #             for doc, score in docs_with_scores:
                    #                 retrieval_results.append({
                    #                     "content": doc.page_content[:500],
                    #                     "score": float(score),
                    #                     "metadata": doc.metadata
                    #                 })
                                
                    #             run_data = {
                    #                 "task": eval_selected_task,
                    #                 "query": eval_query,
                    #                 "retrieved_docs": [doc for doc, _ in docs_with_scores],
                    #                 "retrieval": {
                    #                     "results": retrieval_results,
                    #                     "num_results": len(docs_with_scores)
                    #                 }
                    #             }
                                
                    #             # Run generation if Task B or C
                    #             if eval_selected_task in ["B", "C"]:
                    #                 gen_config = st.session_state.selected_components.get("generator", {})
                    #                 gen_model = gen_config.get("model", model_name)
                                    
                    #                 context = "\n\n".join([doc.page_content for doc, _ in docs_with_scores])
                    #                 prompt_template = gen_config.get("prompt_template", "Answer based on context: {context}\n\nQuestion: {question}")
                    #                 formatted_prompt = prompt_template.format(context=context, question=eval_query)
                                    
                    #                 llm = get_llm(provider, api_key, base_url, gen_model)
                    #                 response = llm.invoke([HumanMessage(content=formatted_prompt)])
                    #                 answer = response.content if hasattr(response, 'content') else str(response)
                                    
                    #                 run_data["generation"] = {"answer": answer, "model": gen_model}
                                
                    #         except Exception as e:
                    #             st.error(f"❌ Pipeline execution failed: {e}")
                    #             run_data = None
                    
                    # if run_data:
                    #     # Run selected evaluation scripts
                    #     st.subheader("📊 Evaluation Results")
                        
                    #     # Get LLM for judge if needed
                    #     judge_llm = None
                    #     if judge_provider != "Same as Generation":
                    #         try:
                    #             judge_llm = get_llm(judge_provider, api_key, base_url, model_name)
                    #         except:
                    #             pass
                    #     else:
                    #         try:
                    #             judge_llm = get_llm(provider, api_key, base_url, model_name)
                    #         except:
                    #             pass
                        
                    #     # Parse ground truth docs
                    #     gt_docs = None
                    #     if eval_ground_truth_docs:
                    #         gt_docs = [d.strip() for d in eval_ground_truth_docs.split(",") if d.strip()]
                        
                    #     # Run each evaluation script
                    #     all_results = []
                    #     for script_name in selected_scripts:
                    #         result = run_evaluation(
                    #             script_name=script_name,
                    #             run_result=run_data,
                    #             ground_truth_answer=eval_ground_truth_answer,
                    #             ground_truth_docs=gt_docs,
                    #             llm=judge_llm
                    #         )
                    #         all_results.append(result)
                        
                    #     eval_time = time.time() - eval_start
                        
                    #     # Calculate overall metrics
                    #     overall_score = sum(r.overall_score for r in all_results) / len(all_results) if all_results else 0
                    #     passed_count = sum(1 for r in all_results if r.passed)
                        
                    #     # Display summary metrics
                    #     summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                    #     with summary_col1:
                    #         st.metric("📊 Overall Score", f"{overall_score:.2%}")
                    #     with summary_col2:
                    #         st.metric("✅ Passed", f"{passed_count}/{len(all_results)}")
                    #     with summary_col3:
                    #         st.metric("📋 Scripts Run", len(all_results))
                    #     with summary_col4:
                    #         st.metric("⏱️ Eval Time", f"{eval_time:.2f}s")
                        
                    #     st.markdown("---")
                        
                    #     # Display individual results
                    #     st.markdown("### 📋 Script Results")
                        
                    #     for result in all_results:
                    #         status_icon = "✅" if result.passed else "❌"
                    #         score_color = "green" if result.passed else "red"
                            
                    #         with st.expander(f"{status_icon} {result.script_name} - Score: {result.overall_score:.2%}", expanded=True):
                    #             # Score and status row
                    #             score_col1, score_col2 = st.columns([2, 3])
                                
                    #             with score_col1:
                    #                 st.markdown(f"**Score:** `{result.overall_score:.4f}`")
                    #                 st.progress(result.overall_score)
                    #                 st.markdown(f"**Status:** {status_icon} {'Passed' if result.passed else 'Failed'}")
                                
                    #             with score_col2:
                    #                 st.markdown("**Explanation:**")
                    #                 st.info(result.explanation)
                                
                    #             # Metrics breakdown
                    #             if result.metrics:
                    #                 st.markdown("**Metrics:**")
                    #                 metric_cols = st.columns(len(result.metrics))
                    #                 for i, (metric_name, metric_value) in enumerate(result.metrics.items()):
                    #                     with metric_cols[i % len(metric_cols)]:
                    #                         if isinstance(metric_value, float):
                    #                             st.metric(metric_name, f"{metric_value:.4f}")
                    #                         else:
                    #                             st.metric(metric_name, str(metric_value))
                                
                    #             # Raw details (optional)
                    #             if show_detailed and result.details:
                    #                 with st.expander("🔍 Raw Evaluation Details"):
                    #                     st.json(result.details)
                        
                        # Export option
                        # if export_results:
                        #     export_data = {
                        #         "overall_score": overall_score,
                        #         "evaluation_time_s": eval_time,
                        #         "results": [
                        #             {
                        #                 "script": r.script_name,
                        #                 "score": r.overall_score,
                        #                 "passed": r.passed,
                        #                 "metrics": r.metrics,
                        #                 "explanation": r.explanation
                        #             }
                        #             for r in all_results
                        #         ]
                        #     }
                        #     st.download_button(
                        #         "📥 Download Results JSON",
                        #         data=json.dumps(export_data, indent=2),
                        #         file_name="evaluation_results.json",
                        #         mime="application/json"
                        #     )
                        
                        # # Raw evaluation logs
                        # with st.expander("🐛 Raw Evaluation Logs", expanded=False):
                        #     st.markdown("### Evaluation Configuration")
                        #     st.write(f"• **Task:** {eval_selected_task}")
                        #     st.write(f"• **Scripts:** {', '.join(selected_scripts)}")
                        #     st.write(f"• **Judge Provider:** {judge_provider}")
                        #     st.write(f"• **Ground Truth Provided:** {'Yes' if eval_ground_truth_answer else 'No'}")
                            
                        #     st.markdown("### Run Data Used")
                        #     st.json({
                        #         "query": run_data.get("query"),
                        #         "task": run_data.get("task"),
                        #         "has_retrieval": "retrieval" in run_data,
                        #         "has_generation": "generation" in run_data,
                        #         "retrieval_count": run_data.get("retrieval", {}).get("num_results", 0)
                        #     })
                            
                        #     st.markdown("### All Results (Raw)")
                        #     for r in all_results:
                        #         st.markdown(f"**{r.script_name}**")
                        #         st.json({
                        #             "overall_score": r.overall_score,
                        #             "metrics": r.metrics,
                        #             "passed": r.passed,
                        #             "explanation": r.explanation,
                        #             "details": r.details
                        #         })
        
        # Show current run result summary
        # if st.session_state.run_result and use_last_run:
        #     with st.expander("📋 Current Run Result (for evaluation)", expanded=False):
        #         run = st.session_state.run_result
        #         st.write(f"**Task:** {run.get('task')}")
        #         st.write(f"**Query:** {run.get('query')}")
        #         st.write(f"**Has Retrieval:** {'✅' if 'retrieval' in run else '❌'}")
        #         st.write(f"**Has Generation:** {'✅' if 'generation' in run else '❌'}")
        #         if 'generation' in run:
        #             st.write(f"**Answer Preview:** {run['generation'].get('answer', '')[:200]}...")

        # ==================== TAB 3: Chat (existing) ====================
    with tab_chat:
        st.header("Chat")
        
        if not st.session_state.current_session_id:
            st.info("Please create or select a session from the sidebar to start chatting.")
        else:
            # Load history for current session
            history = load_session_history(st.session_state.current_session_id)
            
            for message in history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("Ask a question about your documents..."):
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                logger.info(f"Received user query: {prompt}")
                save_message(st.session_state.current_session_id, "user", prompt, collection_name)

                if st.session_state.vector_store is None:
                    st.warning("Please upload and process a JSON file first.")
                else:
                    with st.chat_message("assistant"):
                        try:
                            retriever = get_retriever(st.session_state.vector_store, k=retrieval_top_k)
                            llm = get_llm(provider, api_key, base_url, model_name)
                            rag_chain = create_rag_chain(llm, retriever)
                            
                            logger.info("Invoking RAG chain")
                            response = rag_chain.invoke({"input": prompt})
                            answer = response["answer"]
                            context = response.get("context", [])
                            logger.info("Generated response")
                            
                            st.markdown(answer)
                            
                            # Display References
                            if context:
                                with st.expander("📚 View References"):
                                    for i, doc in enumerate(context):
                                        st.markdown(f"**Reference {i+1}**")
                                        st.markdown(f"*Source:* {doc.metadata.get('title', 'Unknown')} | *ID:* {doc.metadata.get('id', 'N/A')}")
                                        st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                                        st.divider()

                            save_message(st.session_state.current_session_id, "assistant", answer, collection_name)
                        except Exception as e:
                            logger.error(f"Error generating response: {e}", exc_info=True)
                            st.error(f"Error generating response: {e}")

    # ==================== TAB 4: Manage Collection (existing) ====================
    with tab_manage:
        st.header("Manage Collection")
        if st.session_state.vector_store:
            try:
                # Retrieve all documents
                collection_data = st.session_state.vector_store.get()
                num_docs = len(collection_data['ids'])
                st.write(f"**Total Chunks in '{st.session_state.get('current_collection', 'default')}'**: {num_docs}")
                
                # --- Add Manual Chunk ---
                with st.expander("➕ Add Manual Chunk"):
                    with st.form("add_chunk_form"):
                        new_text = st.text_area("Content")
                        new_metadata_str = st.text_area("Metadata (JSON)", value='{"source": "manual"}')
                        submitted = st.form_submit_button("Add Chunk")
                        if submitted and new_text:
                            try:
                                new_metadata = json.loads(new_metadata_str)
                                doc = Document(page_content=new_text, metadata=new_metadata)
                                add_to_vector_store(st.session_state.vector_store, [doc])
                                st.success("Chunk added successfully!")
                                st.rerun()
                            except json.JSONDecodeError:
                                st.error("Invalid JSON for metadata")
                            except Exception as e:
                                st.error(f"Error adding chunk: {e}")

                # --- Inspect & Delete ---
                if num_docs > 0:
                    st.subheader("Inspect & Remove Chunks")
                    
                    # Prepare data for editor
                    data_for_df = []
                    for i in range(num_docs):
                        data_for_df.append({
                            "Select": False,
                            "ID": collection_data['ids'][i],
                            "Content": collection_data['documents'][i],
                            "Metadata": str(collection_data['metadatas'][i]) # Convert to string for display
                        })
                    
                    # Convert to DataFrame to ensure st.data_editor returns a DataFrame
                    df_to_edit = pd.DataFrame(data_for_df)
                    
                    edited_df = st.data_editor(
                        df_to_edit,
                        column_config={
                            "Select": st.column_config.CheckboxColumn(
                                "Select",
                                help="Select to delete",
                                default=False,
                            )
                        },
                        disabled=["ID", "Content", "Metadata"],
                        hide_index=True,
                    )

                    if st.button("Delete Selected Chunks"):
                        selected_rows = edited_df[edited_df.Select]
                        if not selected_rows.empty:
                            ids_to_delete = selected_rows["ID"].tolist()
                            delete_from_vector_store(st.session_state.vector_store, ids_to_delete)
                            st.success(f"Deleted {len(ids_to_delete)} chunks.")
                            st.rerun()
                        else:
                            st.warning("No chunks selected.")
            except Exception as e:
                logger.error(f"Error inspecting vector store: {e}", exc_info=True)
                st.error(f"Error retrieving documents: {e}")
        else:
            st.info("No vector store loaded.")

    # ==================== TAB 5: Database Inspector (existing) ====================
    with tab_db:
        st.header("Database Inspector")
        st.subheader("Sessions Table")
        try:
            sessions = get_sessions()
            if sessions:
                st.dataframe(sessions)
            else:
                st.info("No sessions found.")
        except Exception as e:
            st.error(f"Error loading sessions: {e}")

        st.divider()

        st.subheader("Messages Table")
        # We need a way to get all messages or filter by session
        # Let's add a simple query to get all messages for inspection
        try:
            import sqlite3
            conn = sqlite3.connect("chat_history.db")
            # import pandas as pd # Removed redundant import
            df_messages = pd.read_sql_query("SELECT * FROM messages ORDER BY timestamp DESC", conn)
            conn.close()
            
            if not df_messages.empty:
                st.dataframe(df_messages)
            else:
                st.info("No messages found.")
        except Exception as e:
            st.error(f"Error loading messages: {e}")

if __name__ == "__main__":
    main()
