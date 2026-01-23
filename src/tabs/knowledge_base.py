import streamlit as st
import tempfile
import os
import logging
import json
from src.ingestion import load_json_documents
from src.vector_store import setup_vector_store
from src.file_manager import FileManager

logger = logging.getLogger(__name__)

def render():
    st.header("📚 Knowledge Base Management")
    
    # Initialize session state for embedding config if not present
    if "kb_embed_config" not in st.session_state:
        st.session_state.kb_embed_config = {}

    kb_col1, kb_col2 = st.columns(2)
    
    # Variables to hold selection state
    selected_db_type = "FAISS"
    selected_model_name = "default"
    selected_collection_name = "default"
    is_existing = False
    
    # ==========================================
    # COLUMN 1: Vector Database & Collection
    # ==========================================
    with kb_col1:
        st.subheader("1. Vector Database & Collection")
        db_action = st.radio("Action", ["Create New Collection", "Load Existing"], index=0, key="kb_action")
        
        if db_action == "Load Existing":
            is_existing = True
            
            # 1. Select DB Type
            # Scan directories in 'collections/' to see which DBs exist
            available_dbs = []
            if os.path.exists(FileManager.BASE_COLLECTIONS_DIR):
                available_dbs = [d for d in os.listdir(FileManager.BASE_COLLECTIONS_DIR) 
                                 if os.path.isdir(os.path.join(FileManager.BASE_COLLECTIONS_DIR, d))]
            
            if not available_dbs:
                st.warning("No collections found.")
                st.stop()
                
            selected_db_type = st.selectbox("Select Vector DB", available_dbs, key="kb_sel_db")
            
            # 2. Select Embedding Model (Sub-folder of DB)
            db_path = os.path.join(FileManager.BASE_COLLECTIONS_DIR, selected_db_type)
            available_models = [d for d in os.listdir(db_path) if os.path.isdir(os.path.join(db_path, d))]
            
            if not available_models:
                st.warning(f"No embedding models found for {selected_db_type}.")
                st.stop()
                
            selected_model_name = st.selectbox("Select Embedding Model", available_models, key="kb_sel_model")
            
            # 3. Select Collection (Sub-folder of Model)
            available_collections = FileManager.list_collections(selected_db_type, selected_model_name)
            
            if not available_collections:
                st.warning("No collections found for this configuration.")
                st.stop()
                
            selected_collection_name = st.selectbox("Select Collection", available_collections, key="kb_sel_col")
            
            # Load info.json to populate Col 2
            info_path = os.path.join(
                FileManager.get_collection_path(selected_db_type, selected_model_name, selected_collection_name),
                "info.json"
            )
            
            if os.path.exists(info_path):
                with open(info_path, "r") as f:
                    loaded_info = json.load(f)
                    # Update session state for Col 2 to render strictly
                    st.session_state.kb_loaded_embed_config = loaded_info.get("embedding", {})
            else:
                st.warning("⚠️ 'info.json' missing. Embedding settings unknown.")
                st.session_state.kb_loaded_embed_config = {}

        else: # Create New
            is_existing = False
            selected_db_type = st.selectbox("Vector DB Type", ["FAISS", "Qdrant"], index=0, key="kb_new_db_type")
            selected_collection_name = st.text_input("Collection Name", value="my_collection", key="kb_new_col_name")
            
            # Sanitize name immediately for feedback
            clean_name = FileManager._sanitize(selected_collection_name)
            if clean_name != selected_collection_name:
                st.caption(f"Will be saved as: `{clean_name}`")
                selected_collection_name = clean_name

    # ==========================================
    # COLUMN 2: Embedding Configuration
    # ==========================================
    with kb_col2:
        st.subheader("2. Embedding Configuration")
        
        # If loading existing, we force the display to match the loaded config
        if is_existing:
            loaded_cfg = st.session_state.get("kb_loaded_embed_config", {})
            st.info("🔒 Configuration locked to match selected collection.")
            
            st.text_input("Provider", value=loaded_cfg.get("provider", "Unknown"), disabled=True)
            st.text_input("Model Name", value=loaded_cfg.get("model_name", "Unknown"), disabled=True)
            
            # We construct the config object for the ingestion/setup function
            embedding_config = loaded_cfg
            
        else:
            # Standard Selection Logic
            embedding_provider = st.selectbox("Embedding Provider", ["Local"], index=0, key="kb_prov_new")
            
            embedding_api_key = None
            embed_base_url = None
            model_name = "default"
            
            LOCAL_EMBEDDING_MODELS = {
                "nomic-embed-text": {"dim": 768},
                "mxbai-embed-large": {"dim": 1024},
                "all-minilm": {"dim": 384},
                "all-minilm:l6-v2": {"dim": 384},
                "jina/jina-embeddings-v2-small-en" : {"dim" : 512}
            }
            
            # Local Provider Logic
            embed_base_url = st.text_input("Base URL", value="http://localhost:11434", key="kb_new_url")
            model_name = st.selectbox("Model", list(LOCAL_EMBEDDING_MODELS.keys()), index=0, key="kb_new_local_model")
            embedding_config = {
                "provider": "Local", 
                "base_url": embed_base_url, 
                "model_name": model_name
            }
            
            # Check if this new collection would overwrite an existing one
            target_path = FileManager.get_collection_path(selected_db_type, model_name, selected_collection_name)
            if os.path.exists(target_path):
                st.warning(f"⚠️ Collection already exists at:\n`{target_path}`\nIngesting will append to it.")

    st.divider()

    # ==========================================
    # 3. Data Ingestion
    # ==========================================
    st.subheader("3. Data Ingestion")
    
    uploaded_file = st.file_uploader("Upload Documents (JSON/JSONL)", type=["json", "jsonl"])

    if uploaded_file and st.button("Process & Ingest File"):
        with st.spinner("Processing..."):
            try:
                suffix = ".jsonl" if uploaded_file.name.endswith(".jsonl") else ".json"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                documents = load_json_documents(tmp_file_path)
                
                if documents:
                    logger.info(f"Starting ingestion of {len(documents)} documents into {selected_collection_name} ({selected_db_type})")
                    # Configuration for vector store
                    vd_config = {"vector_db_type": selected_db_type}
                    
                    vector_store = setup_vector_store(
                        documents=documents,
                        embedding_config=embedding_config,
                        collection_name=selected_collection_name,
                        db_type=selected_db_type,
                        db_config=vd_config
                    )
                    
                    st.success(f"Successfully ingested {len(documents)} documents.")
                    saved_path = FileManager.get_collection_path(selected_db_type, embedding_config.get('model_name', 'default'), selected_collection_name)
                    st.info(f"Saved to: `{saved_path}`")
                    logger.info(f"Ingestion complete. Saved {len(documents)} documents to {saved_path}")
                    
                else:
                    st.error("No valid documents found.")
                
                os.remove(tmp_file_path)
                
            except Exception as e:
                st.error(f"Ingestion failed: {e}")
                logger.error(f"Ingestion Error: {e}", exc_info=True)