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
logger = logging.getLogger(__name__)

def render():
    st.header("📚 Knowledge Base Management")
    
    kb_col1, kb_col2 = st.columns(2)
    embedding_config = {}
    vd_config = {}
    collection_infos = {"embedding" : {}, "collection" : {}}
    with kb_col1:
        st.subheader("1. Vector Database & Collection")
        db_existence = st.selectbox("New DB Or Old DB", ["New DB", "Old DB"], index=0)
        collections = os.listdir("collections")
        if db_existence is "New DB":
            vector_db_type = st.selectbox("Vector DB Type", ["FAISS", "Chroma", "Pinecone"], index=0)
            collection_name = st.text_input("Collection Name", value="default", key="collection_name") 
            if collection_name in collections:
                st.warning(
                    f"'{collection_name}' is alreasy exist. "
                    "Please type another name or select old db."
                )
                st.stop()
                
        elif db_existence is "Old DB":
            collection_name = st.selectbox("Select a Collection", collections, index=0)
            file_path = f"collections/{collection_name}/info.json"
            with open(file_path, "r", encoding="utf-8") as f:
                collection_infos = json.load(f)
            vector_db_type = collection_infos["collection"]["vector_db_type"]
               
        vd_config = {"vector_db_type": vector_db_type, "collection_name": collection_name}

    with kb_col2:
        st.subheader("2. Embedding Configuration")
        embedding_provider = st.selectbox("Embedding Provider", ["OpenAI", "Gemini", "Local"], index=2)
        embedding_model_default = collection_infos["embedding"]["model_name"] if "model_name" in collection_infos["embedding"] else None
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
        }      
        if embedding_provider in ("OpenAI", "Gemini"):
            models = OPENAI_EMBEDDING_MODELS if embedding_provider == "OpenAI" else GEMINI_EMBEDDING_MODELS
            batch_limits = (1, 2048, 100) if embedding_provider == "OpenAI" else (1, 100, 10)
            embedding_api_key = st.text_input(f"{embedding_provider} Embedding API Key", type="password")
            if db_existence is "Old DB":
                st.text_input(
                    "Embedding Model",
                    value=embedding_model_default,
                    disabled=True
                )
            else:
                embedding_model_default = st.selectbox(
                    "Embedding Model",
                    list(LOCAL_EMBEDDING_MODELS.keys()),
                    index=1
                )
                embedding_model_default
            model_info = models[embedding_model_default]
            st.caption(f"Dim: {model_info['dim']}")
            embedding_model_config = {"provider": embedding_provider, "api_key": embedding_api_key, "model_name": embedding_model_default, "dimension": model_info['dim']}
        else:
            embed_base_url = st.text_input("Embedding Base URL", value="http://localhost:11434")
            if db_existence is "Old DB":
                st.text_input(
                    "Embedding Model",
                    value=embedding_model_default,
                    disabled=True
                )
            else:
                embedding_model_default = st.selectbox(
                    "Embedding Model",
                    list(LOCAL_EMBEDDING_MODELS.keys())
                )
            embedding_model_config = {"provider": "Local", "base_url": embed_base_url, "model_name": embedding_model_default}

        embedding_config = {"model_name": embedding_model_default}
    
    st.divider()
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
                            metadata_collections = {
                                "embedding" : embedding_config,
                                "collection" : vd_config
                            }
                            vector_store = setup_vector_store(
                                documents, embedding_model_config, collection_name, vector_db_type, metadata_collections
                            )
                            st.success(f"Successfully ingested {len(documents)} documents into {collection_name}")
                    else:
                        uploaded_file = None
                        st.error("No valid documents found.")
                    os.remove(tmp_file_path)
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    # --- Management Section (Merged from old tab) ---
    st.divider()
    # with st.expander("🛠️ Inspect & Manage Collection"):
    #     if st.session_state.vector_store:
    #         try:
    #             collection_data = st.session_state.vector_store.get()
    #             num_docs = len(collection_data['ids'])
    #             st.write(f"Total documents: {num_docs}")
                
    #             if num_docs > 0:
    #                 data_for_df = []
    #                 for i in range(min(num_docs, 100)): # Limit preview
    #                     data_for_df.append({
    #                         "Select": False,
    #                         "ID": collection_data['ids'][i],
    #                         "Content": collection_data['documents'][i][:100] + "...",
    #                         "Metadata": str(collection_data['metadatas'][i])
    #                     })
                    
    #                 df_to_edit = pd.DataFrame(data_for_df)
    #                 edited_df = st.data_editor(df_to_edit, column_config={"Select": st.column_config.CheckboxColumn("Select", default=False)}, hide_index=True)

    #                 if st.button("Delete Selected"):
    #                     ids_to_delete = edited_df[edited_df.Select]["ID"].tolist()
    #                     if ids_to_delete:
    #                         delete_from_vector_store(st.session_state.vector_store, ids_to_delete)
    #                         st.success(f"Deleted {len(ids_to_delete)} chunks.")
    #                         st.rerun()
    #         except Exception as e:
    #             st.error(f"Error inspecting DB: {e}")
    #     else:
    #         st.info("No active vector store.")
