import streamlit as st
import pandas as pd
import tempfile
import os
import logging
import sys
import json
from src.ingestion import load_json_documents, chunk_documents
from src.vector_store import setup_vector_store, get_retriever, add_to_vector_store, delete_from_vector_store
from src.llm_client import get_llm
from src.rag import create_rag_chain
from src.database import init_db, create_session, get_sessions, save_message, load_session_history, delete_session, rename_session
from langchain_core.documents import Document

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

    if embedding_provider == "OpenAI":
        # Reuse API key if already provided, or ask for it
        if not api_key:
             embedding_api_key = st.sidebar.text_input("OpenAI Embedding API Key", type="password")
        else:
             embedding_api_key = api_key
        
        embedding_config = {
            "provider": "OpenAI",
            "api_key": embedding_api_key
        }
    elif embedding_provider == "Gemini":
        # Reuse API key if confirmed Gemini, or ask
        if provider == "Gemini" and api_key:
            embedding_api_key = api_key
        else:
            embedding_api_key = st.sidebar.text_input("Gemini Embedding API Key", type="password")
            
        embedding_config = {
             "provider": "Gemini",
             "api_key": embedding_api_key,
             "model_name": "models/text-embedding-004"
        }
    else:
        # Local Ollama
        # Default to standard Ollama port if not specified, but user might want different URL for embeddings
        embed_base_url = st.sidebar.text_input("Embedding Base URL", value="http://localhost:11434")
        embed_model = st.sidebar.text_input("Embedding Model", value="shaw/dmeta-embedding-zh-small-q4")
        
        embedding_config = {
            "provider": "Local",
            "base_url": embed_base_url,
            "model_name": embed_model
        }

    # Collection Configuration
    st.sidebar.subheader("Collection Management")
    collection_name = st.sidebar.text_input("Collection Name", value="default_collection")
    
    # Initialize Vector Store with selected collection (load existing data)
    if "vector_store" not in st.session_state or st.session_state.get("current_collection") != collection_name:
         # Try to load existing collection
         try:
             st.session_state.vector_store = setup_vector_store(documents=None, embedding_config=embedding_config, collection_name=collection_name)
             st.session_state.current_collection = collection_name
             # st.sidebar.success(f"Loaded collection: {collection_name}")
         except Exception as e:
             # Might fail if collection doesn't exist yet and we try to load without docs, 
             # but Chroma usually handles empty collections gracefully.
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
                        st.session_state.vector_store = setup_vector_store(chunks, embedding_config=embedding_config, collection_name=collection_name)
                        st.session_state.current_collection = collection_name
                        st.sidebar.success(f"Ingested into collection: {collection_name}")
                        logger.info("File processing completed successfully")
                    
                    # Cleanup temp file
                    os.remove(tmp_file_path)
                    
                except Exception as e:
                    logger.error(f"Error processing file: {e}", exc_info=True)
                    st.error(f"Error processing file: {e}")

    # --- Main Content ---
    
    # Tabs for Chat and Management
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🛠️ Manage Collection", "🔍 Database Inspector"])

    with tab1:
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
                            retriever = get_retriever(st.session_state.vector_store)
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

    with tab2:
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

    with tab3:
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
