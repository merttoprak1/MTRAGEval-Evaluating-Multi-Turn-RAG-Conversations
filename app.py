import logging
import streamlit as st
from src.tabs import evaluation, knowledge_base, logs, playground
import sys
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


    with tab_interactive:
        playground.render()

    with tab_batch:
        evaluation.render()

    with tab_kb:
        knowledge_base.render()

    with tab_logs:
        logs.render
    
if __name__ == "__main__":
    main()
