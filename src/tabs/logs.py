import streamlit as st
import os


def render():
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
