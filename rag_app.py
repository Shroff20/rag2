# streamlit run rag_app.py

import streamlit as st
import pandas as pd
import importlib
import shroff_rag 

importlib.reload(shroff_rag)
import tempfile
import os


device = "cuda"
database_folder = "./database_data"
os.environ["GOOGLE_API_KEY"] = "AIzaSyCp2r7wKvf_aNLc1gxzJTdhLrVAwaS-0WM"
allowed_filetpyes = ['.pdf']

def set_session_state(datadict:dict):
    for key, value in datadict.items():
            if key not in st.session_state:
                st.session_state[key] = value


def delete_all_collections():
     st.session_state["DS"].clear_collections()

def initalize():

    if 'DS' not in st.session_state:
        DS = shroff_rag.DataStore(device, database_folder)
        DS.create_collection()
        st.session_state["DS"] = DS

    config = {
        "selected_cols": ["basename", "document"],
        "device": device,
        "messages": [],
    }
    set_session_state(config)


def inititalize_sidebar():
    with st.sidebar:

        h_document_count = st.metric(
        "Loaded documents", st.session_state["DS"].collection.metadata['N_full_documents'], border=True
        )
        st.session_state["h_document_count"] = h_document_count
     
        h_progress= st.progress(
            0 / 100, text=f"Nothing to process"
        )
        st.session_state["h_progress"] = h_progress

        h_uploaded_files = st.file_uploader(
                "Upload documents",
                accept_multiple_files=True,
                type=allowed_filetpyes,
                label_visibility="hidden",
                key = 'h_uploaded_files',
                on_change =  process_files
            )
                  
        st.write(f"database path: {database_folder}")

        st.button(label = 'delete all documents', on_click = delete_all_collections)

    


def process_files():
    N_files = len(st.session_state["h_uploaded_files"])
    print(st.session_state["h_uploaded_files"])
    for i in range(N_files):
        uploaded_file = st.session_state["h_uploaded_files"][i]
        percent_complete = 0.0
        with st.sidebar:
            st.session_state["h_progress"].progress(
                percent_complete / 100, text=f"Processing: {percent_complete:.1f}%"
            )
            temp_dir = tempfile.mkdtemp()
            tmp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(tmp_file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            st.session_state["DS"].add_document(tmp_file_path)
            percent_complete = 100.0 * (i + 1) / N_files
            with st.sidebar:
                st.session_state["h_progress"].progress(
                    percent_complete / 100, text=f"Processing: {percent_complete:.1f}%"
                )


def run_query(query):
    results = st.session_state["DS"].query(query)
    return results


def display_chat():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def clear_chat():
    st.session_state.messages = []



prompt = st.chat_input("Enter your query?", key = 'h_prompt')  # cannot be in a tab to force positioning to bottom

initalize()
inititalize_sidebar()
process_files()


tab1, tab2, tab3, tab4= st.tabs(["query", "search", "custer", "list documents"])


with tab1:
    st.title("Ask me anything")
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # React to user input

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt is not None:
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = run_query(prompt).content
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("user").markdown(prompt)
        st.chat_message("assistant").markdown(response)
    st.button('clear chat', on_click = clear_chat)

with tab4:
    df = st.session_state['DS']._get_simplified_document_df()
    cols = st.multiselect ('Select a column', df.columns, default = st.session_state["selected_cols"])
    st.dataframe(df.loc[:, cols])
    st.session_state["selected_cols"] =  cols
    print(st.session_state["selected_cols"])

