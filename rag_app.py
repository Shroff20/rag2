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
allowed_filetpyes = [".pdf"]


def set_session_state(datadict: dict):
    for key, value in datadict.items():
        if key not in st.session_state:
            print(f'setting {key} = {value}')
            st.session_state[key] = value


def update_document_count():
    N_documents = st.session_state["DS"].collection.metadata["N_full_documents"]
    st.session_state["h_document_count"].metric(
        "Loaded documents", N_documents, border=True
    )


def delete_all_collections():
    st.session_state["DS"].clear_collections()
    update_document_count()


def initalize():

    if "DS" not in st.session_state:
        DS = shroff_rag.DataStore(device, database_folder)
        DS.create_collection()
        st.session_state["DS"] = DS

    config = {
        "messages": [],
        "documents_to_process": []
    }
    set_session_state(config)


def process_files(files):
    N_files = len(files)

    starting_document_count = st.session_state["DS"].collection.metadata[
        "N_full_documents"
    ]
    percent_complete = 0.0
    for i, file in enumerate(files):
        with st.sidebar:
            st.session_state["h_progress"].progress(
                percent_complete / 100, text=f"Processing: {percent_complete:.1f}%"
            )
            temp_dir = tempfile.mkdtemp()
            tmp_file_path = os.path.join(temp_dir, file.name)
            with open(tmp_file_path, "wb") as f:
                f.write(file.getvalue())
            st.session_state["DS"].add_document(tmp_file_path)
            percent_complete = 100.0 * (i + 1) / N_files
            st.session_state["h_progress"].progress(
                percent_complete / 100, text=f"Processing: {percent_complete:.1f}%"
            )
            current_document_count = st.session_state["DS"].collection.metadata[
                "N_full_documents"
            ]
            print("count", current_document_count)

            update_document_count()


def run_query(query):
    results = st.session_state["DS"].query(query)
    return results


def display_chat():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def clear_chat():
    st.session_state.messages = []


prompt = st.chat_input(
    "Enter your query?", key="h_prompt"
)  # cannot be in a tab to force positioning to bottom

initalize()


tab1, tab2, tab3, tab4 = st.tabs(["query", "search", "custer", "list documents"])

with st.sidebar:
    h_document_count = st.metric(
        "Loaded documents",
        st.session_state["DS"].collection.metadata["N_full_documents"],
        border=True,
    )
    st.session_state["h_document_count"] = h_document_count

    h_progress = st.progress(0 / 100, text=f"Nothing to process")
    st.session_state["h_progress"] = h_progress

    with st.form("my-form", clear_on_submit=True):
        h_uploaded_files = st.file_uploader(
            "Upload documents",
            accept_multiple_files=True,
            type=allowed_filetpyes,
            label_visibility="hidden",
            #key="h_uploaded_files",
        )
        submitted = st.form_submit_button("process")
        if submitted:
            st.session_state['documents_to_process'].append(h_uploaded_files)
            process_files(files=h_uploaded_files)

    st.write(f"database path: {database_folder}")

    st.button(label="delete all documents", on_click=delete_all_collections)


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
    st.button("clear chat", on_click=clear_chat)

with tab4:
    df = st.session_state["DS"]._get_simplified_document_df()

    col_options = ['basename', 'document', 'id', 'creation_date', 'fullpath', 'author', 'page_lengths', 'file_ext', 'modification_date', 'upload_date']
    selcted_cols = st.multiselect("Select a column", col_options, default=["basename", "document"], key = 'h_multiselect')

    print(f"making dataframe with {selcted_cols}")

    allowed_cols = [col for col in selcted_cols if col in df.columns]
    st.dataframe(df[allowed_cols])

    st.write('Note: "document" column does not display the full text')