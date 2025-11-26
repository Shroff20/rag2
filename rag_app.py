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

def initialize_session(device):

    DS = shroff_rag.DataStore(device, database_folder)
    DS.create_collection()

    st.session_state["device"] = device
    st.session_state["DS"] = DS
    st.session_state["h_uploaded_files"] = None

    st.title("Shroff Search")
    st.write("Search your local documents!")

    st.text_input(
        "Enter your question", value="", help="Enter a question", key="query_textboox"
    )
    with st.sidebar:
        h_document_count = st.metric(
            "Loaded documents", st.session_state["DS"].collection.metadata['N_full_documents'], border=True
        )
        h_uploaded_files = st.file_uploader(
            "Upload documents",
            accept_multiple_files=True,
            type=allowed_filetpyes,
            label_visibility="hidden",
        )

        h_progress= st.progress(
                0 / 100, text=f"Nothing to process"
            )

    st.session_state["h_document_count"] = h_document_count
    st.session_state["h_uploaded_files"] = h_uploaded_files
    st.session_state["h_progress"] = h_progress


def add_files():

    uploaded_files = st.session_state["h_uploaded_files"]

    if uploaded_files is not None:
        percent_complete = 0.0
        with st.sidebar:
            st.session_state["h_progress"].progress(
                percent_complete / 100, text=f"Processing: {percent_complete:.1f}%"
            )

        for i, uploaded_file in enumerate(uploaded_files):

            temp_dir = tempfile.mkdtemp()
            tmp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(tmp_file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            st.session_state["DS"].add_document(tmp_file_path)

            percent_complete = 100.0 * (i + 1) / len(uploaded_files)
            with st.sidebar:
                st.session_state["h_progress"].progress(
                    percent_complete / 100, text=f"Processing: {percent_complete:.1f}%"
                )

            st.session_state["h_document_count"].metric(
                "Documents", value= st.session_state["DS"].collection.metadata['N_full_documents'], border = True
            )
        with st.sidebar:
            st.session_state["h_progress"].progress(100, text="Up to date")


def search(query):
    results = st.session_state["DS"].search(query)
    h_results = st.write(results)

initialize_session(device)
add_files()

tab1, tab2, tab3 = st.tabs(["search", "cluster", "review documents"])

with tab1:
    pass
    

with tab3:
    df = st.session_state['DS']._get_simplified_document_df()
    cols = st.multiselect ('Select a column', df.columns, default = ['basename', 'document' ])
    st.dataframe(df.loc[:, cols])

# if query:
#     results = DS.search(query)
#     h_results = st.write(results)


#streamlit run rag_app.py