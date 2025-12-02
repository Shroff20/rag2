import sys
sys.path.append("..")
import streamlit as st
import streamlit_functions as sf
import tempfile
import os
import source.database as database
import importlib
importlib.reload(database)
importlib.reload(sf)


st.set_page_config(
    page_title="search",
    page_icon="📄",
)


st.title("Manage Documents")

sf.initialize_app()
sf.make_sidebar()


def process_files(files):
    N_files = len(files)
    st.session_state["status_pca_valid"] = (
        False  # added documents, so need to recompute pca
    )

    starting_document_count = st.session_state["VD"].collection.metadata[
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

            try:
                st.session_state["VD"].add_documents(tmp_file_path)
            except Exception as e:
                print(f"could not process {file}", e)

            percent_complete = 100.0 * (i + 1) / N_files
            st.session_state["h_progress"].progress(
                percent_complete / 100, text=f"Processing: {percent_complete:.1f}%"
            )
            current_document_count = st.session_state["VD"].collection.metadata[
                "N_full_documents"
            ]
            print("count", current_document_count)

            sf.update_document_count()

def delete_all_collections():
    st.session_state["VD"].clear_collections()
    sf.update_document_count()



with st.form("add-form", clear_on_submit=True):
    st.markdown("### Add files")
    h_uploaded_files = st.file_uploader(
        "Upload documents",
        accept_multiple_files=True,
        type=st.session_state["allowed_filetpyes"],
        label_visibility="hidden",
        # key="h_uploaded_files",
    )
    submitted = st.form_submit_button("▶️ process")
    h_progress = st.progress(0 / 100, text=f"Nothing to process")
    st.session_state["h_progress"] = h_progress

    if submitted:
        st.session_state["documents_to_process"].append(h_uploaded_files)
        process_files(files=h_uploaded_files)


with st.container( border = True):
    st.markdown("### Delete files")
    st.button(label="⚠️ delete all documents", on_click=delete_all_collections)

with st.container( border = True):
    st.markdown("### List files")
    if 'df_documents' not in st.session_state:
        sf.update_df_documents()

    col_options = ['basename', 'document', 'id', 'creation_date', 'fullpath', 'author', 'page_lengths', 'file_ext', 'modification_date', 'upload_date']
    selcted_cols = st.multiselect("Select columns", col_options, default=["basename", "document"], key = 'h_multiselect')

    refresh = st.button(label = '🔄 refresh', on_click=sf.update_df_documents)

    if refresh:
        allowed_cols = [col for col in selcted_cols if col in st.session_state['df_documents']]
        st.dataframe(st.session_state['df_documents'].loc[:, allowed_cols])