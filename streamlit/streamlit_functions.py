import sys

sys.path.append("..")
import streamlit as st
import source.database as database


def make_sidebar():
    with st.sidebar:
        h_document_count = st.metric(
            "Loaded documents",
            st.session_state["VD"].collection.metadata["N_full_documents"],
            border=True,
        )
        st.session_state["h_document_count"] = h_document_count

        st.markdown("DocHunt v 0.1.0 @ S. Shroff")  # App version or other info


def update_document_count():
    N_documents = st.session_state["VD"].collection.metadata["N_full_documents"]
    st.session_state["h_document_count"].metric(
        "Loaded documents", N_documents, border=True
    )


def update_df_documents():

    df_documents =  st.session_state["VD"].get(
        include=["metadatas", "documents"],
        document_length_limit=100,
        get_kwargs={"where": {"source_type": "full document"}},
    )
    df_documents = database._reorder_cols(df_documents, ["basename", "id", "document_type", "document"])

    st.session_state["df_documents"] = df_documents


def _set_session_state(datadict: dict):
    for key, value in datadict.items():
        if key not in st.session_state:
            print(f"setting {key} = {value}")
            st.session_state[key] = value


def _get_defaults():
    d = {}
    d["device"] = "cuda"
    d["database_path"] = "D:/database"
    d["api_key"] = "AIzaSyCV0Otml_ldT7JPtHy_WhR8TpN3T-apyFg"
    d["allowed_filetpyes"] = [".pdf", ".txt", ".csv"]
    d["documents_to_process"] = []
    return d


def _update_connection():
    VD = database.VectorDatabase(
        st.session_state["device"], st.session_state["database_path"]
    )
    VD.create_collection()
    st.session_state["VD"] = VD


def initialize_app():
    d = _get_defaults()
    _set_session_state(d)
    if "VD" not in st.session_state:
        _update_connection()
