import sys

sys.path.append("..")
import streamlit as st
import streamlit_functions as sf
import source.database as database
import pandas as pd
import importlib
importlib.reload(database)


st.set_page_config(
    page_title="search",
    page_icon="🔍",
)

st.title("Search Page")
sf.initialize_app()
sf.make_sidebar()

with st.sidebar.container(border=True):  # TODO: make this persistent across pages
    st.markdown("## Settings")
    N_return_docs = st.number_input(
        label="\\# return documents",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        help="number of most similar documents to return for each query",
    )


def perform_search():
    query = st.session_state["search_query"]
    df_results = st.session_state["VD"].search(
        query, where={"source_type": "full document"}, k=N_return_docs
    )
    df_results = database._reorder_cols(
        df_results, ["distance", "basename", "document"]
    )
    print(st.session_state)
    st.session_state["df_search_results"] = df_results


tab1, tab2 = st.tabs(["By document", "By text query"])

with tab1:
    if (
        "df_documents" not in st.session_state
        or not st.session_state["status_df_documents_valid"]
    ):
        sf.update_df_documents()

    st.text("Select a document to search similar documents:")

    event = st.dataframe(
        st.session_state["df_documents"],
        on_select="rerun",
        selection_mode="single-row",
        hide_index=False,
    )

    if event is not None and len(event["selection"]["rows"]) > 0:
        selected_row_idx = event["selection"]["rows"][0]
        selected_doc = st.session_state["df_documents"].loc[
            st.session_state["df_documents"].index[selected_row_idx], :
        ]
        st.text(f"{N_return_docs} similar documents to {selected_doc.basename}:")

        df_results = st.session_state["VD"].search_by_id(
            id=selected_doc.id, where={"source_type": "full document"}, k=N_return_docs
        )
        df_results = database._reorder_cols(
            df_results, ["distance", "basename", "document"]
        )
        st.dataframe(df_results)

    # options = st.session_state['df_documents']['basename'].tolist()
    # selected_fruit = st.selectbox("Select a fruit (type to filter)", options)

with tab2:
    st.text_input(
        label="Search query",
        key="search_query",
        on_change=perform_search,
        value=st.session_state.get("search_query", ""),
    )
    st.text(f"{N_return_docs} most similar documents:")
    st.dataframe(st.session_state["df_search_results"])
