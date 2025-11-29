# streamlit run rag_app.py

import streamlit as st
import pandas as pd
import importlib
import shroff_rag
importlib.reload(shroff_rag)
import tempfile
import os
import plotly.express as px

device = "cuda"
default_database_dir = "./database_data"
default_api_key = "AIzaSyCp2r7wKvf_aNLc1gxzJTdhLrVAwaS-0WM"
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


def update_connection():
    DS = shroff_rag.DataStore(device, st.session_state['database_path'])
    DS.create_collection()
    st.session_state["DS"] = DS

def initalize():
    config = {
        "messages": [],
        "documents_to_process": [],
        "database_path" : default_database_dir,
        "api_key" : default_api_key,
        "status_pca_valid" : False
    }
    set_session_state(config)

    if "DS" not in st.session_state:
        update_connection()


def process_files(files):
    N_files = len(files)
    st.session_state['status_pca_valid'] = False # added documents, so need to recompute pca

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


def set_api_key(env_var = "GOOGLE_API_KEY"):
    os.environ[env_var] = st.session_state['api_key']
    print(f"set {env_var}={st.session_state['api_key']}")


initalize()


tab1, tab2, tab3, tab4 = st.tabs(["query", "search", "cluster", "list documents"])

with st.sidebar:
    h_document_count = st.metric(
        "Loaded documents",
        st.session_state["DS"].collection.metadata["N_full_documents"],
        border=True,
    )
    st.session_state["h_document_count"] = h_document_count


    with st.form("my-form", clear_on_submit=True):
        st.header("Add files")
        h_uploaded_files = st.file_uploader(
            "Upload documents",
            accept_multiple_files=True,
            type=allowed_filetpyes,
            label_visibility="hidden",
            #key="h_uploaded_files",
        )
        submitted = st.form_submit_button("▶️ process")
        h_progress = st.progress(0 / 100, text=f"Nothing to process")
        st.session_state["h_progress"] = h_progress

        
        if submitted:
            st.session_state['documents_to_process'].append(h_uploaded_files)
            process_files(files=h_uploaded_files)

    with st.container( border = True):
        st.header("Database")
        st.text_input(label = 'database path' , key = 'database_path', on_change=update_connection())
        st.button(label="⚠️ delete all documents", on_click=delete_all_collections)
    
    with st.container( border = True):
        st.header("Language model")  
        st.text_input(label = 'API key', key = 'api_key', on_change=set_api_key, help = 'enter your API key for the language model')


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

with tab3:
    n_clusters = st.slider(label = "# clusters", min_value=1, max_value = 10, value = 3)
    plot_3d = st.toggle('plot 3d', value = False)
    run_clustering = st.button('▶️ run clustering')


    if run_clustering:
        if st.session_state['status_pca_valid'] == False:
            st.session_state['DS'].compute_pca()
            st.session_state['status_pca_valid'] = True

        labels, inertia, df_pca = st.session_state['DS'].run_kmeans(n_clusters = n_clusters)
        df_meta = st.session_state['DS']._get_simplified_document_df().set_index('id')
        df_pca = df_pca.join(df_meta)
        print(f'ran clustering with n_clusters = {n_clusters}')
        df_pca['label'] = [f'cluster {label+1}' for label in labels]

        if plot_3d:
            fig = px.scatter_3d(data_frame=df_pca, x=df_pca.columns[0], y=df_pca.columns[1], z = df_pca.columns[3], color='label', hover_data=['basename', df_pca.index])
            st.plotly_chart(fig)
        else:
            fig = px.scatter(data_frame=df_pca, x=df_pca.columns[0], y=df_pca.columns[1], color='label', hover_data=['basename', df_pca.index])
            st.plotly_chart(fig)



with tab4:
    df = st.session_state["DS"]._get_simplified_document_df()

    col_options = ['basename', 'document', 'id', 'creation_date', 'fullpath', 'author', 'page_lengths', 'file_ext', 'modification_date', 'upload_date']
    selcted_cols = st.multiselect("Select a column", col_options, default=["basename", "document"], key = 'h_multiselect')

    print(f"making dataframe with {selcted_cols}")

    allowed_cols = [col for col in selcted_cols if col in df.columns]
    st.dataframe(df[allowed_cols])

    st.write('Note: "document" column does not display the full text')