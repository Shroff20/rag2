import sys
sys.path.append("..")
import streamlit as st
import streamlit_functions as sf
import source.database as database
import importlib
import cluster
import plotly.express as px
import pandas as pd

importlib.reload(database)
importlib.reload(cluster)
importlib.reload(sf)

st.set_page_config(
    page_title="Cluster",
    page_icon="✨",
)
st.title("Cluster")
sf.initialize_app()
sf.make_sidebar()


with st.sidebar.container( border = True):
    st.markdown('## Settings')
    n_clusters = st.slider(label = "# clusters", min_value=1, max_value = 10, value = 3)
    plot_3d = st.toggle('plot 3d', value = False)
    autorun = st.toggle('autorun', value = True)



run_clustering = st.button('▶️ run clustering')

if run_clustering or autorun:

    CA = cluster.ClusterAnalyis(st.session_state["VD"])

    if not st.session_state['status_pca_valid']:
        CA.calculate_pca()
        st.session_state['status_pca_valid'] = True

    CA.perform_kmeans_clustering(n_clusters = n_clusters)
    
    df = st.session_state['VD'].get(keep_cols = ['id', 'basename', 'kmeans_cluster_idx', 'pca_embedding'])
    for i in range(3):
        df[f'pca_{i}'] = df['pca_embedding'].apply(lambda x: x[i])
    df['label'] = [f'cluster {label}' for label in  df['kmeans_cluster_idx']]

    if plot_3d:
        fig = px.scatter_3d(data_frame=df, x='pca_0', y='pca_1', z = 'pca_2', color='label', hover_data=['basename'])
        fig.update_layout(scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))) # Adjust eye coordinates
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0)) # Set margins to zero
        st.plotly_chart(fig)
    else:
        fig = px.scatter(data_frame=df, x='pca_0', y='pca_1', color='label', hover_data=['basename'])
        st.plotly_chart(fig)

    df_counts = pd.Series(df['label'], name = 'cluster').value_counts(sort = False).sort_index()
    st.dataframe(df_counts, width='content')

    #st.dataframe(df)

    
