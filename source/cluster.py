from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

class ClusterAnalyis():

    def __init__(self, vector_database):
        
        self.vector_database = vector_database
        pass


    def calculate_pca(self, n_components = None, save = True):

        data = self.vector_database.collection.get(include=["embeddings"])
        ids = data["ids"]
        embeddings = data["embeddings"]
        print(f"embeddings matrix is {embeddings.shape}")

        pca = PCA(n_components=n_components, random_state=0)
        pca_embeddings = pca.fit_transform(embeddings)
        print(f"pca matrix is {pca_embeddings.shape}")

        metadatas = [{"pca_embedding": str(x.tolist())} for x in list(pca_embeddings)]

        if save:
            self.vector_database.update_metadata(ids, metadatas)
        self.pca = pca

        return pca_embeddings


    def perform_kmeans_clustering(self, n_clusters = 3, save = True):

        df = self.vector_database.get(include=["metadatas"], keep_cols =['id', 'pca_embedding'])
        ids = df["id"].tolist()
        embeddings = np.vstack(df['pca_embedding'])
        del df
        print(f"embeddings matrix is {embeddings.shape}")

        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(embeddings)
        print(f'performed kmeans clustering with {n_clusters} clusters')
        print(pd.Series(cluster_labels, name = 'kmeans_cluster_idx').value_counts(sort = False))
        

        metadatas = [{"kmeans_cluster_idx": int(x)} for x in list(cluster_labels)]
        self.kmeans = kmeans

        if save:
            self.vector_database.update_metadata(ids, metadatas)
    
