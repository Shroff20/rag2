from sklearn.decomposition import PCA
from sklearn.cluster import KMeans



class ClusterAnalyis():

    def __init__(self, vector_database):
        
        self.vector_database = vector_database
        pass


    def calculate_pca(self, n_components = None):

        data = self.vector_database.collection.get(include=["embeddings"])
        ids = data["ids"]
        embeddings = data["embeddings"]
        print(f"embeddings matrix is {embeddings.shape}")

        pca = PCA(n_components=n_components, random_state=0)
        pca_embeddings = pca.fit_transform(embeddings)
        print(f"pca matrix is {pca_embeddings.shape}")

        metadatas = [{"pca_embedding": str(x.tolist())} for x in list(pca_embeddings)]

        self.vector_database.update_metadata(ids, metadatas)

    def perform_kmeans_clustering(self, n_clusters = 5):

        data = self.vector_database.collection.get(include=["embeddings"])
        ids = data["ids"]
        embeddings = data["embeddings"]
        print(f"embeddings matrix is {embeddings.shape}")

        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(embeddings)
        print(f"cluster labels length is {len(cluster_labels)}")

        metadatas = [{"kmeans_cluster_idx": int(x)} for x in list(cluster_labels)]

        self.vector_database.update_metadata(ids, metadatas)
    
