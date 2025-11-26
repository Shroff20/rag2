import chromadb
from sentence_transformers import SentenceTransformer
from chromadb import Documents, EmbeddingFunction, Embeddings
import json
import pandas as pd


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 50)


class DataStore:

    def __init__(self, device, database_folder):

        self.device = device
        self.database_folder = database_folder
        self.collection = None
        self.client = chromadb.PersistentClient(path=database_folder)

    def create_collection(self, collection_name="documents"):

        collection = self.client.create_collection(
            name=collection_name,
            embedding_function=CustomEmbeddingFunction(self.device),
            get_or_create=True,
            metadata={"hnsw:space": "cosine"},
        )
        self.collection = collection
        print(f"created collection: {collection}")

    def add(self, ids, documents, metadatas):

        metadatas = [convert_datatypes(metadata) for metadata in metadatas]
        print(metadatas)

        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def search(self, query, k=10):
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
        )
        df_results_list = _results_to_df(results)
        return df_results_list

    def clear_collections(self):
        collections = self.client.list_collections()
        for collection in collections:
            self.client.delete_collection(collection.name)
            print(f"deleted {collection}")

    def info(self):
        collections = self.client.list_collections()
        print(f"collections: {collections}")


class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self, device):
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2", model_kwargs={"torch_dtype": "float16"}, device=device
        )
        self.name = "all-MiniLM-L6-v2"

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(input, normalize_embeddings=True)
        return embeddings.tolist()


def convert_datatypes(data: dict):

    for key, value in data.items():
        if type(value) == list:
            data[key] = json.dumps(data[key])

    return data


def _results_to_df(results):

    N_results = len(results["ids"])
    df_results_list = []

    for iresult in range(N_results):
        df_meta = pd.DataFrame.from_dict(results["metadatas"][iresult])
        df_meta = df_meta.drop("id", axis=1)
        df_ids = pd.DataFrame({"id": results["ids"][iresult]})
        df_distances = pd.DataFrame({"distance": results["distances"][iresult]})

        df_results = pd.concat(
            [
                df_distances,
                df_ids,
                df_meta,
            ],
            axis=1,
        )

        cols_to_move_to_front = ["distance", "basename"]
        df_results = _reorder_cols(df_results, cols_to_move_to_front)

        df_results_list.append(df_results)

    

    return df_results_list


def _reorder_cols(df, cols_to_move_to_front):
    cols_to_move_to_front = [col for col in cols_to_move_to_front if col in df.columns]
    df = df[
            cols_to_move_to_front
            + [col for col in df.columns if col not in cols_to_move_to_front]
        ]
    return df