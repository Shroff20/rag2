import chromadb
from sentence_transformers import SentenceTransformer
from chromadb import Documents, EmbeddingFunction, Embeddings
import json

class DataStore:

    def __init__(self, device, database_folder):

        self.device = device
        self.database_folder = database_folder
        self.collection = None
        self.client = chromadb.PersistentClient(path=database_folder)


    def create_collection(self, collection_name="documents"):

        collection = self.client.create_collection(
            name=collection_name,
            embedding_function=CustomEmbeddingFunction(
                self.device
            ), 
            get_or_create=True,
            metadata={"hnsw:space": "cosine"} 
        )
        self.collection = collection

    def add(self, ids, documents, metadatas):

        metadatas = [convert_datatypes(metadata) for metadata in metadatas]
        print(metadatas)

        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def search(self, query, k=10):
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
        )
        return results
    
    def clear_collections(self):
        collections = self.client.list_collections()
        for collection in collections:
            self.client.delete_collection(collection.name)

    def info(self):
        collections = self.client.list_collections()


class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self, device):
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2", model_kwargs={"torch_dtype": "float16"}, device=device
        )
        self.name = "all-MiniLM-L6-v2"

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(input, normalize_embeddings = True)
        return embeddings.tolist()


def convert_datatypes(data: dict):

    for key, value in data.items():
        if type(value) == list:
            data[key] = json.dumps(data[key])
            
    return data
