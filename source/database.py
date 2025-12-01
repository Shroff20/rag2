import os

os.environ["OMP_NUM_THREADS"] = (
    "1"  # avoid windows memory leak in sklearn kmeans, set before Kmeans importimport parsers
)
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb import Documents, EmbeddingFunction, Embeddings
import json
import pandas as pd
from langchain.chat_models import init_chat_model
import source.parsers as parsers
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import json
import ast
import numpy as np
import source.rag as rag

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 50)


class VectorDatabase:

    def __init__(self, device, database_folder):

        self.device = device
        self.database_folder = database_folder
        self.collection = None
        self.client = chromadb.PersistentClient(path=database_folder)
        self.llm_model = None

    def create_collection(self, collection_name="documents"):

        collection = self.client.create_collection(
            name=collection_name,
            embedding_function=CustomEmbeddingFunction(self.device),
            get_or_create=True,
            metadata={"hnsw:space": "cosine"},
        )
        self.collection = collection
        self._update_number_of_documents()

        print(
            f"created collection: {collection} with {collection.metadata['N_full_documents']} documents"
        )

    def add_documents(self, filenames, chunk_size=1000, chunk_overlap=250, chunk=True):

        if type(filenames) is not list:
            filenames = [filenames,]

        all_data = []
        for filename in filenames:
            data = parsers.parse_document(filename)
            all_data += data

        all_data = [x.convert_to_vector_database_format() for x in all_data]  #(id, document, metadata)
        all_data =  list(map(list, zip(*all_data)))

        ids = all_data[0]
        documents = all_data[1]
        metadatas = all_data[2]

        self.collection.upsert(
            ids=ids, documents = documents, metadatas = metadatas)
        
        N_documents = len(filenames)
        N_chunks = len(ids) - N_documents

        print(f"- added {N_documents} documents, {N_chunks} chunks")

        self._update_number_of_documents()

    def search(self, query, k=10, **kwargs):
        results = self.collection.query(query_texts=[query], n_results=k, **kwargs)
        df_results = _results_to_df(results)
        print(f"found {len(df_results)} similar documents")
        return df_results

    def search_by_id(self, id, k=10, **kwargs):
        r = self.collection.get(id, include=["embeddings"])
        embedding = r["embeddings"]
        results = self.collection.query(
            query_embeddings=embedding,
            include=["distances", "metadatas", "documents"],
            **kwargs,
        )
        df_results = _results_to_df(results[0])
        df_results = df_results.sort_values("distance")
        return df_results

    def query(self, query, k=10):

        if self.llm_model == None:
            self.llm_model = init_chat_model("google_genai:gemini-2.5-flash-lite")

        df_results = self.search(query=query, k=k)
        llm_query = rag.build_rag_prompt(query, df_results)
        response = self.llm_model.invoke(llm_query)
        # print(response)
        return response

    def clear_collections(self):
        collections = self.client.list_collections()
        for collection in collections:
            r = collection.get(include=[])
            if len(r["ids"]) > 0:  # will result in error if there is an empty list
                collection.delete(r["ids"])
            print(f"deleted data in {collection}")
        self._update_number_of_documents()

    def info(self):
        collections = self.client.list_collections()
        print(f"collections: {collections}")
        for collection in collections:
            num_records = collection.count()
            print(f'{collection} has {num_records} records')
                  
    # def compute_pca(self, n_components=None):

    #     data = self.collection.get(include=["embeddings"])

    #     ids = data["ids"]
    #     embeddings = data["embeddings"]
    #     print(f"embeddings matrix is {embeddings.shape}")
    #     pca = PCA(n_components=n_components)
    #     pca_embeddings = pca.fit_transform(embeddings)
    #     print(f"pca matrix is {pca_embeddings.shape}")

    #     # columns = [f'pca_{x}' for x in range(pca_embeddings.shape[1])]
    #     # df_pca  = pd.DataFrame(pca_embeddings, index = ids, columns = columns)
    #     # df_pca

    #     metadatas = [{"pca": json.dumps(x.tolist())} for x in list(pca_embeddings)]

    #     max_batch_size = (
    #         self.client.get_max_batch_size()
    #     )  # cannot excede max batch size when accessing database
    #     for i in range(0, len(ids), max_batch_size):
    #         self.collection.update(
    #             ids=ids[i : i + max_batch_size],
    #             metadatas=metadatas[i : i + max_batch_size],
    #         )
    #     self.pca = pca

    #     print("saved pca data to vector database")

    # def run_kmeans(self, n_clusters):
    #     pca_matrix = self._get_pca_matrix()
    #     kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(
    #         pca_matrix
    #     )
    #     labels = kmeans.labels_
    #     inertia = kmeans.inertia_
    #     self.KMeans = KMeans
    #     return labels, inertia, pca_matrix

    # def _get_pca_matrix(self, where={"source_type": "full document"}):
    #     data = self.collection.get(include=[], where=where)
    #     ids = data["ids"]

    #     pca_matrix = []

    #     for id in ids:
    #         pca = self.collection.get(ids=id).get("metadatas")[0]["pca"]
    #         pca = np.array(ast.literal_eval(pca), dtype=np.float16)
    #         pca_matrix.append(pca)

    #     pca_matrix = np.vstack(pca_matrix)
    #     df_pca = pd.DataFrame(
    #         pca_matrix,
    #         index=ids,
    #         columns=[f"pca_{x}" for x in range(pca_matrix.shape[1])],
    #     )

    #     return df_pca

    def _update_number_of_documents(self):
        if self.collection is not None:
            r = self.collection.get(where={"source_type": "full document"}, include=[])
            N_documents = len(r["ids"])
            self.collection.metadata["N_full_documents"] = N_documents
            num_records = self.collection.count()
            self.collection.metadata["N_records"] = num_records


    def _get_simplified_document_df(self, max_doc_length=100):

        results = self.collection.get(where={"chunk_idx": -1})

        documents = results["documents"]
        truncated_documents = [
            s[:max_doc_length] + "... (truncated)" for s in documents
        ]

        df_meta = pd.DataFrame.from_dict(results["metadatas"])
        df_ids = pd.DataFrame({"id": results["ids"]})
        df_documents = pd.DataFrame({"document": truncated_documents})
        df_results = pd.concat(
            [df_ids, df_meta, df_documents],
            axis=1,
        )

        drop_cols = ["chunk_idx", "hash", "source_type", "N_chunks", "source_id", "pca"]

        df_results = df_results.drop(columns=drop_cols, errors="ignore")

        df_results = _reorder_cols(df_results, ["basename", "document"])

        return df_results


class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self, device):
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2", model_kwargs={"torch_dtype": "float16"}, device=device
        )
        self.name = "all-MiniLM-L6-v2"

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(input, normalize_embeddings=True)
        return embeddings.tolist()




def _results_to_df(results, document_length_limit = None, idx = 0, keep_cols = None):
    
    if  type(results['ids'][0]) == list:  # we have a list of results
        slicer = idx
    else:
        slicer = slice(None)

    d = {}
    if results.get('distances') is not None:
        d['distance'] = results['distances'][slicer]
    if results.get('ids') is not None:
        d['id'] = results['ids'][idx][slicer]
    if results.get('documents') is not None:  
        d['document'] = results['documents'][slicer]
    if results.get('embeddings') is not None:
        d['embedding'] = [row for row in results['embeddings'][slicer]]
    df = pd.DataFrame.from_dict(d)

    if results.get('metadatas') is not None:
        df_metadata = pd.DataFrame(results['metadatas'][slicer], index = range(len(df))).drop(columns = 'id')
        df = pd.concat([df, df_metadata], axis = 1).reset_index(drop = True)

    datetime_format =  r"%Y-%m-%d %H:%M:%S.%f"
    type_converter = {}
    type_converter['creation_date'] = lambda x:  pd.to_datetime(x, format = datetime_format)
    type_converter['upload_date'] = lambda x:  pd.to_datetime(x, format = datetime_format)
    type_converter['modification_date'] = lambda x:  pd.to_datetime(x, format = datetime_format)
    type_converter['page_lengths'] = lambda x:  x.apply(ast.literal_eval)
    type_converter['document'] = lambda x:  x.str[:document_length_limit]

    for col in df.columns:
        if col in type_converter.keys():
            df[col] = type_converter[col](df[col])

    if keep_cols is not None:
        cols = [col for col in df.columns if col in keep_cols]
        df = df[cols]

    return df


def _reorder_cols(df, cols_to_move_to_front):
    cols_to_move_to_front = [col for col in cols_to_move_to_front if col in df.columns]
    df = df[
        cols_to_move_to_front
        + [col for col in df.columns if col not in cols_to_move_to_front]
    ]
    return df

