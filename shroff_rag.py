import chromadb
from sentence_transformers import SentenceTransformer
from chromadb import Documents, EmbeddingFunction, Embeddings
import json
import pandas as pd
from langchain.chat_models import init_chat_model
import parsers
import os

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 50)


class DataStore:

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

        print(f"created collection: {collection} with {collection.metadata['N_full_documents']} documents")

    def add_document(self, filename, chunk_size=1000, chunk_overlap=250, chunk=True):

        extension = os.path.splitext(filename)[1]

        if extension == '.pdf':
            id, document, metadata = parsers.parse_pdf(filename)
        else:
            raise(Exception('extension not allowed'))

        metadata = convert_datatypes(metadata)

        self.collection.add(
            ids=[
                id,
            ],
            documents=[
                document,
            ],
            metadatas=[
                metadata,
            ],
        )
        print(f"- added {filename}")

        if chunk:
            chunks_ids, chunk_documents, chunk_metadatas = parsers.chunk_documents(
                document, metadata, chunk_size, chunk_overlap
            )
            self.collection.add(
                ids=chunks_ids, documents=chunk_documents, metadatas=chunk_metadatas
            )
            print(f"   - added {len(chunks_ids)} chunks from {filename}")
        self._update_number_of_documents()

    def search(self, query, k=10):
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
        )
        df_results_list = _results_to_df(results)
        df_results = df_results_list[0]
        print(f'found {len(df_results)} similar documents')
        return df_results

    def query(self, query, k=10):

        if self.llm_model == None:
            self.llm_model = init_chat_model("google_genai:gemini-2.5-flash-lite")

        df_results = self.search(query=query, k=k)
        llm_query = build_rag_prompt(query, df_results)
        response = self.llm_model.invoke(llm_query)
        # print(response)
        return response

    def clear_collections(self):
        collections = self.client.list_collections()
        for collection in collections:
            r = collection.get(include = [])
            collection.delete(r['ids'])
            print(f"deleted data in {collection}")
        self._update_number_of_documents()

    def info(self):
        collections = self.client.list_collections()
        print(f"collections: {collections}")

    def _update_number_of_documents(self):
        r = self.collection.get(where={"source_type":"full document"}, include = [])
        self.collection.metadata['N_full_documents'] = len(r['ids'])

    def _get_simplified_document_df(self, max_doc_length = 100):

        results = self.collection.get(where = {'chunk_idx':-1})

        documents = results["documents"]
        truncated_documents = [s[:max_doc_length] + '... (truncated)' for s in documents]

        df_meta = pd.DataFrame.from_dict(results["metadatas"])
        df_ids = pd.DataFrame({"id": results["ids"]})
        df_documents = pd.DataFrame({"document": truncated_documents})
        df_results = pd.concat(
            [df_ids, df_meta, df_documents],
            axis=1,
        )


        drop_cols =  ['chunk_idx', 'hash', 'source_type', 'N_chunks', 'source_id', 'pca']

        df_results = df_results.drop(columns = drop_cols, errors = 'ignore')

        df_results = _reorder_cols(df_results, ['basename', 'document'])

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
        df_ids = pd.DataFrame({"id": results["ids"][iresult]})
        df_distances = pd.DataFrame({"distance": results["distances"][iresult]})
        df_documents = pd.DataFrame({"document": results["documents"][iresult]})

        df_results = pd.concat(
            [df_distances, df_ids, df_meta, df_documents],
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


def build_rag_prompt(query, df_results):
    # 1. Join your retrieved documents into a single string
    # Assuming 'documents' is a list of strings or objects with .page_content

    divider = "-------------------------------------------\n"

    context_text = ""

    for idx in df_results.index:
        context_text += divider
        context_text += "<filename>: " + df_results.loc[idx, "basename"] + "\n"
        context_text += df_results.loc[idx, "document"] + "\n"
        context_text += divider

    # 2. The Template
    template = f"""
You are a technical assistant helping answer questions based strictly on the provided documents.

Guidelines:
- Answer the question based ONLY on the context below.
- If the context does not contain the answer, say "I cannot answer this based on the provided documents."
- If you can answer, at the end of your response, append the citation like this: (source: <filename>).

<context>
{context_text}
</context>
{divider}
Question: {query}
"""
    return template
