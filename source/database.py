import os
import sys
sys.path.append("..")
os.environ["OMP_NUM_THREADS"] = (
    "1"  # avoid windows memory leak in sklearn kmeans, set before Kmeans importimport parsers
)
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb import Documents, EmbeddingFunction, Embeddings
import pandas as pd
from langchain.chat_models import init_chat_model
import source.parsers as parsers
import pandas as pd
import ast
import numpy as np
import source.rag as rag
from functools import reduce
import time
import multiprocessing as mp

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

        #add_files_parallel(filenames, database_path=self.database_folder, collection_name=self.collection.name, device = self.device)

        all_data = []
        for filename in filenames:
            # check if document already has been added, skip if so
            id = parsers.generate_document_uuid_from_path(filename)
            r = self.collection.get(ids = id, include = [])
            id_exists = len(r['ids'])>0
            if id_exists:
                print(f"- [{filename}] already exists in database, skipping")
            else:
                data = parsers.parse_document(filename)
                all_data += data

        if len(all_data) > 0: # this code will fail if there are no new documents to add

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
        print(results)
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
        df_results = _results_to_df(results)
        df_results = df_results.sort_values("distance")
        print(f"found {len(df_results)} similar documents")
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



    def get(self, processing_limit = 100, include = ["documents", "metadatas", "embeddings"],  get_kwargs = {}, keep_cols = None, document_length_limit = None):

        offset = 0
        total_items = self.collection.count()

        df_results = []
        while offset < total_items:
            chunk_results = self.collection.get(
                offset=offset,
                limit=processing_limit,
                include=include, **get_kwargs  # Specify what to retrieve
            )
            if len(chunk_results['ids']) >0:
                df_results_chunk = _results_to_df(chunk_results, keep_cols= keep_cols, document_length_limit= document_length_limit)
                df_results.append(df_results_chunk)
            offset += processing_limit
        if len(df_results) > 0:
             df_results = pd.concat(df_results, axis = 0).copy() # solve fragmantation issue
        else:
            df_results = pd.DataFrame()
            
        df_results = df_results.reset_index(drop=True)

        return df_results
    
    def update_metadata(self, ids, metadatas, processing_limit = None):
    
        if processing_limit is None:
            processing_limit = self.client.get_max_batch_size() # cannot excede max batch size when accessing database

        for i in range(0, len(ids), processing_limit):
            self.collection.update(
                ids=ids[i : i + processing_limit],
                metadatas=metadatas[i : i + processing_limit],
        )
        updated_keys = reduce(lambda x, y: x.union(set(y.keys())), metadatas, set())    

        print(f"updated metadata for {len(ids)} records for the following keys: {updated_keys}")
    
    
    def _update_number_of_documents(self):
        if self.collection is not None:
            r = self.collection.get(where={"source_type": "full document"}, include=[])
            N_documents = len(r["ids"])
            self.collection.metadata["N_full_documents"] = N_documents
            num_records = self.collection.count()
            self.collection.metadata["N_records"] = num_records



class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self, device):
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2", model_kwargs={"torch_dtype": "float32"}, device=device
        )
        self._name = "all-MiniLM-L6-v2"
    
    def name(self):
        return self._name

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
        d['id'] = results['ids'][slicer]
    if results.get('documents') is not None:  
        d['document'] = results['documents'][slicer]
    if results.get('embeddings') is not None:
        d['embedding'] = [np.array(row, dtype = np.float32) for row in results['embeddings'][slicer]]
    df = pd.DataFrame.from_dict(d)

    if results.get('metadatas') is not None:
        df_metadata = pd.DataFrame(results['metadatas'][slicer], index = range(len(df))).drop(columns = 'id', errors = 'ignore')
        df = pd.concat([df, df_metadata], axis = 1).reset_index(drop = True)


    def string_to_vec(df, dtype = np.float16): #TODO fix this
        df = df.replace(np.nan, '')
        df = df.apply(ast.literal_eval)
        df = df.apply(lambda x: np.array(x, dtype = np.float16))
        return df



    datetime_format =  r"%Y-%m-%d %H:%M:%S"
    type_converter = {}
    type_converter['creation_date'] = lambda x:  pd.to_datetime(x, format = datetime_format)
    type_converter['upload_date'] = lambda x:  pd.to_datetime(x, format = datetime_format)
    type_converter['modification_date'] = lambda x:  pd.to_datetime(x, format = datetime_format)
    type_converter['page_lengths'] = lambda x:  string_to_vec(x, dtype = np.int32)
    type_converter['document'] = lambda x:  x.str[:document_length_limit]
    type_converter['pca_embedding'] = string_to_vec

    for col in df.columns:
        if col in type_converter.keys():
            df[col] = type_converter[col](df[col])

    if keep_cols is not None:
        cols = [col for col in df.columns if col in keep_cols]
        df = df[cols]

    df = df.reset_index(drop=True)

    return df


def _reorder_cols(df, cols_to_move_to_front):
    cols_to_move_to_front = [col for col in cols_to_move_to_front if col in df.columns]
    df = df[
        cols_to_move_to_front
        + [col for col in df.columns if col not in cols_to_move_to_front]
    ]
    return df


def _producer(filenames, queue, batch_size=1000):

    documents = []
    metadatas = []
    ids = []

    for filename in filenames:

        data = [
            x.convert_to_vector_database_format()
            for x in parsers.parse_document(filename)
        ]  # id, document, metadata
        ids += [x[0] for x in data]
        documents += [x[1] for x in data]
        metadatas += [x[2] for x in data]

        if len(ids) >= batch_size:
            # When batch size is reached, put the batch into the queue
            queue.put((ids, documents, metadatas))
            documents = []
            metadatas = []
            ids = []
            print("added batch to queue")

    queue.put((ids, documents, metadatas))
    print("added last batch to queue")


# Worker function to get items from the queue
def _consumer(queue, database_path, collection_name, device):

    chroma_client = chromadb.PersistentClient(path=database_path)
    collection = chroma_client.get_collection(
        name=collection_name, embedding_function=CustomEmbeddingFunction(device=device)
    )

    while True:
        # Check for items in queue, this process blocks until queue has items to process.
        batch = queue.get()
        if batch is None:
            break

        # Add to collection
        collection.add(
            ids=batch[0],
            documents=batch[1],
            metadatas=batch[2],
        )  # id, document, metadata
        print(f"{collection.count()} records in collection")


def add_files_parallel(filenames, database_path, collection_name, device):

    # Create a shared queue
    queue = mp.Queue()

    # Create producer and consumer processes.
    producer_process = mp.Process(
        target=_producer,
        args=(
            filenames,
            queue,
        ),
    )
    consumer_process = mp.Process(
        target=_consumer,
        args=(queue, database_path, collection_name, device),
    )

    start_time = time.time()

    # Start processes
    producer_process.start()
    consumer_process.start()

    # Wait for producer to finish producing
    producer_process.join()

    # Signal consumer to stop consuming by putting None into the queue. Need 2 None's to stop 2 consumers.
    queue.put(None)

    # Wait for consumer to finish consuming
    consumer_process.join()

    print(
        f"Done adding document in parallel: Elapsed seconds: {time.time()-start_time:.0f}"
    )