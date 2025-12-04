
from source.database import add_files_parallel
import os
import glob


# def _producer(filenames, queue, batch_size=1000):

#     documents = []
#     metadatas = []
#     ids = []

#     for filename in filenames:

#         data = [
#             x.convert_to_vector_database_format()
#             for x in parsers.parse_document(filename)
#         ]  # id, document, metadata
#         ids += [x[0] for x in data]
#         documents += [x[1] for x in data]
#         metadatas += [x[2] for x in data]

#         if len(ids) >= batch_size:
#             # When batch size is reached, put the batch into the queue
#             queue.put((ids, documents, metadatas))
#             documents = []
#             metadatas = []
#             ids = []
#             print("added batch to queue")

#     queue.put((ids, documents, metadatas))
#     print("added last batch to queue")


# # Worker function to get items from the queue
# def _consumer(queue, device):

#     chroma_client = chromadb.PersistentClient(path="my_vectordb")
#     collection = chroma_client.get_or_create_collection(
#         name="got", embedding_function=CustomEmbeddingFunction(device=device)
#     )

#     while True:
#         # Check for items in queue, this process blocks until queue has items to process.
#         batch = queue.get()
#         if batch is None:
#             break

#         print(batch[0])

#         # Add to collection
#         collection.add(
#             ids=batch[0],
#             documents=batch[1],
#             metadatas=batch[2],
#         )  # id, document, metadata
#         print(f"{collection.count()} records in collection")


# def add_files_parallel(filenames, database_path, collection_name, device):

#     # Create a shared queue
#     queue = mp.Queue()

#     # Create producer and consumer processes.
#     producer_process = mp.Process(
#         target=_producer,
#         args=(
#             filenames,
#             queue,
#         ),
#     )
#     consumer_process = mp.Process(
#         target=_consumer,
#         args=(queue, device),
#     )

#     start_time = time.time()

#     # Start processes
#     producer_process.start()
#     consumer_process.start()

#     # Wait for producer to finish producing
#     producer_process.join()

#     # Signal consumer to stop consuming by putting None into the queue. Need 2 None's to stop 2 consumers.
#     queue.put(None)

#     # Wait for consumer to finish consuming
#     consumer_process.join()

#     print(
#         f"Done adding document in parallel: Elapsed seconds: {time.time()-start_time:.0f}"
#     )


if __name__ == "__main__":
    
    device = "cuda"  # or 'cpu'
    database_path = "my_vectordb"
    collection_name = "documents"
    glob_pattern = os.path.join(".", "data", "*")
    filenames = glob.glob(glob_pattern)

    add_files_parallel(filenames, database_path, collection_name, device)