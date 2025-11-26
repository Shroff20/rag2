import os

os.environ["OMP_NUM_THREADS"] = (
    "1"  # avoid windows memory leak in sklearn kmeans, set before Kmeans importimport parsers
)
import pypdf
import os
import datetime
import hashlib
import uuid
from pypdf import PdfReader
import json

def parse_pdf(filename, verbose=False):

    basename = os.path.basename(filename)


    modification_timestamp = os.path.getmtime(filename)
    modification_date = datetime.datetime.fromtimestamp(modification_timestamp)

    creation_timestamp = os.path.getctime(filename)
    creation_date = datetime.datetime.fromtimestamp(creation_timestamp)

    upload_date = datetime.datetime.now()

    reader = PdfReader(filename, "rb")
    metadata = dict(reader.metadata) if reader.metadata else {}
    author = metadata.get("/Author", "")
    author = author[0] if isinstance(author, list) else author

    page_lengths = []
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text() + "\n"
        page_lengths.append(len(page_text))
        full_text += page_text

    with open(filename, "rb") as file:
        bytes_data = file.read()

    hash_document = str(sha256sum(filename))
    uuid_document = str(uuid.uuid5(uuid.NAMESPACE_URL, basename+full_text))

    id = uuid_document
    document = full_text
    metadata = {
        "basename": basename,
        "fullpath": filename,
        "hash": hash_document,
        "id": uuid_document,
        "modification_date": str(modification_date),
        "creation_date": str(creation_date),
        "upload_date": str(upload_date),
        "author": str(author),
        #"bytes": bytes_data,  #TODO: get this working
        #"text": full_text,
        "page_lengths": page_lengths,
        "file_ext": ".pdf",
        "source_type": "full document"
    }

    return id, document, metadata


def sha256sum(filename):
    with open(filename, "rb", buffering=0) as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

