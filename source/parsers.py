from dataclasses import dataclass, field, asdict
import os
import datetime
import hashlib
import uuid
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import get_origin, Union, get_args
import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
import copy


def generate_document_uuid(basename, full_text, chunk_idx):
    uuid_document = str(
        uuid.uuid5(uuid.NAMESPACE_URL, basename + full_text + str(chunk_idx))
    )
    return uuid_document


def sha256sum(filename):
    print(filename)
    with open(filename, "rb", buffering=0) as f:
        return str(hashlib.file_digest(f, "sha256").hexdigest())


@dataclass
class ParserOuput:

    fullpath: str
    document: str

    author: str = ""
    page_lengths: tuple[int] = ()

    source_type: str = "full document"
    pca: tuple[float] = ()
    tfidf: tuple[float] = ()
    chunk_idx: int = -1  # use -1 for full document
    N_chunks: int = 0  # total number of chunks

    file_ext: str = field(default=None)
    basename: str = field(default=None)
    id: str = field(default=None)
    file_hash: str = field(default=None)
    title: str = field(default=None)
    source_id: str = field(default=None)
    upload_date: datetime.datetime = field(default=None)
    modification_date: datetime.datetime = field(default=None)
    creation_date: datetime.datetime = field(default=None)

    def set_field_if_None(self, fieldname, value):
        if getattr(self, fieldname) is None:
            setattr(self, fieldname, value)

    def __post_init__(self):
        self.set_field_if_None("basename", os.path.basename(self.fullpath))
        self.set_field_if_None(
            "id", generate_document_uuid(self.basename, self.document, self.chunk_idx)
        )
        self.set_field_if_None("file_hash", sha256sum(self.fullpath))
        self.set_field_if_None("title", self.basename)
        self.set_field_if_None("source_id", self.id)
        self.set_field_if_None("upload_date", datetime.datetime.now())
        self.set_field_if_None(
            "modification_date",
            datetime.datetime.fromtimestamp(os.path.getmtime(self.fullpath)),
        )
        self.set_field_if_None(
            "creation_date",
            datetime.datetime.fromtimestamp(os.path.getctime(self.fullpath)),
        )
        self.set_field_if_None("file_ext", os.path.splitext(self.fullpath)[1])

        self.check_types()

    def check_types(self):
        for name, field_type in self.__annotations__.items():
            val = self.__dict__.get(name)

            origin = get_origin(field_type)

            # CASE 1: Handle Unions (e.g., "datetime | None" or "Optional[int]")
            # We must check if the value is ANY of the allowed types in the Union.
            # We check for both 'typing.Union' and Python 3.10+ 'types.UnionType' (| syntax).
            if origin is Union or (
                hasattr(types, "UnionType") and origin is types.UnionType
            ):
                actual_check_type = get_args(field_type)

            # CASE 2: Handle Generics (e.g., List[int])
            # We strip the arguments and check against the base class (e.g., list).
            elif origin is not None:
                actual_check_type = origin

            # CASE 3: Simple Types (e.g., int, str, datetime)
            else:
                actual_check_type = field_type

            # Perform the check
            if not isinstance(val, actual_check_type):
                current_type = type(val)
                raise TypeError(
                    f"The field `{name}` was assigned by `{current_type}` instead of `{field_type}`"
                )

    def convert_to_vector_database_format(self):

        metadata = asdict(self)

        allowed_types = [str, int, float]

        for key, value in metadata.items():
            if type(metadata[key]) not in allowed_types:
                metadata[key] = str(
                    value
                )  # this seems to convert datetime well (instead of json.dumps)

        id = metadata["id"]
        document = metadata["document"]
        metadata.pop("document")  # keeping id in metadata

        return id, document, metadata


def get_chunks_from_document(parsed_document : ParserOuput, chunk_size=2000, chunk_overlap=250):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    chunk_documents = text_splitter.split_text(parsed_document.document)
    N_chunks = len(chunk_documents)
    parsed_chunks = []

    for i in range(N_chunks):
        parsed_chunk = copy.deepcopy(parsed_document)
        parsed_chunk.document = chunk_documents[i]
        parsed_chunk.chunk_idx = i
        parsed_chunk.N_chunks = N_chunks
        parsed_chunk.id = generate_document_uuid(parsed_chunk.basename,parsed_chunk.document, parsed_chunk.chunk_idx)
        parsed_chunk.source_type = 'chunked document'

        parsed_chunks.append(parsed_chunk)

    return parsed_chunks


def parse_document(filename, chunk_document = True, chunk_size=2000, chunk_overlap=250):

    ext =  os.path.splitext(filename)[1]

    #TODO: update this to a map or switch case
    if ext == '.pdf':
        D = parse_pdf(filename)
    elif ext == '.txt':
        D = parse_txt(filename)
    elif ext == 'csv':
        D = parse_csv(filename)
    else:
        raise(Exception(f'do not know how to parse {ext}'))
    
    if chunk_document:
        chunks = get_chunks_from_document(D, chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    else:
        chunks = []

    all_data = [D,] + chunks

    return all_data



def parse_pdf(filename):

    reader = PdfReader(filename, "rb")
    page_lengths = []
    document = ""
    for page in reader.pages:
        page_text = page.extract_text() + "\n"
        page_lengths.append(len(page_text))
        document += page_text

    metadata = dict(reader.metadata) if reader.metadata else {}
    author = metadata.get("/Author", "")
    author = author[0] if isinstance(author, list) else author

    # with open(filename, "rb") as file:
    #     print(filename)
    #     bytes_data = file.read()

    P = ParserOuput(
        document=document,
        fullpath=filename,
        author=author,
        page_lengths=tuple(page_lengths),
    )
    return P


def parse_txt(filename):
    with open(filename, "r") as f:
        document = f.read()
        page_lengths = tuple(
            [
                len(document),
            ]
        )
        author = ""  # TODO: see if there is any author metadata

    P = ParserOuput(
        document=document,
        fullpath=filename,
        author=author,
        page_lengths=tuple(page_lengths),
    )

    return P


def parse_csv(filename):

    P = parse_txt(filename)

    return P
