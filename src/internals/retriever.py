import logging
import os

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

CONNECTION_STRING = os.getenv("DSN")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

if not COLLECTION_NAME or not CONNECTION_STRING:
    logging.error(
        "COLLECTION_NAME or CONNECTION_STRING environment variable is not set"
    )
    raise RuntimeError(
        "COLLECTION_NAME or CONNECTION_STRING environment variable is not set"
    )

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
)


def build_retriever() -> VectorStoreRetriever:
    """Create an PG vector (postgresql) store from document chunks and return a retriever."""

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )

    return vector_store.as_retriever()
