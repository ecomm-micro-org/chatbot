from langchain.tools import tool
from langchain_core.vectorstores import VectorStoreRetriever

from .retriever import build_retriever

_retriever: VectorStoreRetriever = build_retriever()


@tool
def retrieve_from_vector_db(query: str) -> str:
    """
    Search and return information about the query from a vector database.

    params: {query:string}
    """
    docs = _retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


@tool
def fetch_from_product_data_from_table(product_name: str) -> str:
    """
    Fetches the product from the databse

    params: {product_name:string}
    """
    pass


tools = [retrieve_from_vector_db, fetch_from_product_data_from_table]
