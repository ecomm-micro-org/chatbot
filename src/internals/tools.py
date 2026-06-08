import json

from langchain.tools import tool
from langchain_core.vectorstores import VectorStoreRetriever
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from src.crud.models import Product


def make_product_tools(db: Session, retriever: VectorStoreRetriever):
    """
    Factory function to create product tools
    """

    @tool
    def get_product_by_id(product_id: int) -> str:
        """
        Fetch a single product by its numeric ID.

        params: {product_id:integer}
        """
        product = db.get(Product, product_id)
        if product is None:
            return f"No product found with id={product_id}"
        return json.dumps(_serialize(product))

    @tool
    def fetch_product_by_name(product_name: str) -> str:
        """
        Fetch products whose name contains the given string (case-insensitive).

        params: {product_name:string}
        """
        stmt = select(Product).where(Product.name.ilike(f"%{product_name}%"))
        results = [_serialize(p) for p in db.scalars(stmt).all()]
        if not results:
            return f"No products found matching '{product_name}'"
        return json.dumps(results)

    @tool
    def get_product_by_category(category: str, limit: int = 50) -> str:
        """
        Fetch products in a category ordered by rating.

        params: {category:string,limit:integer}
        """
        stmt = (
            select(Product)
            .where(Product.category == category)
            .order_by(desc(Product.rating))
            .limit(limit)
        )
        results = [_serialize(p) for p in db.scalars(stmt).all()]
        if not results:
            return f"No products found in category '{category}'"
        return json.dumps(results)

    @tool
    def retrieve_from_vector_db(query: str) -> str:
        """
        Search and return relevant product information from the vector database.

        params: {query: string}
        """
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant documents found for the query."
        return "\n\n".join(doc.page_content for doc in docs)

    return [
        retrieve_from_vector_db,
        get_product_by_id,
        fetch_product_by_name,
        get_product_by_category,
    ]


def _serialize(product: Product | None) -> dict | None:
    "Convert a product ORM to a dict that the llm can read"
    if product is None:
        return None
    return {
        "id": product.id,
        "name": product.name,
        "price": float(product.price) if product.price else None,
        "original_price": float(product.original_price),
        "category": product.category,
        "description": product.description,
        "rating": float(product.rating) if product.rating else None,
        "reviews": product.reviews,
        "stock": product.stock,
        "in_stock": product.in_stock,
        "tags": product.tags,
    }
