from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.db.db import get_db
from src.internals.retriever import build_retriever

from .tools import make_product_tools

CHAT_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """
    You are a customer support assistant. You help customers with product listings and assisting them with what products are available in our catalog.

    RULES:
    - You must be kind and helpful.
    - You use the tools provided to ONLY fetch or READ product details from the database.
    - You help customers decide what they should buy.
    - If a category has more than one product then you suggest them the best one.
"""

retriever = build_retriever()
db = next(get_db())
tools = make_product_tools(db=db, retriever=retriever)

llm = init_chat_model(model=CHAT_MODEL, model_provider="groq", temperature=0)
llm = llm.bind_tools(tools=tools)


def create_workflow() -> CompiledStateGraph[
    MessagesState, None, MessagesState, MessagesState
]:
    builder = StateGraph(MessagesState)

    builder.add_node("chatbot", chatbot)
    builder.add_node("tools", ToolNode(tools=tools))

    builder.add_edge(START, "chatbot")

    builder.add_conditional_edges(
        "chatbot",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    builder.add_edge(
        "tools",
        "chatbot",
    )

    return builder.compile()


def chatbot(state: MessagesState) -> MessagesState:
    """
    Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using a tool, or simply respond to the user.
    """
    system = {"role": "system", "content": SYSTEM_PROMPT}

    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}


def main():
    from IPython.display import Image

    graph = create_workflow()
    img = Image(graph.get_graph().draw_mermaid_png())

    with open("graph.png", "wb") as f:
        f.write(img.data)


if __name__ == "__main__":
    main()
