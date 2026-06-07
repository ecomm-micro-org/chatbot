from IPython.display import Image
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .nodes import chatbot
from .tools import tools


def workflow() -> StateGraph:
    workflow = StateGraph(MessagesState)

    workflow.add_node("chatbot", chatbot)
    workflow.add_node("tools", ToolNode(tools=tools))

    workflow.add_edge(START, "chatbot")

    workflow.add_conditional_edges(
        "chatbot",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    workflow.add_edge(
        "tools",
        "chatbot",
    )

    graph = workflow.compile()
    return graph


def main():
    graph = workflow()
    img = Image(graph.get_graph().draw_mermaid_png())

    with open("graph.png", "wb") as f:
        f.write(img.data)


if __name__ == "__main__":
    main()
