import re
from typing import Literal

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Respond ONLY with raw JSON, no markdown, no explanation:\n"
    '{{ "binary_score": "yes" }} or {{ "binary_score": "no" }}'
)


class RelevenceGrader(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


grader_model = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages([("human", GRADE_PROMPT)])


def strip_think_args(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


parser = JsonOutputParser()

grader_chain = prompt | grader_model | parser


def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    response = (prompt | grader_model).invoke(
        {"question": question, "context": context}
    )
    cleaned_response = strip_think_args(response.content)

    try:
        result = parser.parse(cleaned_response)
        score = result.get("binary_score", "no")
    except Exception:
        score = "no"

    if score == "yes":
        return "generate_answer"
    return "rewrite_question"
