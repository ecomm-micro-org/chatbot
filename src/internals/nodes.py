import os
import re

import psycopg2
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.graph import MessagesState

from .response_model import llm_with_tools
from .retriever import build_retriever

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks."
    "Use the following pieces of retrieved context to answer the question."
    "If you don't know the answer, just say that you don't know."
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)
REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)


GENERATE_SQL_PROMPT = (
    "You are a sql expert.\n"
    "Using the database schema information provided below generate a SQl query that answers the user's quertion the best\n"
    "the query must be optimal\n"
    "return only the SQL query without any markdown formatting or explanations.\n"
    "here is the databse schema:"
    "\n ------- \n"
    "{rag_context}"
    "\n ------- \n"
)


_retriever: VectorStoreRetriever = build_retriever()
llm = init_chat_model("openai/gpt-oss-120b", model_provider="groq", temperature=0)


def chatbot(state: MessagesState) -> MessagesState:
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using a tool, or simply respond to the user.
    """
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def generate_answer(state: MessagesState) -> MessagesState:
    """Generate an answer."""

    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


def rewrite_question(state: MessagesState) -> MessagesState:
    """Rewrite the original user question."""

    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]}


def generate_sql_query(state: MessagesState) -> dict:
    """Generate sql query if needed"""

    message = state["messages"][-1].content

    if not message or not message.strip():
        error_message = "No valid user message found for SQL generation"
        return {
            "sql_query": "",
            "message": state["messages"] + [SystemMessage(content=error_message)],
        }

    try:
        rag_context = _retriever.invoke(message)
        print(f"rag context is {rag_context}")
        if not rag_context:
            rag_context = "no relevant database schema found"
    except Exception as e:
        rag_context = f"error retrieving schema context {str(e)}"

    prompt = GENERATE_SQL_PROMPT.format(rag_context=rag_context)
    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        return {
            "sql_query": response.content,
            "messages": state["messages"]
            + [
                SystemMessage(
                    content=f"geenrated sql query using rag context : {response.content}"
                )
            ],
        }
    except Exception as e:
        error_msg = f"Error generating SQL: {str(e)}"
        return {
            "sql_query": "",
            "messages": state["messages"] + [SystemMessage(content=error_msg)],
        }


def _extract_sql_query(state: MessagesState) -> str:
    """Extract SQL query from state messages"""

    if not state.messages:
        return ""

    latest_message = state.messages[-1].content

    sql_patterns = [
        r"```sql\s*(.*?)\s*```",  # SQL in code blocks
        r"```\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|WITH).*?```",  # SQL without sql tag
        r"(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|WITH)\s+.*?(?:;|$)",  # SQL statements
    ]

    for pattern in sql_patterns:
        matches = re.findall(pattern, latest_message, re.IGNORECASE | re.DOTALL)
        if matches:
            return matches[0].strip()

    return ""


def _convert_sql_response_to_text(sql_results: str) -> str:
    """
    Convert SQL query results to natural language text.
    This represents the "Convert response to text" step in the flowchart.
    """
    if not sql_results:
        return "No results to convert."

    # Simple conversion - in a real implementation, you might use an LLM here
    # to convert the results to more natural language
    if "Query Results:" in sql_results:
        return f"Here are the results from your query:\n{sql_results}"
    elif "Query executed successfully" in sql_results:
        return f"Your query was executed successfully. {sql_results}"
    else:
        return f"Query result: {sql_results}"


def execute_sql_query(state: MessagesState) -> dict:
    """Executes the generated SQL query based on given state and returns text output"""

    sql_query = state.get("sql_query", "")
    if not sql_query:
        sql_query = _extract_sql_query(state)

    if not sql_query:
        return {
            "status": "error",
            "results": "No SQL query found in the conversation. Please provide a valid SQL query.",
            "sql_output": "No SQL query available to execute.",
        }

    try:
        DSN = os.getenv("DSN")
        conn = psycopg2.connect(dsn=DSN)
        cursor = conn.cursor()

        cursor.execute(sql_query)

        columns = (
            [description[0] for description in cursor.description]
            if cursor.description
            else []
        )

        rows = cursor.fetchall()

        # Print execution results
        print(f"✅ Query executed successfully!")
        print(f"📊 Results: {len(rows)} rows returned")
        if columns:
            print(f"📋 Columns: {', '.join(columns)}")

        if not rows:
            result_text = "query executed successfully. no results returned."
        else:
            if columns:
                col_widths = [len(col) for col in columns]
                for row in rows:
                    for i, cell in enumerate(row):
                        col_widths[i] = max(col_widths[i], len(str(cell)))

                header = "|".join(
                    f"{col:<{col_widths[i]}}" for i, col in enumerate(columns)
                )
                separator = "-" * len(header)

                formatted_rows = []
                for row in rows:
                    formatted_row = " | ".join(
                        f"{str(cell):<{col_widths[i]}}" for i, cell in enumerate(row)
                    )
                    formatted_rows.append(formatted_row)

                result_text = f"Query Results:\n{separator}\n{header}\n{separator}\n"
                result_text += "\n".join(formatted_rows)
                result_text += f"\n{separator}\nTotal rows: {len(rows)}"
            else:
                # For queries that don't return data (INSERT, UPDATE, DELETE, etc.)
                result_text = f"Query executed successfully. {len(rows)} rows affected."

        conn.close()
        natural_language_result = _convert_sql_response_to_text(result_text)

        return {
            "status": "success",
            "results": result_text,
            "sql_query": sql_query,
            "sql_output": natural_language_result,
            "row_count": len(rows),
            "messages": state.messages
            + [SystemMessage(content=natural_language_result)],
        }

    except psycopg2.Error as e:
        error_msg = f"Database error: {str(e)}"
        print(f"❌ [services/sql/execute_sql_query.py:execute_sql_query] {error_msg}")
        return {
            "status": "error",
            "results": error_msg,
            "sql_query": sql_query,
            "sql_output": error_msg,
            "messages": state["messages"] + [SystemMessage(content=error_msg)],
        }
    except Exception as e:
        error_msg = f"Error executing SQL query: {str(e)}"
        print(f"❌ [services/sql/execute_sql_query.py:execute_sql_query] {error_msg}")
        return {
            "status": "error",
            "results": error_msg,
            "sql_query": sql_query,
            "sql_output": error_msg,
            "messages": state["messages"] + [SystemMessage(content=error_msg)],
        }
