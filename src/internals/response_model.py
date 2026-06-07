from langchain.chat_models import init_chat_model
from langchain_core.messages import trim_messages
from .tools import tools

llm = init_chat_model(
    model="llama-3.3-70b-versatile", model_provider="groq", temperature=0
)

llm_with_tools = llm.bind_tools(tools=tools)


trimmer = trim_messages(
    max_tokens=6000,
    strategy="last",
    token_counter=llm,
    include_system=True,
    allow_partial=False,
)

