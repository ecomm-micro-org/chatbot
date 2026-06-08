import asyncio
import logging
import signal

import grpc
from grpc.aio import InternalError
from langchain.chat_models import init_chat_model

from src.chat.models import AgentResponse
from src.chatbot_pb2 import (
    ChatRequest,
    ChatResponse,
    GenerateDescriptionRequest,
    GenerateDescriptionResponse,
)
from src.chatbot_pb2_grpc import (
    ChatbotServiceServicer,
    add_ChatbotServiceServicer_to_server,
)
from src.internals.workflow import create_workflow

graph = None

GENERATOR_CHAT_MODEL = "llama-3.1-8b-instant"
GENERATE_PROMPT = """
    You are a description generator for products sold on an ecommerce application.
    Write a description for the following product with the following details:
    product name : {product_name}
    product price : {product_price}
    product category : {product_category}
    tags associated with the product : {tags}

    provide only the description of the product and nothing else,avoid technical jargon.
"""

raw_llm = init_chat_model(
    model=GENERATOR_CHAT_MODEL, model_provider="groq", temperature=0.7
)


class Chatbot(ChatbotServiceServicer):
    async def Chat(self, request: ChatRequest, context):
        """Endpoint to interact with the langgraph agents"""
        if graph is None:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("graph is not initialized")
            raise InternalError("graph is not initialized")

        inputs = {"messages": [{"role": "user", "content": request.message}]}
        result = graph.invoke(inputs)

        res = AgentResponse.from_raw(result)
        response = res.final_answer
        if response is None:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("something went wrong")
            raise InternalError("something went wrong")

        return ChatResponse(response=response)

    async def GenerateDescription(self, request: GenerateDescriptionRequest, context):
        """Endpoint to generate product descriptions"""
        prompt = GENERATE_PROMPT.format(
            product_name=request.product_name,
            product_price=request.product_price,
            product_category=request.product_category,
            tags=request.tags,
        )
        response = raw_llm.invoke([{"role": "user", "content": prompt}])
        return GenerateDescriptionResponse(response=response.content)


async def serve() -> None:
    global graph
    graph = create_workflow()

    server = grpc.aio.server()
    add_ChatbotServiceServicer_to_server(Chatbot(), server)
    listen_addr = "[::]:8086"
    server.add_insecure_port(listen_addr)
    logging.info("server listening on port :8086")
    await server.start()

    loop = asyncio.get_running_loop()

    async def shutdown():
        logging.info("Shutting down grpc server")
        await server.stop(grace=30)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))

    await server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    asyncio.run(serve())
