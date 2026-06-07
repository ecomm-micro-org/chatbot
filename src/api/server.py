import os
import sys

from src.generate.models import LLMResponse
from src.internals.response_model import llm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../gen/pb"))


import asyncio
import logging
from contextlib import asynccontextmanager

import grpc
from grpc.aio import InternalError

from gen.pb import chatbot_pb2, chatbot_pb2_grpc
from src.chat.models import AgentResponse
from src.internals.workflow import workflow

graph = None


class Chatbot(chatbot_pb2_grpc.ChatbotServiceServicer):
    async def Chat(self, request: chatbot_pb2.ChatRequest, context):
        """Endpoint to interact with the langgraph agents"""
        if graph is None:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("graph is not initialized")
            raise InternalError("graph is not initialized")

        inputs = {"messages": [("user", request.message)]}
        result = graph.invoke(inputs)

        res = AgentResponse.from_raw(result)
        response = res.final_answer
        if response is None:
            raise InternalError("something went wrong")

        return chatbot_pb2.ChatResponse(response=response)

    async def GenerateDescription(
        self, request: chatbot_pb2.GenerateDescriptionRequest, context
    ):
        """Endpoint to generate product descriptions"""

        GENERATE_PROMPT = """"
            You are a description generator for products sold on an ecommerce application.
            Write a description for the following product with the following details:
            product name : {product_name}
            product price : {product_price}
            product category : {product_category}
            tags associated with the product : {tags}

            provide only the description of the product and nothing else,avoid techincal jargon
        """

        prompt = GENERATE_PROMPT.format(
            product_name=request.product_name,
            product_price=request.product_price,
            product_category=request.product_category,
            tags=request.tags,
        )
        response: LLMResponse = llm.invoke([{"role": "user", "content": prompt}])
        return chatbot_pb2.GenerateDescriptionResponse(response=response.content)


async def serve() -> None:
    global graph
    graph = workflow()

    server = grpc.aio.server()
    chatbot_pb2_grpc.add_ChatbotServiceServicer_to_server(Chatbot(), server)
    listen_addr = "[::]:8086"
    server.add_insecure_port(listen_addr)
    logging.info("server listening on port :8086")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())
