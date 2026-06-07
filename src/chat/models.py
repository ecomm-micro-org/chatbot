from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, model_validator


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


class MessageRole(str, Enum):
    HUMAN = "human"
    AI = "ai"
    TOOL = "tool"


class Product(BaseModel):
    name: str
    price: float
    original_price: float
    category: str
    description: str
    rating: float
    reviews: int
    stock: int
    in_stock: bool
    tags: list[str] = Field(default_factory=list)

    @classmethod
    def from_raw_block(cls, block: str) -> "Product":
        """Parse a single product block from the ToolMessage content string."""
        data: dict[str, Any] = {}
        for line in block.strip().splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()

            if key == "Name":
                data["name"] = value
            elif key == "Price":
                data["price"] = float(value)
            elif key == "OriginalPrice":
                data["original_price"] = float(value)
            elif key == "Category":
                data["category"] = value
            elif key == "Description":
                data["description"] = value
            elif key == "Rating":
                data["rating"] = float(value)
            elif key == "Reviews":
                data["reviews"] = int(value)
            elif key == "Stock":
                data["stock"] = int(value)
            elif key == "InStock":
                data["in_stock"] = value.lower() == "true"
            elif key == "Tags":
                cleaned = value.strip("[]")
                data["tags"] = [t.strip() for t in cleaned.split() if t.strip()]

        return cls(**data)

    @classmethod
    def parse_tool_content(cls, content: str) -> list["Product"]:
        """
        Split a multi-product ToolMessage content string into individual
        Product objects.  Products are separated by blank lines.
        """
        # Each product block starts with "Name:"
        blocks: list[str] = []
        current: list[str] = []
        for line in content.splitlines():
            if line.startswith("Name:") and current:
                blocks.append("\n".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            blocks.append("\n".join(current))

        return [cls.from_raw_block(b) for b in blocks if b.strip()]


class TokenUsageDetails(BaseModel):
    """Optional nested token detail fields (may be None)."""

    accepted_prediction_tokens: Optional[int] = None
    audio_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    rejected_prediction_tokens: Optional[int] = None


class TokenUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    completion_tokens_details: Optional[TokenUsageDetails] = None
    prompt_tokens_details: Optional[Any] = None
    # Timing fields (present in some providers)
    queue_time: Optional[float] = None
    prompt_time: Optional[float] = None
    completion_time: Optional[float] = None
    total_time: Optional[float] = None


class ResponseMetadata(BaseModel):
    """Metadata attached to AI messages by the LLM provider."""

    token_usage: Optional[TokenUsage] = None
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    system_fingerprint: Optional[str] = None
    id: Optional[str] = None
    service_tier: Optional[str] = None
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None


class UsageMetadata(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_token_details: Optional[dict[str, Any]] = Field(default_factory=dict)
    output_token_details: Optional[dict[str, Any]] = Field(default_factory=dict)


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any]
    id: str
    type: str = "tool_call"


class HumanMessage(BaseModel):
    role: MessageRole = MessageRole.HUMAN
    content: str
    id: Optional[str] = None

    class Config:
        extra = "allow"


class AIMessage(BaseModel):
    role: MessageRole = MessageRole.AI
    content: str
    id: Optional[str] = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    invalid_tool_calls: list[Any] = Field(default_factory=list)
    response_metadata: Optional[ResponseMetadata] = None
    usage_metadata: Optional[UsageMetadata] = None

    class Config:
        extra = "allow"


class ToolMessage(BaseModel):
    role: MessageRole = MessageRole.TOOL
    content: str
    name: Optional[str] = None
    id: Optional[str] = None
    tool_call_id: Optional[str] = None

    # Parsed products — populated automatically
    products: list[Product] = Field(default_factory=list)

    @model_validator(mode="after")
    def parse_products(self) -> "ToolMessage":
        if self.content:
            try:
                self.products = Product.parse_tool_content(self.content)
            except Exception:
                pass  # Leave products empty if parsing fails
        return self

    class Config:
        extra = "allow"


AgentMessage = Union[HumanMessage, AIMessage, ToolMessage]


class AgentResponse(BaseModel):
    """Wraps the full list of messages returned by a LangChain agent run."""

    messages: list[AgentMessage]

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "AgentResponse":
        """
        Parse the raw dict produced by a LangGraph / LangChain agent.

        Expects the shape: {'messages': [HumanMessage(...), AIMessage(...), ...]}
        The LangChain message objects are converted to plain dicts first.
        """
        parsed: list[AgentMessage] = []
        for msg in raw.get("messages", []):
            msg_dict = _langchain_msg_to_dict(msg)
            role = msg_dict.get("role") or _infer_role(msg)
            if role == MessageRole.HUMAN:
                parsed.append(HumanMessage(**msg_dict))
            elif role == MessageRole.TOOL:
                parsed.append(ToolMessage(**msg_dict))
            else:
                parsed.append(AIMessage(**msg_dict))
        return cls(messages=parsed)

    # -- Convenience accessors --

    @property
    def human_messages(self) -> list[HumanMessage]:
        return [m for m in self.messages if isinstance(m, HumanMessage)]

    @property
    def ai_messages(self) -> list[AIMessage]:
        return [m for m in self.messages if isinstance(m, AIMessage)]

    @property
    def tool_messages(self) -> list[ToolMessage]:
        return [m for m in self.messages if isinstance(m, ToolMessage)]

    @property
    def all_products(self) -> list[Product]:
        """Flat list of every product found across all ToolMessages."""
        products: list[Product] = []
        for tm in self.tool_messages:
            products.extend(tm.products)
        return products

    @property
    def final_answer(self) -> Optional[str]:
        """Content of the last AIMessage that has no pending tool calls."""
        for msg in reversed(self.messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                return msg.content
        return None


def _langchain_msg_to_dict(msg: Any) -> dict[str, Any]:
    """Convert a LangChain message object (or plain dict) to a plain dict."""
    if isinstance(msg, dict):
        return msg

    # LangChain messages expose these attributes
    data: dict[str, Any] = {
        "content": getattr(msg, "content", ""),
        "id": getattr(msg, "id", None),
    }

    type_name = type(msg).__name__
    if "Human" in type_name:
        data["role"] = MessageRole.HUMAN
    elif "Tool" in type_name:
        data["role"] = MessageRole.TOOL
        data["name"] = getattr(msg, "name", None)
        data["tool_call_id"] = getattr(msg, "tool_call_id", None)
    else:
        data["role"] = MessageRole.AI
        data["tool_calls"] = getattr(msg, "tool_calls", [])
        data["invalid_tool_calls"] = getattr(msg, "invalid_tool_calls", [])
        raw_meta = getattr(msg, "response_metadata", {}) or {}
        data["response_metadata"] = raw_meta
        raw_usage = getattr(msg, "usage_metadata", {}) or {}
        data["usage_metadata"] = raw_usage

    return data


def _infer_role(msg: Any) -> MessageRole:
    type_name = type(msg).__name__
    if "Human" in type_name:
        return MessageRole.HUMAN
    if "Tool" in type_name:
        return MessageRole.TOOL
    return MessageRole.AI
