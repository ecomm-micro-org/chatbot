from typing import Any, List, Optional

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    completion_tokens_details: Optional[Any] = None
    prompt_tokens_details: Optional[Any] = None
    queue_time: float
    prompt_time: float
    completion_time: float
    total_time: float


class ResponseMetadata(BaseModel):
    token_usage: TokenUsage
    model_provider: str
    model_name: str
    system_fingerprint: str
    id: str
    service_tier: str
    finish_reason: str
    logprobs: Optional[Any] = None


class UsageMetadata(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_token_details: dict = Field(default_factory=dict)
    output_token_details: dict = Field(default_factory=dict)


class AdditionalKwargs(BaseModel):
    refusal: Optional[str] = None


class LLMResponse(BaseModel):
    content: str
    additional_kwargs: AdditionalKwargs
    response_metadata: ResponseMetadata
    type: str
    name: Optional[str] = None
    id: str
    tool_calls: list = Field(default_factory=list)
    invalid_tool_calls: list = Field(default_factory=list)
    usage_metadata: UsageMetadata
