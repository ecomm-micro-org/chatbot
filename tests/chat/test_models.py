"""Comprehensive pytest tests for src.chat.models."""

from unittest.mock import MagicMock

import pytest

from src.chat.models import (
    AgentResponse,
    AIMessage,
    HumanMessage,
    MessageRole,
    Product,
    ResponseMetadata,
    TokenUsage,
    TokenUsageDetails,
    ToolCall,
    ToolMessage,
    UsageMetadata,
    _infer_role,
    _langchain_msg_to_dict,
)

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

SINGLE_PRODUCT_BLOCK = """\
Name: Test Widget
Price: 29.99
OriginalPrice: 39.99
Category: Electronics
Description: A handy widget
Rating: 4.5
Reviews: 120
Stock: 50
InStock: true
Tags: [gadget widget]
"""

MULTI_PRODUCT_CONTENT = """\
Name: Alpha
Price: 10.0
OriginalPrice: 15.0
Category: Books
Description: First book
Rating: 4.0
Reviews: 50
Stock: 100
InStock: true
Tags: [reading]
Name: Beta
Price: 20.0
OriginalPrice: 25.0
Category: Music
Description: Second album
Rating: 3.5
Reviews: 30
Stock: 0
InStock: false
Tags: [audio music]
"""


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _human(content: str) -> HumanMessage:
    return HumanMessage(content=content)


def _ai(content: str, tool_calls: list | None = None) -> AIMessage:
    return AIMessage(content=content, tool_calls=tool_calls or [])


def _tool(content: str) -> ToolMessage:
    return ToolMessage(content=content)


# ---------------------------------------------------------------------------
# Fake LangChain-like message objects for from_raw / _langchain_msg_to_dict
# ---------------------------------------------------------------------------


class _FakeHumanMsg:
    """Minimal stand-in for a LangChain HumanMessage."""

    def __init__(self, content: str, id: str | None = None):
        self.content = content
        self.id = id


class _FakeAIMsg:
    """Minimal stand-in for a LangChain AIMessage.

    ``usage_metadata`` defaults to a complete dict (not None/empty) so that the
    object can be round-tripped through ``AgentResponse.from_raw`` without
    triggering a Pydantic validation error on ``AIMessage.usage_metadata``.
    The ``test_ai_usage_metadata_none_becomes_empty_dict`` test explicitly sets
    the attribute to None to exercise the falsy-→-{} branch in isolation.
    """

    def __init__(
        self,
        content: str,
        id: str | None = None,
        tool_calls=None,
        invalid_tool_calls=None,
        response_metadata=None,
        usage_metadata=None,
    ):
        self.content = content
        self.id = id
        self.tool_calls = tool_calls or []
        self.invalid_tool_calls = invalid_tool_calls or []
        self.response_metadata = (
            response_metadata if response_metadata is not None else {}
        )
        # Use a valid token-count dict by default so that _langchain_msg_to_dict
        # produces a UsageMetadata-compatible payload for end-to-end from_raw tests.
        self.usage_metadata = (
            usage_metadata
            if usage_metadata is not None
            else {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        )


class _FakeToolMsg:
    """Minimal stand-in for a LangChain ToolMessage."""

    def __init__(
        self,
        content: str,
        id: str | None = None,
        name: str | None = None,
        tool_call_id: str | None = None,
    ):
        self.content = content
        self.id = id
        self.name = name
        self.tool_call_id = tool_call_id


# ===========================================================================
# MessageRole
# ===========================================================================


class TestMessageRole:
    def test_values(self):
        assert MessageRole.HUMAN == "human"
        assert MessageRole.AI == "ai"
        assert MessageRole.TOOL == "tool"

    def test_is_string_subclass(self):
        for role in (MessageRole.HUMAN, MessageRole.AI, MessageRole.TOOL):
            assert isinstance(role, str)

    def test_equality_with_plain_string(self):
        assert MessageRole.HUMAN == "human"
        assert MessageRole.AI == "ai"
        assert MessageRole.TOOL == "tool"

    def test_members_are_distinct(self):
        assert MessageRole.HUMAN != MessageRole.AI
        assert MessageRole.AI != MessageRole.TOOL


# ===========================================================================
# Product.from_raw_block
# ===========================================================================


class TestProductFromRawBlock:
    def test_complete_valid_block(self):
        p = Product.from_raw_block(SINGLE_PRODUCT_BLOCK)
        assert p.name == "Test Widget"
        assert p.price == 29.99
        assert p.original_price == 39.99
        assert p.category == "Electronics"
        assert p.description == "A handy widget"
        assert p.rating == 4.5
        assert p.reviews == 120
        assert p.stock == 50
        assert p.in_stock is True
        assert p.tags == ["gadget", "widget"]

    def test_in_stock_false_lowercase(self):
        block = (
            "Name: Sold Out\nPrice: 5.0\nOriginalPrice: 10.0\nCategory: Misc\n"
            "Description: Gone\nRating: 3.0\nReviews: 5\nStock: 0\nInStock: false\nTags: []\n"
        )
        p = Product.from_raw_block(block)
        assert p.in_stock is False
        assert p.stock == 0

    def test_in_stock_true_titlecase(self):
        # "True".lower() == "true" -> truthy
        block = (
            "Name: Available\nPrice: 5.0\nOriginalPrice: 5.0\nCategory: C\n"
            "Description: D\nRating: 1.0\nReviews: 1\nStock: 10\nInStock: True\nTags: []\n"
        )
        p = Product.from_raw_block(block)
        assert p.in_stock is True

    def test_in_stock_false_uppercase(self):
        block = (
            "Name: Out\nPrice: 1.0\nOriginalPrice: 1.0\nCategory: C\n"
            "Description: D\nRating: 1.0\nReviews: 1\nStock: 0\nInStock: FALSE\nTags: []\n"
        )
        p = Product.from_raw_block(block)
        assert p.in_stock is False

    def test_colon_in_description_value(self):
        # partition(":") splits only on the first colon, so the rest is preserved
        block = (
            "Name: Gadget\nPrice: 9.99\nOriginalPrice: 14.99\nCategory: Tech\n"
            "Description: Works great: buy it now: seriously\n"
            "Rating: 4.8\nReviews: 200\nStock: 30\nInStock: true\nTags: [tech]\n"
        )
        p = Product.from_raw_block(block)
        assert p.description == "Works great: buy it now: seriously"

    def test_multiple_tags(self):
        block = (
            "Name: Tagged\nPrice: 1.0\nOriginalPrice: 2.0\nCategory: C\n"
            "Description: D\nRating: 1.0\nReviews: 1\nStock: 1\nInStock: true\n"
            "Tags: [a b c d]\n"
        )
        p = Product.from_raw_block(block)
        assert p.tags == ["a", "b", "c", "d"]

    def test_empty_tags(self):
        block = (
            "Name: No Tags\nPrice: 1.0\nOriginalPrice: 2.0\nCategory: C\n"
            "Description: D\nRating: 1.0\nReviews: 1\nStock: 1\nInStock: true\n"
            "Tags: []\n"
        )
        p = Product.from_raw_block(block)
        assert p.tags == []

    def test_lines_without_colon_are_skipped(self):
        block = (
            "Name: Widget\n"
            "this line has no colon and must be ignored\n"
            "Price: 5.0\nOriginalPrice: 5.0\nCategory: C\n"
            "Description: D\nRating: 1.0\nReviews: 1\nStock: 1\nInStock: true\nTags: []\n"
        )
        p = Product.from_raw_block(block)
        assert p.name == "Widget"
        assert p.price == 5.0

    def test_key_value_whitespace_stripped(self):
        block = (
            "  Name  :   Spaced Widget  \n"
            "  Price  :  99.0  \n"
            "OriginalPrice: 99.0\nCategory: C\nDescription: D\n"
            "Rating: 1.0\nReviews: 1\nStock: 1\nInStock: true\nTags: []\n"
        )
        p = Product.from_raw_block(block)
        assert p.name == "Spaced Widget"
        assert p.price == 99.0

    def test_missing_required_field_raises(self):
        # No Price field -> Pydantic ValidationError
        block = (
            "Name: Incomplete\nCategory: C\nDescription: D\n"
            "Rating: 1.0\nReviews: 1\nStock: 1\nInStock: true\nTags: []\n"
        )
        with pytest.raises(Exception):
            Product.from_raw_block(block)

    def test_zero_price_and_reviews(self):
        block = (
            "Name: Free\nPrice: 0.0\nOriginalPrice: 0.0\nCategory: C\n"
            "Description: D\nRating: 0.0\nReviews: 0\nStock: 0\nInStock: false\nTags: []\n"
        )
        p = Product.from_raw_block(block)
        assert p.price == 0.0
        assert p.reviews == 0

    def test_unknown_keys_are_ignored(self):
        # Unknown keys like "Foo" are simply not assigned to data
        block = (
            "Name: Widget\nFoo: bar\nPrice: 5.0\nOriginalPrice: 5.0\nCategory: C\n"
            "Description: D\nRating: 1.0\nReviews: 1\nStock: 1\nInStock: true\nTags: []\n"
        )
        p = Product.from_raw_block(block)
        assert p.name == "Widget"


# ===========================================================================
# Product.parse_tool_content
# ===========================================================================


class TestProductParseToolContent:
    def test_single_product(self):
        products = Product.parse_tool_content(SINGLE_PRODUCT_BLOCK)
        assert len(products) == 1
        assert products[0].name == "Test Widget"

    def test_multi_product(self):
        products = Product.parse_tool_content(MULTI_PRODUCT_CONTENT)
        assert len(products) == 2
        assert products[0].name == "Alpha"
        assert products[1].name == "Beta"

    def test_multi_product_field_correctness(self):
        products = Product.parse_tool_content(MULTI_PRODUCT_CONTENT)
        assert products[0].in_stock is True
        assert products[0].category == "Books"
        assert products[1].in_stock is False
        assert products[1].price == 20.0

    def test_three_products(self):
        block = (
            "Name: P1\nPrice: 1.0\nOriginalPrice: 1.0\nCategory: C\nDescription: D\n"
            "Rating: 1.0\nReviews: 1\nStock: 1\nInStock: true\nTags: []\n"
            "Name: P2\nPrice: 2.0\nOriginalPrice: 2.0\nCategory: C\nDescription: D\n"
            "Rating: 2.0\nReviews: 2\nStock: 2\nInStock: true\nTags: []\n"
            "Name: P3\nPrice: 3.0\nOriginalPrice: 3.0\nCategory: C\nDescription: D\n"
            "Rating: 3.0\nReviews: 3\nStock: 3\nInStock: false\nTags: [x]\n"
        )
        products = Product.parse_tool_content(block)
        assert len(products) == 3
        assert products[2].name == "P3"
        assert products[2].in_stock is False
        assert products[2].tags == ["x"]

    def test_empty_string_returns_empty_list(self):
        assert Product.parse_tool_content("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert Product.parse_tool_content("   \n\n  ") == []

    def test_content_without_name_lines_propagates_error(self):
        # No "Name:" line means one block is built; from_raw_block raises because
        # required fields are missing. parse_tool_content does NOT swallow errors.
        with pytest.raises(Exception):
            Product.parse_tool_content("Price: 10.0\nCategory: Books")

    def test_blocks_split_on_each_name_line(self):
        content = MULTI_PRODUCT_CONTENT
        products = Product.parse_tool_content(content)
        # Ensure second product is truly a separate entity
        assert products[1].category == "Music"

    def test_leading_content_before_first_name_raises(self):
        # Any non-empty line before the first "Name:" forms its own block.
        # from_raw_block raises ValidationError for that block (all required
        # fields are missing), so parse_tool_content propagates the error.
        content = (
            "SomeHeader: ignored\n"
            "Name: Widget\nPrice: 5.0\nOriginalPrice: 5.0\nCategory: C\n"
            "Description: D\nRating: 1.0\nReviews: 1\nStock: 1\nInStock: true\nTags: []\n"
        )
        with pytest.raises(Exception):
            Product.parse_tool_content(content)


# ===========================================================================
# ToolCall
# ===========================================================================


class TestToolCall:
    def test_basic_creation(self):
        tc = ToolCall(name="search", args={"query": "shoes"}, id="tc-001")
        assert tc.name == "search"
        assert tc.args == {"query": "shoes"}
        assert tc.id == "tc-001"

    def test_default_type_is_tool_call(self):
        tc = ToolCall(name="noop", args={}, id="tc-002")
        assert tc.type == "tool_call"

    def test_custom_type_overridden(self):
        tc = ToolCall(name="lookup", args={}, id="tc-003", type="custom")
        assert tc.type == "custom"

    def test_empty_args(self):
        tc = ToolCall(name="noop", args={}, id="tc-004")
        assert tc.args == {}

    def test_complex_nested_args(self):
        tc = ToolCall(
            name="filter",
            args={"category": "shoes", "max_price": 100, "tags": ["sale", "new"]},
            id="tc-005",
        )
        assert tc.args["category"] == "shoes"
        assert tc.args["max_price"] == 100
        assert tc.args["tags"] == ["sale", "new"]

    def test_missing_required_field_raises(self):
        with pytest.raises(Exception):
            ToolCall(name="search", id="tc-006")  # type: ignore[call-arg]  # missing args


# ===========================================================================
# HumanMessage
# ===========================================================================


class TestHumanMessage:
    def test_default_role_is_human(self):
        msg = HumanMessage(content="Hi!")
        assert msg.role == MessageRole.HUMAN
        assert msg.role == "human"

    def test_content_stored(self):
        msg = HumanMessage(content="Tell me about shoes")
        assert msg.content == "Tell me about shoes"

    def test_id_defaults_to_none(self):
        msg = HumanMessage(content="Hi")
        assert msg.id is None

    def test_id_can_be_set(self):
        msg = HumanMessage(content="Hi", id="msg-1")
        assert msg.id == "msg-1"

    def test_empty_content_allowed(self):
        msg = HumanMessage(content="")
        assert msg.content == ""

    def test_extra_fields_allowed(self):
        # Config: extra = "allow"
        msg = HumanMessage(content="Hi", extra_field="extra_value")
        assert msg.extra_field == "extra_value"  # type: ignore[attr-defined]


# ===========================================================================
# AIMessage
# ===========================================================================


class TestAIMessage:
    def test_default_role_is_ai(self):
        msg = AIMessage(content="Hello!")
        assert msg.role == MessageRole.AI
        assert msg.role == "ai"

    def test_tool_calls_default_empty(self):
        msg = AIMessage(content="Response")
        assert msg.tool_calls == []

    def test_invalid_tool_calls_default_empty(self):
        msg = AIMessage(content="Response")
        assert msg.invalid_tool_calls == []

    def test_with_tool_calls(self):
        tc = ToolCall(name="search", args={}, id="tc-1")
        msg = AIMessage(content="", tool_calls=[tc])
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "search"

    def test_id_defaults_to_none(self):
        msg = AIMessage(content="Hi")
        assert msg.id is None

    def test_id_can_be_set(self):
        msg = AIMessage(content="Hi", id="ai-1")
        assert msg.id == "ai-1"

    def test_response_metadata_attached(self):
        meta = ResponseMetadata(finish_reason="stop", model_name="gpt-4")
        msg = AIMessage(content="Done", response_metadata=meta)
        assert msg.response_metadata is not None
        assert msg.response_metadata.finish_reason == "stop"

    def test_usage_metadata_attached(self):
        usage = UsageMetadata(input_tokens=10, output_tokens=20, total_tokens=30)
        msg = AIMessage(content="Done", usage_metadata=usage)
        assert msg.usage_metadata is not None
        assert msg.usage_metadata.total_tokens == 30

    def test_empty_content_allowed(self):
        msg = AIMessage(content="")
        assert msg.content == ""

    def test_extra_fields_allowed(self):
        msg = AIMessage(content="Hi", custom_attr="value")
        assert msg.custom_attr == "value"  # type: ignore[attr-defined]


class TestToolMessage:
    def test_default_role_is_tool(self):
        msg = ToolMessage(content="result")
        assert msg.role == MessageRole.TOOL
        assert msg.role == "tool"

    def test_optional_fields_default_none(self):
        msg = ToolMessage(content="result")
        assert msg.name is None
        assert msg.id is None
        assert msg.tool_call_id is None

    def test_optional_fields_set(self):
        msg = ToolMessage(
            content="result", name="search", id="tm-1", tool_call_id="tc-1"
        )
        assert msg.name == "search"
        assert msg.id == "tm-1"
        assert msg.tool_call_id == "tc-1"

    def test_auto_parses_single_product(self):
        msg = ToolMessage(content=SINGLE_PRODUCT_BLOCK)
        assert len(msg.products) == 1
        assert msg.products[0].name == "Test Widget"

    def test_auto_parses_multiple_products(self):
        msg = ToolMessage(content=MULTI_PRODUCT_CONTENT)
        assert len(msg.products) == 2
        assert msg.products[0].name == "Alpha"
        assert msg.products[1].name == "Beta"

    def test_empty_content_yields_no_products(self):
        msg = ToolMessage(content="")
        assert msg.products == []

    def test_whitespace_content_yields_no_products(self):
        msg = ToolMessage(content="   \n\n  ")
        assert msg.products == []

    def test_malformed_content_silently_yields_no_products(self):
        # parse_tool_content raises when a block lacks required fields;
        # the model_validator catches the exception and leaves products = [].
        msg = ToolMessage(content="this is not valid product data at all!!!")
        assert msg.products == []

    def test_extra_fields_allowed(self):
        msg = ToolMessage(content="result", extra_key="extra_value")
        assert msg.extra_key == "extra_value"  # type: ignore[attr-defined]

    def test_product_in_stock_parsed_correctly(self):
        msg = ToolMessage(content=MULTI_PRODUCT_CONTENT)
        assert msg.products[0].in_stock is True
        assert msg.products[1].in_stock is False


class TestTokenUsageDetails:
    def test_all_fields_optional_default_none(self):
        d = TokenUsageDetails()
        assert d.accepted_prediction_tokens is None
        assert d.audio_tokens is None
        assert d.reasoning_tokens is None
        assert d.rejected_prediction_tokens is None

    def test_with_values(self):
        d = TokenUsageDetails(
            accepted_prediction_tokens=10,
            audio_tokens=0,
            reasoning_tokens=5,
            rejected_prediction_tokens=2,
        )
        assert d.accepted_prediction_tokens == 10
        assert d.reasoning_tokens == 5
        assert d.rejected_prediction_tokens == 2


class TestTokenUsage:
    def test_required_fields(self):
        u = TokenUsage(completion_tokens=50, prompt_tokens=100, total_tokens=150)
        assert u.completion_tokens == 50
        assert u.prompt_tokens == 100
        assert u.total_tokens == 150

    def test_optional_fields_default_none(self):
        u = TokenUsage(completion_tokens=10, prompt_tokens=20, total_tokens=30)
        assert u.completion_tokens_details is None
        assert u.queue_time is None
        assert u.prompt_time is None
        assert u.completion_time is None
        assert u.total_time is None

    def test_with_completion_details(self):
        details = TokenUsageDetails(reasoning_tokens=5)
        u = TokenUsage(
            completion_tokens=10,
            prompt_tokens=20,
            total_tokens=30,
            completion_tokens_details=details,
        )
        assert u.completion_tokens_details is not None
        assert u.completion_tokens_details.reasoning_tokens == 5

    def test_with_timing_fields(self):
        u = TokenUsage(
            completion_tokens=1,
            prompt_tokens=1,
            total_tokens=2,
            queue_time=0.01,
            prompt_time=0.1,
            completion_time=0.2,
            total_time=0.31,
        )
        assert u.queue_time == pytest.approx(0.01)
        assert u.total_time == pytest.approx(0.31)


class TestResponseMetadata:
    def test_all_fields_optional(self):
        meta = ResponseMetadata()
        assert meta.token_usage is None
        assert meta.model_provider is None
        assert meta.model_name is None
        assert meta.system_fingerprint is None
        assert meta.id is None
        assert meta.service_tier is None
        assert meta.finish_reason is None
        assert meta.logprobs is None

    def test_with_all_fields(self):
        usage = TokenUsage(completion_tokens=5, prompt_tokens=10, total_tokens=15)
        meta = ResponseMetadata(
            token_usage=usage,
            model_provider="openai",
            model_name="gpt-4",
            finish_reason="stop",
            id="resp-001",
            service_tier="default",
        )
        assert meta.model_provider == "openai"
        assert meta.finish_reason == "stop"
        assert meta.token_usage is not None
        assert meta.token_usage.total_tokens == 15


class TestUsageMetadata:
    def test_required_fields(self):
        um = UsageMetadata(input_tokens=5, output_tokens=10, total_tokens=15)
        assert um.input_tokens == 5
        assert um.output_tokens == 10
        assert um.total_tokens == 15

    def test_optional_details_default_empty_dict(self):
        um = UsageMetadata(input_tokens=5, output_tokens=10, total_tokens=15)
        assert um.input_token_details == {}
        assert um.output_token_details == {}

    def test_with_detail_dicts(self):
        um = UsageMetadata(
            input_tokens=5,
            output_tokens=10,
            total_tokens=15,
            input_token_details={"cache_read": 3},
            output_token_details={"reasoning": 7},
        )
        assert um.input_token_details["cache_read"] == 3
        assert um.output_token_details["reasoning"] == 7

    def test_missing_required_fields_raise(self):
        with pytest.raises(Exception):
            UsageMetadata(input_tokens=5)  # type: ignore[call-arg]


class TestAgentResponseProperties:
    # --- human_messages ---

    def test_human_messages_filtered(self):
        resp = AgentResponse(messages=[_human("Hi"), _ai("Hello"), _human("Again")])
        assert len(resp.human_messages) == 2
        assert all(isinstance(m, HumanMessage) for m in resp.human_messages)
        assert resp.human_messages[0].content == "Hi"
        assert resp.human_messages[1].content == "Again"

    def test_human_messages_empty_when_none(self):
        resp = AgentResponse(messages=[_ai("Only AI")])
        assert resp.human_messages == []

    # --- ai_messages ---

    def test_ai_messages_filtered(self):
        resp = AgentResponse(messages=[_human("Hi"), _ai("R1"), _ai("R2")])
        assert len(resp.ai_messages) == 2
        assert all(isinstance(m, AIMessage) for m in resp.ai_messages)

    def test_ai_messages_empty_when_none(self):
        resp = AgentResponse(messages=[_human("Hi")])
        assert resp.ai_messages == []

    # --- tool_messages ---

    def test_tool_messages_filtered(self):
        resp = AgentResponse(messages=[_ai("Searching"), _tool("result")])
        assert len(resp.tool_messages) == 1
        assert isinstance(resp.tool_messages[0], ToolMessage)

    def test_tool_messages_empty_when_none(self):
        resp = AgentResponse(messages=[_human("Hi"), _ai("Hi")])
        assert resp.tool_messages == []

    # --- empty messages list ---

    def test_all_properties_empty_when_no_messages(self):
        resp = AgentResponse(messages=[])
        assert resp.human_messages == []
        assert resp.ai_messages == []
        assert resp.tool_messages == []
        assert resp.all_products == []
        assert resp.final_answer is None

    # --- all_products ---

    def test_all_products_from_single_tool_message(self):
        resp = AgentResponse(messages=[_ai("Searching"), _tool(MULTI_PRODUCT_CONTENT)])
        products = resp.all_products
        assert len(products) == 2
        assert products[0].name == "Alpha"
        assert products[1].name == "Beta"

    def test_all_products_aggregated_across_multiple_tool_messages(self):
        resp = AgentResponse(
            messages=[_tool(SINGLE_PRODUCT_BLOCK), _tool(MULTI_PRODUCT_CONTENT)]
        )
        assert len(resp.all_products) == 3  # 1 + 2

    def test_all_products_empty_when_no_tool_messages(self):
        resp = AgentResponse(messages=[_human("Hi"), _ai("Hello")])
        assert resp.all_products == []

    def test_all_products_empty_when_tool_message_has_no_valid_products(self):
        resp = AgentResponse(messages=[_tool("no valid product data here")])
        assert resp.all_products == []

    # --- final_answer ---

    def test_final_answer_returns_last_ai_without_tool_calls(self):
        resp = AgentResponse(messages=[_human("Hi"), _ai("Final answer")])
        assert resp.final_answer == "Final answer"

    def test_final_answer_skips_ai_messages_with_tool_calls(self):
        tc = ToolCall(name="search", args={}, id="tc-1")
        resp = AgentResponse(
            messages=[
                _ai("", tool_calls=[tc]),
                _tool(""),
                _ai("Here is the result"),
            ]
        )
        assert resp.final_answer == "Here is the result"

    def test_final_answer_returns_last_when_multiple_eligible(self):
        resp = AgentResponse(messages=[_ai("First answer"), _ai("Second answer")])
        assert resp.final_answer == "Second answer"

    def test_final_answer_none_when_all_ai_have_tool_calls(self):
        tc = ToolCall(name="search", args={}, id="tc-1")
        resp = AgentResponse(messages=[_human("Hi"), _ai("", tool_calls=[tc])])
        assert resp.final_answer is None

    def test_final_answer_none_when_no_ai_messages(self):
        resp = AgentResponse(messages=[_human("Hi")])
        assert resp.final_answer is None

    def test_final_answer_with_empty_content_ai(self):
        # An AI message with empty content and no tool_calls is still eligible
        resp = AgentResponse(messages=[_ai("First"), _ai("")])
        assert resp.final_answer == ""


# AgentResponse.from_raw
class TestAgentResponseFromRaw:
    # --- dict-based messages ---

    def test_dict_human_message(self):
        raw = {"messages": [{"role": "human", "content": "Hello"}]}
        resp = AgentResponse.from_raw(raw)
        assert len(resp.messages) == 1
        assert isinstance(resp.messages[0], HumanMessage)
        assert resp.messages[0].content == "Hello"

    def test_dict_ai_message(self):
        raw = {"messages": [{"role": "ai", "content": "Hi there"}]}
        resp = AgentResponse.from_raw(raw)
        assert isinstance(resp.messages[0], AIMessage)
        assert resp.messages[0].content == "Hi there"

    def test_dict_tool_message(self):
        raw = {"messages": [{"role": "tool", "content": "search result"}]}
        resp = AgentResponse.from_raw(raw)
        assert isinstance(resp.messages[0], ToolMessage)
        assert resp.messages[0].content == "search result"

    def test_dict_unknown_role_raises(self):
        # from_raw falls through to AIMessage(**msg_dict) for unrecognised roles.
        # Pydantic rejects "unknown" because it is not a valid MessageRole value.
        raw = {"messages": [{"role": "unknown", "content": "mystery"}]}
        with pytest.raises(Exception):
            AgentResponse.from_raw(raw)

    def test_dict_mixed_message_sequence(self):
        raw = {
            "messages": [
                {"role": "human", "content": "Find shoes"},
                {"role": "ai", "content": "Searching..."},
                {"role": "tool", "content": ""},
                {"role": "ai", "content": "Here are shoes"},
            ]
        }
        resp = AgentResponse.from_raw(raw)
        assert len(resp.messages) == 4
        assert isinstance(resp.messages[0], HumanMessage)
        assert isinstance(resp.messages[1], AIMessage)
        assert isinstance(resp.messages[2], ToolMessage)
        assert isinstance(resp.messages[3], AIMessage)

    def test_empty_messages_list(self):
        resp = AgentResponse.from_raw({"messages": []})
        assert resp.messages == []

    def test_missing_messages_key(self):
        resp = AgentResponse.from_raw({})
        assert resp.messages == []

    # --- object-based (LangChain-like) messages ---

    def test_object_human_message(self):
        raw = {"messages": [_FakeHumanMsg(content="Hello from obj", id="h-1")]}
        resp = AgentResponse.from_raw(raw)
        assert isinstance(resp.messages[0], HumanMessage)
        assert resp.messages[0].content == "Hello from obj"
        assert resp.messages[0].id == "h-1"

    def test_object_ai_message(self):
        raw = {"messages": [_FakeAIMsg(content="AI reply", id="a-1")]}
        resp = AgentResponse.from_raw(raw)
        assert isinstance(resp.messages[0], AIMessage)
        assert resp.messages[0].content == "AI reply"

    def test_object_tool_message(self):
        raw = {
            "messages": [
                _FakeToolMsg(
                    content="tool result", id="t-1", name="search", tool_call_id="tc-1"
                )
            ]
        }
        resp = AgentResponse.from_raw(raw)
        assert isinstance(resp.messages[0], ToolMessage)
        assert resp.messages[0].content == "tool result"
        assert resp.messages[0].name == "search"
        assert resp.messages[0].tool_call_id == "tc-1"

    def test_object_ai_with_tool_calls_list(self):
        tc_dict = {
            "name": "search",
            "args": {"q": "shoes"},
            "id": "tc-1",
            "type": "tool_call",
        }
        raw = {"messages": [_FakeAIMsg(content="", tool_calls=[tc_dict])]}
        resp = AgentResponse.from_raw(raw)
        ai = resp.messages[0]
        assert isinstance(ai, AIMessage)
        # tool_calls were passed as raw dicts; AIMessage accepts them via extra="allow" or
        # they must be coerced – either way the message is created
        assert len(ai.tool_calls) == 1

    def test_mixed_dict_and_object_messages(self):
        raw = {
            "messages": [
                {"role": "human", "content": "Hi"},
                _FakeAIMsg(content="Response"),
            ]
        }
        resp = AgentResponse.from_raw(raw)
        assert isinstance(resp.messages[0], HumanMessage)
        assert isinstance(resp.messages[1], AIMessage)


# ===========================================================================
# _langchain_msg_to_dict
# ===========================================================================


class TestLangchainMsgToDict:
    def test_dict_passthrough(self):
        d = {"role": "human", "content": "Hi"}
        result = _langchain_msg_to_dict(d)
        assert result is d  # same object, not a copy

    def test_human_like_object(self):
        msg = _FakeHumanMsg(content="Hello", id="h-1")
        result = _langchain_msg_to_dict(msg)
        assert result["content"] == "Hello"
        assert result["id"] == "h-1"
        assert result["role"] == MessageRole.HUMAN

    def test_tool_like_object_role_and_extra_fields(self):
        msg = _FakeToolMsg(
            content="result", id="t-1", name="search_tool", tool_call_id="tc-99"
        )
        result = _langchain_msg_to_dict(msg)
        assert result["content"] == "result"
        assert result["id"] == "t-1"
        assert result["role"] == MessageRole.TOOL
        assert result["name"] == "search_tool"
        assert result["tool_call_id"] == "tc-99"

    def test_ai_like_object_role_and_extra_fields(self):
        msg = _FakeAIMsg(content="response", id="a-1")
        result = _langchain_msg_to_dict(msg)
        assert result["content"] == "response"
        assert result["id"] == "a-1"
        assert result["role"] == MessageRole.AI
        assert result["tool_calls"] == []
        assert result["invalid_tool_calls"] == []
        assert result["response_metadata"] == {}
        # _FakeAIMsg defaults to a complete dict so the result reflects that
        assert result["usage_metadata"] == {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    def test_ai_response_metadata_none_becomes_empty_dict(self):
        msg = _FakeAIMsg(content="hi")
        msg.response_metadata = None  # falsy -> or {} kicks in
        result = _langchain_msg_to_dict(msg)
        assert result["response_metadata"] == {}

    def test_ai_usage_metadata_none_becomes_empty_dict(self):
        msg = _FakeAIMsg(content="hi")
        msg.usage_metadata = None
        result = _langchain_msg_to_dict(msg)
        assert result["usage_metadata"] == {}

    def test_ai_with_tool_calls_preserved(self):
        tc = {"name": "search", "args": {}, "id": "tc-1", "type": "tool_call"}
        msg = _FakeAIMsg(content="", tool_calls=[tc])
        result = _langchain_msg_to_dict(msg)
        assert result["tool_calls"] == [tc]

    def test_missing_id_attribute_defaults_none(self):
        class NoId:
            content = "hello"

        result = _langchain_msg_to_dict(NoId())
        assert result["id"] is None

    def test_missing_content_attribute_defaults_empty_string(self):
        class NoContent:
            pass

        result = _langchain_msg_to_dict(NoContent())
        assert result["content"] == ""


# ===========================================================================
# _infer_role
# ===========================================================================


class TestInferRole:
    def test_human_in_class_name(self):
        class HumanMessage:
            pass

        assert _infer_role(HumanMessage()) == MessageRole.HUMAN

    def test_tool_in_class_name(self):
        class ToolMessage:
            pass

        assert _infer_role(ToolMessage()) == MessageRole.TOOL

    def test_ai_class_name_defaults_to_ai(self):
        class AIMessage:
            pass

        assert _infer_role(AIMessage()) == MessageRole.AI

    def test_unrecognised_class_name_defaults_to_ai(self):
        class SomethingElse:
            pass

        assert _infer_role(SomethingElse()) == MessageRole.AI

    def test_dict_object_defaults_to_ai(self):
        # type({}).__name__ == "dict" – no "Human" or "Tool" substring
        assert _infer_role({}) == MessageRole.AI

    def test_human_substring_in_longer_name_matches(self):
        class NotAHumanMessage:
            pass

        # "NotAHumanMessage" contains "Human"
        assert _infer_role(NotAHumanMessage()) == MessageRole.HUMAN

    def test_tool_substring_in_longer_name_matches(self):
        class MyToolHandler:
            pass

        assert _infer_role(MyToolHandler()) == MessageRole.TOOL

    def test_human_takes_precedence_over_tool_when_both_present(self):
        # "HumanToolMessage" contains both "Human" and "Tool";
        # the if/elif chain checks Human first so it returns HUMAN.
        class HumanToolMessage:
            pass

        assert _infer_role(HumanToolMessage()) == MessageRole.HUMAN


# ===========================================================================
# Integration: full conversation round-trip
# ===========================================================================


class TestAgentResponseIntegration:
    def test_full_conversation_with_products_and_final_answer(self):
        raw = {
            "messages": [
                {"role": "human", "content": "Show me headphones"},
                {"role": "ai", "content": "Let me search for you", "tool_calls": []},
                {"role": "tool", "content": MULTI_PRODUCT_CONTENT},
                {"role": "ai", "content": "Here are two headphone options"},
            ]
        }
        resp = AgentResponse.from_raw(raw)

        assert len(resp.human_messages) == 1
        assert len(resp.ai_messages) == 2
        assert len(resp.tool_messages) == 1
        assert len(resp.all_products) == 2
        assert resp.final_answer == "Here are two headphone options"

    def test_final_answer_none_when_only_tool_messages(self):
        raw = {
            "messages": [
                {"role": "human", "content": "Hi"},
                {"role": "tool", "content": ""},
            ]
        }
        resp = AgentResponse.from_raw(raw)
        assert resp.final_answer is None

    def test_final_answer_uses_last_ai_without_tool_calls(self):
        tc = ToolCall(name="search", args={}, id="tc-1")
        resp = AgentResponse(
            messages=[
                HumanMessage(content="Find shoes"),
                AIMessage(content="", tool_calls=[tc]),
                ToolMessage(content=""),
                AIMessage(content="I found some shoes!"),
            ]
        )
        assert resp.final_answer == "I found some shoes!"

    def test_all_products_from_object_based_tool_message(self):
        raw = {"messages": [_FakeToolMsg(content=MULTI_PRODUCT_CONTENT)]}
        resp = AgentResponse.from_raw(raw)
        assert len(resp.all_products) == 2

    def test_product_fields_accessible_via_all_products(self):
        resp = AgentResponse(messages=[ToolMessage(content=SINGLE_PRODUCT_BLOCK)])
        p = resp.all_products[0]
        assert p.name == "Test Widget"
        assert p.price == 29.99
        assert p.in_stock is True
        assert p.tags == ["gadget", "widget"]
