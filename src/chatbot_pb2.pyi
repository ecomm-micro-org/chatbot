from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ChatRequest(_message.Message):
    __slots__ = ("message",)
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    message: str
    def __init__(self, message: _Optional[str] = ...) -> None: ...

class ChatResponse(_message.Message):
    __slots__ = ("response",)
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    response: str
    def __init__(self, response: _Optional[str] = ...) -> None: ...

class GenerateDescriptionRequest(_message.Message):
    __slots__ = ("product_name", "product_price", "product_category", "tags")
    PRODUCT_NAME_FIELD_NUMBER: _ClassVar[int]
    PRODUCT_PRICE_FIELD_NUMBER: _ClassVar[int]
    PRODUCT_CATEGORY_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    product_name: str
    product_price: float
    product_category: str
    tags: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, product_name: _Optional[str] = ..., product_price: _Optional[float] = ..., product_category: _Optional[str] = ..., tags: _Optional[_Iterable[str]] = ...) -> None: ...

class GenerateDescriptionResponse(_message.Message):
    __slots__ = ("response",)
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    response: str
    def __init__(self, response: _Optional[str] = ...) -> None: ...
