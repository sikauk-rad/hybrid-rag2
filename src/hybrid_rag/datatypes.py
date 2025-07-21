from typing import Literal, Self, TypedDict
from datetime import date, datetime
from pydantic import BaseModel, ConfigDict
from pydantic_core import from_json
from abc import ABC
from enum import Enum


class Role(str, Enum):
    SYSTEM = 'system'
    ASSISTANT = 'assistant'
    USER = 'user'


class BaseAzureModelDetails(ABC, BaseModel):

    model_config = ConfigDict(
        extra = 'allow',
        frozen = True,
        use_enum_values = True,
        validate_assignment = True,
        strict = True,
        protected_namespaces = (),
    )

    model_name: str
    base_model_name: str
    model_version: str | None
    azure_endpoint: str
    azure_version: str
    api_key: str
    function: Literal['chat', 'embedding', 'cross-encoding', 'reasoning']


    @classmethod
    def from_json(
        cls,
        json_string: str,
    ) -> Self:

        model_dict = from_json(
            data = json_string,
        )
        return cls.from_dict(model_dict)


    @classmethod
    def from_dict(
        cls,
        model_dict: dict,
    ) -> Self:

        if isinstance(model_dict.get('knowledge_cutoff_date'), str):
            model_dict = model_dict | {
                'knowledge_cutoff_date': datetime.fromisoformat(
                    model_dict['knowledge_cutoff_date']
                ),
            }
        return cls(**model_dict)


    def to_json(
        self,
    ) -> str:

        return self.model_dump_json(
            indent = None,
            exclude = None,
            exclude_defaults = False,
            exclude_none = False,
        )


class AzureChatModelDetails(BaseAzureModelDetails):
    model_name: str
    base_model_name: str
    knowledge_cutoff_date: date | datetime | None
    model_version: str | None
    tokeniser: str
    tokeniser_type: Literal['tiktoken', 'HuggingFace']
    token_input_limit: int
    token_output_limit: int | None
    azure_endpoint: str
    azure_version: str
    api_key: str
    supports_structured: bool | None
    function: Literal['chat'] = 'chat'


class AzureEmbeddingModelDetails(BaseAzureModelDetails):
    model_name: str
    base_model_name: str
    model_version: str | None
    tokeniser: str | None
    tokeniser_type: Literal['tiktoken', 'HuggingFace'] | None
    token_input_limit: int
    azure_endpoint: str
    azure_version: str
    api_key: str
    function: Literal['embedding'] = 'embedding'


class AzureCrossEncoderModelDetails(BaseAzureModelDetails):
    model_name: str
    base_model_name: str
    model_version: str | None
    tokeniser: str | None
    tokeniser_type: Literal['tiktoken', 'HuggingFace'] | None
    token_input_limit: int
    azure_endpoint: str
    azure_version: str
    api_key: str
    function: Literal['cross-encoder'] = 'cross-encoder'


class AzureMessageType(TypedDict, total = True):
    role: Role
    content: str


class AzureMessageCountType(TypedDict, total = True):
    role: Role
    content: str
    tokens: int