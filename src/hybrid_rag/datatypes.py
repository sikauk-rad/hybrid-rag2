from collections.abc import Iterable
from datetime import date
from enum import Enum
from typing import Literal, Self

import msgspec


class ModelFunction(str, Enum):
    CHAT = 'chat'
    EMBEDDING = 'embedding'
    CROSS_ENCODER = 'cross-encoder'


class BaseAzureModelDetails(msgspec.Struct, kw_only = True):

    _registry = {}
    model_name: str
    base_model_name: str
    tokeniser: str
    tokeniser_type: Literal['tiktoken', 'HuggingFace']
    token_input_limit: int
    azure_endpoint: str
    azure_version: str
    api_key: str
    function: ModelFunction

    @classmethod
    def register(cls, function: ModelFunction):
        def decorator(subclass):
            cls._registry[function] = subclass
            return subclass
        return decorator


    @classmethod
    def from_dict_factory(
        cls,
        model_details: dict,
    ) -> Self:

        if cls is not BaseAzureModelDetails:
            raise TypeError('from_dict_factory should only be called from the base class.')
        function = model_details.get('function')
        if function is None:
            raise KeyError('model_details does not contain function key.')
        try:
            model_function = ModelFunction(function)
        except ValueError:
            raise KeyError(f'model function {function} not a valid ModelFunction enum.')
        model_cls = cls._registry.get(function)
        if not model_cls:
            raise KeyError('model function not registered with BaseAzureModelDetails.')
        return model_cls.from_dict(model_details)


    @classmethod
    def from_dict(
        cls,
        dict: dict,
    ) -> Self:

        return msgspec.convert(
            obj = dict,
            type = cls,
        )


    @classmethod
    def from_dicts(
        cls,
        dicts: Iterable[dict],
    ) -> list[Self]:

        return msgspec.convert(
            obj = [*dicts],
            type = list[cls],
        )


    @classmethod
    def from_json(
        cls,
        json: bytes | str,
        multiple: bool = False,
    ) -> Self | list[Self]:

        return msgspec.json.decode(
            json, 
            type = list[cls] if multiple else cls,
        )


    def to_json(
        self,
    ) -> bytes:

        return msgspec.json.encode(self)


@BaseAzureModelDetails.register(function=ModelFunction.CHAT)
class AzureChatModelDetails(
    BaseAzureModelDetails, 
    kw_only = True,
    frozen = True,
):
    model_name: str
    base_model_name: str
    knowledge_cutoff_date: date | None
    model_version: str | None
    tokeniser: str
    tokeniser_type: Literal['tiktoken', 'HuggingFace']
    token_input_limit: int
    token_output_limit: int | None
    azure_endpoint: str
    azure_version: str
    api_key: str
    supports_structured: bool | None

    def __post_init__(
        self,
    ) -> None:

        msgspec.structs.force_setattr(
            self,
            'function',
            ModelFunction.CHAT,
        )


@BaseAzureModelDetails.register(function=ModelFunction.EMBEDDING)
class AzureEmbeddingModelDetails(
    BaseAzureModelDetails, 
    kw_only = True,
    frozen = True,
):
    model_name: str
    base_model_name: str
    model_version: str | None
    tokeniser: str | None
    tokeniser_type: Literal['tiktoken', 'HuggingFace'] | None
    token_input_limit: int
    azure_endpoint: str
    azure_version: str
    api_key: str

    def __post_init__(
        self,
    ) -> None:

        msgspec.structs.force_setattr(
            self,
            'function',
            ModelFunction.EMBEDDING,
        )


@BaseAzureModelDetails.register(function=ModelFunction.CROSS_ENCODER)
class AzureCrossEncoderModelDetails(
    BaseAzureModelDetails, 
    kw_only = True,
    frozen = True,
):
    model_name: str
    base_model_name: str
    model_version: str | None
    tokeniser: str | None
    tokeniser_type: Literal['tiktoken', 'HuggingFace'] | None
    token_input_limit: int
    azure_endpoint: str
    azure_version: str
    api_key: str

    def __post_init__(
        self,
    ) -> None:

        msgspec.structs.force_setattr(
            self,
            'function',
            ModelFunction.CROSS_ENCODER,
        )