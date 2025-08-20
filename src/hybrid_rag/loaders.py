from collections.abc import Iterable
from itertools import groupby
from operator import attrgetter
from typing import Literal

from beartype import beartype
from openai import AzureOpenAI, AsyncAzureOpenAI, OpenAI, AsyncOpenAI

from .azure_interfaces import AzureChatModelInterface, AzureEmbeddingModelInterface
from .base import TokeniserInterface
from .datatypes import BaseAzureModelDetails, AzureChatModelDetails, AzureEmbeddingModelDetails
from .text_transformers.embedding_transformers import EmbeddingCache
from .tokeniser_interfaces import OpenAITokeniserInterface, HuggingFaceTokeniserInterface




@beartype
def load_tokeniser(
    tokeniser_type: Literal['tiktoken', 'HuggingFace'],
    tokeniser: str,
    huggingface_token: str | None = None,
) -> TokeniserInterface:

    match tokeniser_type:
        case 'tiktoken':
            return OpenAITokeniserInterface(
                tokeniser
            )
        case 'HuggingFace':
            return HuggingFaceTokeniserInterface(
                tokeniser,
                token = huggingface_token,
            )


@beartype
def load_azure_clients(
    api_key: str,
    api_version: str | None = None,
    azure_endpoint: str | None = None,
    max_retries: int = 1000,
) -> tuple[AzureOpenAI, AsyncAzureOpenAI]:

    """
    Load Azure OpenAI clients for synchronous and asynchronous operations.

    Args:
        api_key (str): The API key for authenticating with Azure OpenAI.
        api_version (str | None): The version of the Azure OpenAI API. Must be provided.
        azure_endpoint (str | None): The endpoint URL for the Azure OpenAI service. Must be provided.
        max_retries (int): The maximum number of retries for API calls (default is 1000).

    Returns:
        tuple[AzureOpenAI, AsyncAzureOpenAI]: A tuple containing the synchronous and asynchronous Azure OpenAI clients.

    Raises:
        TypeError: If `api_version` or `azure_endpoint` is not provided.
    """

    if not api_version or not azure_endpoint:
        raise TypeError(
            'azure_version and azure_endpoint must be provided.'
        )

    return (
        AzureOpenAI(
            api_key = api_key,
            azure_endpoint = azure_endpoint,
            api_version = api_version,
            max_retries = max_retries,
        ),
        AsyncAzureOpenAI(
            api_key = api_key,
            azure_endpoint = azure_endpoint,
            api_version = api_version,
            max_retries = max_retries,
        ),
    )


@beartype
def load_openai_clients(
    api_key: str,
    max_retries: int = 1000,
) -> tuple[OpenAI, AsyncOpenAI]:

    """
    Load OpenAI clients for synchronous and asynchronous operations.

    Args:
        api_key (str): The API key for authenticating with OpenAI.
        max_retries (int): The maximum number of retries for API calls (default is 1000).

    Returns:
        tuple[OpenAI, AsyncOpenAI]: A tuple containing the synchronous and asynchronous OpenAI clients.
    """

    return (
        OpenAI(
            api_key = api_key,
            max_retries = max_retries,
        ),
        AsyncOpenAI(
            api_key = api_key,
            max_retries = max_retries,
        )
    )


def load_azure_model_details_from_dicts(
    model_details: Iterable[dict],
) -> list[BaseAzureModelDetails]:

    return [BaseAzureModelDetails.from_dict_factory(
        model_details = model_detail
    ) for model_detail in model_details]


@beartype
def load_model_from_azure_model_details(
    sync_client: AzureOpenAI,
    async_client: AsyncAzureOpenAI,
    azure_model_details: BaseAzureModelDetails,
    huggingface_token: str | None = None,
    max_cache_size: int = 100_000,
) -> AzureChatModelInterface | AzureEmbeddingModelInterface:

    match azure_model_details.function:
        case 'embedding':
            model = AzureEmbeddingModelInterface(
                sync_client = sync_client,
                async_client = async_client,
                model_name = azure_model_details.model_name,
                base_model_name = azure_model_details.base_model_name,
                cache = EmbeddingCache(
                    cache = {},
                    max_size = max_cache_size,
                    model_name = azure_model_details.base_model_name,
                ),
            )
        case 'chat':
            model = AzureChatModelInterface(
                sync_client = sync_client,
                async_client = async_client,
                model_name = azure_model_details.model_name,
                base_model_name = azure_model_details.base_model_name,
                tokeniser = load_tokeniser(
                    tokeniser = azure_model_details.tokeniser,
                    tokeniser_type = azure_model_details.tokeniser_type,
                    huggingface_token = huggingface_token,
                ),
                token_input_limit = azure_model_details.token_input_limit,
                token_output_limit = azure_model_details.token_output_limit,
                knowledge_cutoff_date = azure_model_details.knowledge_cutoff_date,
                supports_structured = azure_model_details.supports_structured,
            )
        case _:
            raise ValueError('unrecognised function.')
    
    return model


@beartype
def load_clients_and_models_from_azure_model_details(
    azure_model_details: Iterable[BaseAzureModelDetails],
    huggingface_token: str | None = None,
) -> tuple[list[AzureOpenAI], list[AsyncAzureOpenAI], list[AzureEmbeddingModelInterface | AzureChatModelInterface]]:

    sync_clients, async_clients, azure_models = [], [], []
    grouping_key = attrgetter('azure_endpoint', 'azure_version', 'api_key')
    for (endpoint, version, api_key), model_details_iter in groupby(
        sorted(azure_model_details, key = grouping_key),
        key = grouping_key,
    ):
        sync_client, async_client = load_azure_clients(
            api_key = api_key,
            azure_endpoint = endpoint,
            api_version = version,
        )
        sync_clients.append(sync_client)
        async_clients.append(async_client)
        for model_details in model_details_iter:
            azure_models.append(load_model_from_azure_model_details(
                sync_client = sync_client,
                async_client = async_client,
                azure_model_details = model_details,
                huggingface_token = huggingface_token,
            ))

    return (sync_clients, async_clients, azure_models)


@beartype
def load_clients_and_models_from_dicts(
    model_details: Iterable[dict],
    huggingface_token: str | None = None,
) -> tuple[list[AzureOpenAI], list[AsyncAzureOpenAI], list[AzureEmbeddingModelInterface | AzureChatModelInterface]]:

    azure_model_details = load_azure_model_details_from_dicts(model_details)
    return load_clients_and_models_from_azure_model_details(
        azure_model_details=azure_model_details,
        huggingface_token = huggingface_token,
    )