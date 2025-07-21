from pathlib import Path
from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI
from openai._exceptions import RateLimitError, BadRequestError
from beartype import beartype
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
from asyncio import sleep as asleep
from asyncio import Semaphore
from time import sleep
from numbers import Number
import numpy as np
from .base import EmbeddingModelInterface, ChatModelInterface, TokeniserInterface
from .tokeniser_interfaces import TestTokeniserInterface
from .datatypes import AzureMessageCountType, AzureMessageType
from .utilities import get_allowed_history
from .text_transformers.embedding_transformers import EmbeddingCache
from datetime import date, datetime
from collections.abc import Sequence
from numpy.typing import NDArray
from pydantic import BaseModel


@beartype
class AzureEmbeddingModelInterface(EmbeddingModelInterface):

    def __init__(
        self,
        sync_client: OpenAI | AzureOpenAI,
        async_client: AsyncOpenAI | AsyncAzureOpenAI,
        model_name: str,
        cache: EmbeddingCache,
        base_model_name: str,
        max_concurrent_requests: int = 100,
    ) -> None:

        self.sync_client = sync_client
        self.async_client = async_client
        self.cache = cache
        self.model_name = model_name
        self.base_model_name = base_model_name
        self.bad_requests = []
        self.function = 'embedding'
        self.max_concurrent_requests = max_concurrent_requests


    def update_cache(
        self,
        new_items: dict[str, list[float]],
    ) -> None:

        self.cache.update(new_items)


    def transform(
        self,
        text: str,
        n_retries: int = 100,
    ) -> list[float]:

        embedding = self.cache.retrieve(text)
        if embedding:
            return embedding
        
        for _ in range(n_retries):
            try:
                embedding = self.sync_client.embeddings.create(
                    model = self.model_name,
                    input = text,
                ).data[0].embedding
                self.cache.add(text, embedding)
                break
            except RateLimitError:
                sleep(1)
            except BadRequestError:
                self.bad_requests.append(text)
                break
        else:
            return []

        return embedding
    

    async def atransform(
        self,
        text: str,
        n_retries: int = 100,
    ) -> list[float]:

        embedding = self.cache.retrieve(text)
        if embedding:
            return embedding

        for _ in range(n_retries):
            try:
                embedding = (await self.async_client.embeddings.create(
                    model = self.model_name,
                    input = text,
                )).data[0].embedding
                self.cache.add(text, embedding)
                break
            except RateLimitError:
                await asleep(1)
            except BadRequestError:
                self.bad_requests.append(text)
                break

        else:
            return []

        return embedding


    def transform_multiple(
        self,
        texts: Sequence[str],
        n_retries: int = 100,
        save_path: Path | None = None,
        fail_on_overwrite: bool = True,
    ) -> list[list[float]]:
    
        try:
            embeddings = [self.transform(
                text = text,
                n_retries = n_retries,
            ) for text in tqdm(
                texts,
                position = 0,
                leave = True,
                desc = f'embedding with {self.base_model_name}'
            )]

        finally:
            if save_path:
                self.cache.save(save_path, fail_on_overwrite)

        return embeddings


    async def atransform_multiple(
        self,
        texts: Sequence[str],
        n_retries: int = 100,
        save_path: Path | None = None,
        fail_on_overwrite: bool = True,
    ) -> list[list[float]]:

        try:
            async with Semaphore(self.max_concurrent_requests):
                coroutines = [self.atransform(
                    text = text,
                    n_retries = n_retries,
                ) for text in texts]
                embeddings = await tqdm_asyncio.gather(
                    *coroutines,
                    position = 0,
                    leave = True,
                    desc = f'embedding with {self.base_model_name}'
                )

        finally:
            if save_path:
                self.cache.save(save_path, fail_on_overwrite)

        return embeddings


@beartype
class AzureChatModelInterface(ChatModelInterface):

    def __init__(
        self,
        sync_client: OpenAI | AzureOpenAI,
        async_client: AsyncOpenAI | AsyncAzureOpenAI,
        model_name: str,
        base_model_name: str,
        tokeniser: TokeniserInterface,
        token_input_limit: int,
        token_output_limit: int | None = None,
        knowledge_cutoff_date: date | datetime | None = None,
        supports_structured: bool = False,
    ) -> None:

        self.sync_client = sync_client
        self.async_client = async_client
        self.model_name = model_name
        self.tokeniser = tokeniser
        self.token_input_limit = token_input_limit
        self.token_output_limit = token_output_limit
        self.base_model_name = base_model_name
        self.knowledge_cutoff_date = knowledge_cutoff_date
        self.function = 'chat'
        self.chat_parameters = {
            'model': self.model_name,
            'n': 1,
        }
        self.supports_structured = supports_structured


    def trim(
        self,
        messages: Sequence[AzureMessageCountType],
        message_preservation_indices: Sequence[int] | None = None,
        custom_token_limit: int | None = None,
    ) -> list[AzureMessageCountType]:

        return get_allowed_history(
            messages,
            message_preservation_indices = message_preservation_indices,
            token_limit = self.token_input_limit if custom_token_limit is None else custom_token_limit,
        )


    def respond(
        self,
        messages: Sequence[AzureMessageType],
        temperature: Number = 0,
        return_token_count: bool = False,
    ) -> tuple[str, int] | str:

        response = self.sync_client.chat.completions.create(
            **self.chat_parameters,
            temperature = temperature,
            messages = messages,
        )
        answer = response.choices[0].message.content
        return (answer, response.usage.completion_tokens) if return_token_count else answer


    async def arespond(
        self,
        messages: Sequence[AzureMessageType],
        temperature: Number = 0,
        return_token_count: bool = False,
    ) -> tuple[str, int] | str:

        response = await self.async_client.chat.completions.create(
            **self.chat_parameters,
            temperature = temperature,
            messages = messages,
        )
        answer = response.choices[0].message.content
        return (answer, response.usage.completion_tokens) if return_token_count else answer


    def trim_and_respond(
        self,
        messages: Sequence[AzureMessageCountType],
        temperature: Number = 0,
        return_token_count: bool = False,
        message_preservation_indices: Sequence[int] | None = None,
        custom_token_limit: int | None = None,
    ) -> tuple[str, int] | str:

        return self.respond(
            messages = self.trim(
                messages = messages, 
                message_preservation_indices = message_preservation_indices, 
                custom_token_limit = custom_token_limit,
            ),
            temperature = temperature,
            return_token_count = return_token_count,
        )


    async def atrim_and_respond(
        self,
        messages: Sequence[AzureMessageCountType],
        temperature: Number = 0,
        return_token_count: bool = False,
        message_preservation_indices: Sequence[int] | None = None,
        custom_token_limit: int | None = None,
    ) -> tuple[str, int] | str:

        return await self.arespond(
            messages = self.trim(
                messages = messages, 
                message_preservation_indices = message_preservation_indices, 
                custom_token_limit = custom_token_limit,
            ),
            temperature = temperature,
            return_token_count = return_token_count,
        )


    def respond_structured(
        self,
        messages: Sequence[AzureMessageType],
        response_format: type[BaseModel],
        temperature: Number = 0,
        return_token_count: bool = False,
    ) -> tuple[BaseModel, int] | BaseModel:

        if not self.supports_structured:
            raise ValueError('model does not support structured outputs.')

        response = self.sync_client.beta.chat.completions.parse(
            **self.chat_parameters,
            temperature = temperature,
            messages = messages,
            response_format = response_format,
        )

        out = response_format.model_validate_json(response.choices[0].message.content)
        return (out, response.usage.completion_tokens) if return_token_count else out


    async def arespond_structured(
        self,
        messages: Sequence[AzureMessageType],
        response_format: type[BaseModel],
        temperature: Number = 0,
        return_token_count: bool = False,
    ) -> tuple[BaseModel, int] | BaseModel:

        if not self.supports_structured:
            raise ValueError('model does not support structured outputs.')

        response = await self.async_client.beta.chat.completions.parse(
            **self.chat_parameters,
            temperature = temperature,
            messages = messages,
            response_format = response_format,
        )

        out = response_format.model_validate_json(response.choices[0].message.content)
        return (out, response.usage.completion_tokens) if return_token_count else out


    def trim_and_respond_structured(
        self,
        messages: Sequence[AzureMessageCountType],
        response_format: type[BaseModel],
        temperature: Number = 0,
        return_token_count: bool = False,
        message_preservation_indices: Sequence[int] | None = None,
        custom_token_limit: int | None = None,
    ) -> tuple[BaseModel, int] | BaseModel:

        return self.respond_structured(
            messages = self.trim(
                messages = messages, 
                message_preservation_indices = message_preservation_indices, 
                custom_token_limit = custom_token_limit,
            ),
            response_format = response_format,
            temperature = temperature,
            return_token_count = return_token_count,
        )


    async def atrim_and_respond_structured(
        self,
        messages: Sequence[AzureMessageCountType],
        response_format: type[BaseModel],
        temperature: Number = 0,
        return_token_count: bool = False,
        message_preservation_indices: Sequence[int] | None = None,
        custom_token_limit: int | None = None,
    ) -> tuple[BaseModel, int] | BaseModel:

        return await self.arespond_structured(
            messages = self.trim(
                messages = messages, 
                message_preservation_indices = message_preservation_indices, 
                custom_token_limit = custom_token_limit,
            ),
            response_format = response_format,
            temperature = temperature,
            return_token_count = return_token_count,
        )


@beartype
class TestEmbeddingModelInterface(EmbeddingModelInterface):

    def __init__(
        self,
    ) -> None:

        self.rng = np.random.default_rng()


    def transform(
        self,
        text: str
    ) -> NDArray[np.float64]:

        return self.rng.uniform(low = 0, high = 1, size = 100, dtype = 'float64')


    async def atransform(
        self,
        text: str,
    ) -> NDArray[np.float64]:

        return self.transform(text)


    def transform_multiple(
        self,
        texts: Sequence[str],
    ) -> NDArray[np.float64]:

        return self.rng.uniform(low = 0, high = 1, size = (len(texts), 100), dtype = 'float64')


    async def atransform_multiple(
        self,
        texts: Sequence[str],
    ) -> NDArray[np.float64]:

        return self.transform_multiple(texts)


class TestChatModelInterface(ChatModelInterface):

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:

        self.model_name = 'test'
        self.token_input_limit = 1_000_000
        self.base_model_name = 'test'
        self.tokeniser = TestTokeniserInterface()
        self.token_input_limit = 1_000_000
        self.knowledge_cutoff_date = None


    def respond(
        self,
        messages: Sequence[AzureMessageType | AzureMessageCountType],
        return_token_count: bool = False,
        *args,
        **kwargs,
    ) -> str | tuple[str, int]:

        message = messages[-1]['content']
        if return_token_count:
            return (message, 1)
        else:
            return message


    async def arespond(
        self,
        messages: Sequence[AzureMessageType | AzureMessageCountType],
        return_token_count: bool = False,
        *args,
        **kwargs,
    ) -> str:

        return self.respond(messages=messages, return_token_count=return_token_count,*args, **kwargs)