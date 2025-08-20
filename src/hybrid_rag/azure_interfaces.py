from asyncio import Semaphore, create_task
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path
from typing import TypeVar

from beartype import beartype
from llm_utilities.datatypes import AzureMessageCountType, AzureMessageType
import numpy as np
from numpy.typing import NDArray
from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI
from openai._exceptions import RateLimitError, BadRequestError
from pydantic import BaseModel
import tenacity
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from .base import EmbeddingModelInterface, ChatModelInterface, TokeniserInterface
from .tokeniser_interfaces import TestTokeniserInterface
from .utilities import get_allowed_history
from .text_transformers.embedding_transformers import EmbeddingCache

T = TypeVar('T', bound = BaseModel)


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
        new_items: dict[str, Sequence[float]],
    ) -> None:

        self.cache.update(new_items)


    @tenacity.retry(
        stop = tenacity.stop_after_attempt(100) | tenacity.stop_after_delay(60) ,
        wait = tenacity.wait_exponential_jitter(initial = 1, max = 10, jitter = 0.5),
        retry = tenacity.retry_if_exception_type(RateLimitError),
    )
    def transform(
        self,
        text: str,
    ) -> list[float]:

        embedding = self.cache.retrieve(text)
        if embedding:
            return embedding

        try:
            embedding = self.sync_client.embeddings.create(
                model = self.model_name,
                input = text,
            ).data[0].embedding
        except BadRequestError:
            self.bad_requests.append(text)
            return []
        else:
            self.cache.add(text, embedding)

        return embedding
    

    @tenacity.retry(
        stop = tenacity.stop_after_attempt(100) | tenacity.stop_after_delay(60) ,
        wait = tenacity.wait_exponential_jitter(initial = 1, max = 10, jitter = 0.5),
        retry = tenacity.retry_if_exception_type(RateLimitError),
    )
    async def atransform(
        self,
        text: str,
        n_retries: int = 100,
    ) -> list[float]:

        embedding = self.cache.retrieve(text)
        if embedding:
            return embedding

        try:
            embedding = (await self.async_client.embeddings.create(
                model = self.model_name,
                input = text,
            )).data[0].embedding
        except BadRequestError:
            self.bad_requests.append(text)
            return []
        else:
            self.cache.add(text, embedding)

        return embedding


    def transform_multiple(
        self,
        texts: Sequence[str],
        save_path: Path | None = None,
        fail_on_overwrite: bool = True,
    ) -> list[list[float]]:
    
        try:
            embeddings = []
            for text in tqdm(
                texts,
                position = 0,
                leave = True,
                desc = f'embedding with {self.base_model_name}'
            ):
                try:
                    embeddings.append(self.transform(text = text))
                except tenacity.RetryError:
                    embeddings.append([])

        finally:
            if save_path:
                self.cache.save(save_path, fail_on_overwrite)

        return embeddings


    async def atransform_multiple(
        self,
        texts: Sequence[str],
        save_path: Path | None = None,
        fail_on_overwrite: bool = True,
    ) -> list[list[float]]:

        semaphore = Semaphore(self.max_concurrent_requests)
        async def sem_aware_transform(text) -> list[float]:
            async with semaphore:
                try:
                    return await self.atransform(text)
                except tenacity.RetryError:
                    return []

        tasks = [create_task(sem_aware_transform(text)) for text in texts]

        try:
            for coro in tqdm_asyncio.as_completed(
                tasks,
                total=len(tasks),
                desc=f'embedding with {self.base_model_name}',
                position=0,
                leave=True,
            ):
                result = await coro
        finally:
            if save_path:
                self.cache.save(save_path, fail_on_overwrite)

        return [task.result() for task in tasks]


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
        temperature: int | float = 0,
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
        temperature: int | float = 0,
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
        temperature: int | float = 0,
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
        temperature: int | float = 0,
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
        response_format: type[T],
        temperature: int | float = 0,
        return_token_count: bool = False,
    ) -> tuple[T, int] | T:

        if not self.supports_structured:
            raise ValueError('model does not support structured outputs.')

        response = self.sync_client.beta.chat.completions.parse(
            **self.chat_parameters,
            temperature = temperature,
            messages = messages,
            response_format = response_format,
        )

        out = response.choices[0].message.parsed
        if not out:
            raise ValueError('unsuccessful prompt.')
        return (out, response.usage.completion_tokens) if return_token_count else out


    async def arespond_structured(
        self,
        messages: Sequence[AzureMessageType],
        response_format: type[T],
        temperature: int | float = 0,
        return_token_count: bool = False,
    ) -> tuple[T, int] | T:

        if not self.supports_structured:
            raise ValueError('model does not support structured outputs.')

        response = await self.async_client.beta.chat.completions.parse(
            **self.chat_parameters,
            temperature = temperature,
            messages = messages,
            response_format = response_format,
        )

        out = response.choices[0].message.parsed
        if not out:
            raise ValueError('unsuccessful prompt.')
        else:
            return (out, response.usage.completion_tokens) if return_token_count else out


    def trim_and_respond_structured(
        self,
        messages: Sequence[AzureMessageCountType],
        response_format: type[T],
        temperature: int | float = 0,
        return_token_count: bool = False,
        message_preservation_indices: Sequence[int] | None = None,
        custom_token_limit: int | None = None,
    ) -> tuple[T, int] | T:

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
        response_format: type[T],
        temperature: int | float = 0,
        return_token_count: bool = False,
        message_preservation_indices: Sequence[int] | None = None,
        custom_token_limit: int | None = None,
    ) -> tuple[T, int] | T:

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

        return self.rng.uniform(low = 0., high = 1., size = 100, dtype = 'float64')


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
    ) -> str | tuple[str, int]:

        return self.respond(messages=messages, return_token_count=return_token_count,*args, **kwargs)