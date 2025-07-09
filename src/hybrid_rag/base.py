from typing import Self
from collections.abc import Iterable, Hashable, Sequence
from abc import ABC, abstractmethod
from numbers import Number
from pathlib import Path
from sklearn.exceptions import NotFittedError
from datetime import date, datetime


class TextTransformer(ABC):

    @abstractmethod
    def fit(
        self,
        texts: Sequence[str],
        *args,
        **kwargs,
    ) -> Self:
        ...


    @abstractmethod
    async def afit(
        self,
        texts: Sequence[str],
        *args,
        **kwargs,
    ) -> Self:
        ...


    @abstractmethod
    def fit_transform(
        self,
        texts: Sequence[str],
        *args,
        **kwargs,
    ):
        ...


    @abstractmethod
    async def afit_transform(
        self,
        texts: Sequence[str],
        *args,
        **kwargs,
    ):
        ...


    def _check_fit(
        self,
    ) -> None:

        if not self._is_fit:
            raise NotFittedError('fit or fit_transform must be called first.')


    @abstractmethod
    def transform(
        self,
        text: str,
        *args,
        **kwargs,
    ):
        ...


    @abstractmethod
    async def atransform(
        self,
        text: str,
        *args,
        **kwargs,
    ):
        ...


    @abstractmethod
    def transform_multiple(
        self,
        texts: Sequence[str],
        *args,
        **kwargs,
    ):
        ...


    @abstractmethod
    async def atransform_multiple(
        self,
        texts: Sequence[str],
        *args,
        **kwargs,
    ):
        ...


    @abstractmethod
    def score(
        self,
        text: str,
        document_indices: Sequence[int] | None,
        documents,
        *args,
        **kwargs,
    ) -> Iterable[Number]:
        ...


    @abstractmethod
    async def ascore(
        self,
        text: str,
        document_indices: Sequence[int] | None,
        documents,
        *args,
        **kwargs,
    ) -> Iterable[Number]:
        ...


    @classmethod
    @abstractmethod
    def load_from_file(
        cls,
        file_path: Path,
        *args,
        **kwargs,
    ) -> Self:
        ...


    @abstractmethod
    def save_to_file(
        self,
        save_path: Path,
        *args,
        **kwargs,
    ):
        ...


class TokeniserInterface(ABC):

    @abstractmethod
    def tokenise(
        self,
        text: str,
    ) -> Sequence[Number]:
        ...

    @abstractmethod
    def tokenise_multiple(
        self,
        texts: Sequence[str],
    ) -> Sequence[Sequence[Number]]:
        ...

    @abstractmethod
    def get_token_length(
        self,
        text: str,
    ) -> int:
        ...

    @abstractmethod
    def get_token_lengths(
        self,
        texts: Sequence[str],
    ) -> Sequence[int]:
        ...

    @property
    @abstractmethod
    def tokeniser_id(
        self,
    ) -> Hashable:
        ...

    def __hash__(
        self,
    ) -> int:

        return hash(self.tokeniser_id)


class ChatModelInterface(ABC):

    @abstractmethod
    def __init__(
        self,
        tokeniser: TokeniserInterface,
        token_input_limit: int,
        base_model_name: str | None = None,
        knowledge_cutoff_date: date | datetime | None = None,
        *args,
        **kwargs,
    ) -> None:
        ...

    @abstractmethod
    def respond(
        self,
        text: str,
        *args,
        **kwargs,
    ):
        ...

    @abstractmethod
    async def arespond(
        self,
        text: str,
        *args,
        **kwargs,
    ):
        ...


class EmbeddingModelInterface(ABC):

    @abstractmethod
    def transform():
        ...

    @abstractmethod
    async def atransform():
        ...

    @abstractmethod
    def transform_multiple():
        ...

    @abstractmethod
    async def atransform_multiple():
        ...