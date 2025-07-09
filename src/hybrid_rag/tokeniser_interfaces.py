from tiktoken import get_encoding
from typing import Literal, Any
from beartype import beartype
import numpy as np
from collections.abc import Sequence
from numpy.typing import NDArray
from .base import TokeniserInterface
from tiktoken import get_encoding
from typing import Literal, Any
from beartype import beartype
import numpy as np
from collections.abc import Sequence
from numpy.typing import NDArray
from hybrid_rag.base import TokeniserInterface


@beartype
class OpenAITokeniserInterface(TokeniserInterface):

    __slots__ = ('encoding', 'tokeniser')

    def __init__(
        self,
        encoding: Literal[
            'o200k_base', 
            'cl100k_base', 
            'p50k_base', 
            'r50k_base', 
            'p50k_edit', 
            'gpt2',
        ],
    ) -> None:

        self.encoding = encoding
        self.tokeniser = get_encoding(encoding)


    @property
    def tokeniser_id(
        self,
    ) -> str:

        return self.encoding


    def tokenise(
        self,
        text: str,
    ) -> list[int]:

        return self.tokeniser.encode(text)


    def tokenise_multiple(
        self,
        texts: Sequence[str],
    ) -> NDArray[np.integer] | list[list[int]]:

        return self.tokeniser.encode_batch(texts)


    def get_token_length(
        self,
        text: str,
    ) -> int:

        return len(self.tokeniser.encode(text))


    def get_token_lengths(
        self,
        texts: Sequence[str],
    ) -> list[int]:

        return [*map(len, self.tokeniser.encode_batch(texts))]


    def __eq__(
        self,
        other: Any,
    ) -> bool:

        if isinstance(other, OpenAITokeniserInterface):
            return self.encoding == other.encoding
        else:
            return False


@beartype
class HuggingFaceTokeniserInterface(TokeniserInterface):

    __slots__ = ('model_id', 'tokeniser')

    def __init__(
        self,
        model_id: str,
        token: str | None,
    ) -> None:

        from transformers import AutoTokenizer

        self.model_id = model_id
        self.tokeniser = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path = model_id,
            token = token,
        )


    @property
    def tokeniser_id(
        self,
    ) -> str:

        return self.model_id


    def tokenise(
        self,
        text: str,
    ) -> list[int]:

        return self.tokeniser.encode(text)


    def tokenise_multiple(
        self,
        texts: Sequence[str],
    ) -> NDArray[np.integer] | list[list[int]]:

        return [*map(self.tokeniser.encode, texts)]


    def get_token_length(
        self,
        text: str,
    ) -> int:

        return len(self.tokeniser.encode(text))


    def get_token_lengths(
        self,
        texts: Sequence[str],
    ) -> list[int]:

        return [len(self.tokeniser.encode(text)) for text in texts]


    def __eq__(
        self,
        other: Any,
    ) -> bool:

        if isinstance(other, HuggingFaceTokeniserInterface):
            return self.model_id == other.model_id
        else:
            return False


@beartype
class TestTokeniserInterface(TokeniserInterface):

    def __init__(
        self,
    ) -> None:

        self.model_id = 'test'


    def tokenise(
        self,
        text: str,
    ) -> list[int]:

        return [len(text)]


    def tokenise_multiple(
        self,
        texts: Sequence[str],
    ) -> list[list[int]]:

        return [[len(text)] for text in texts]


    def get_token_length(
        self,
        text: str,
    ) -> Literal[1]:

        return 1


    def get_token_lengths(
        self,
        texts: Sequence[str],
    ) -> list[Literal[1]]:
        ...


    @property
    def tokeniser_id(
        self,
    ) -> str:

        return self.model_id