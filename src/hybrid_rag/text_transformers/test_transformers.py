from collections.abc import Sequence
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import NotFittedError

from ..base import TextTransformer


class TestTextTransformer(TextTransformer):

    def __init__(
        self,
        autogenerate_texts: bool = True,
        *args,
        **kwargs,
    ) -> None:

        if not autogenerate_texts:
            self._is_fit = False
            return
        self._is_fit = True
        self.fit(['here', 'is a random set', 'of texts for scoring!'])


    def fit(
        self,
        texts: Sequence[str],
        *args,
        **kwargs,
    ) -> Self:

        self.texts = texts
        self.documents = np.fromiter(
            map(len, texts),
            dtype = 'float64',
            count = len(self.texts),
        )
        self._is_fit = True
        return self


    async def afit(
        self,
        texts: Sequence[str],
        *args,
        **kwargs,
    ) -> Self:

        self.fit(texts)
        return self


    def fit_transform(
        self,
        texts: Sequence[str],
        *args,
        **kwargs,
    ):

        self.fit(texts)
        return self.documents


    async def afit_transform(
        self,
        texts: Sequence[str],
        *args,
        **kwargs,
    ):

        self.fit(texts)
        return self.documents


    def _check_fit(
        self,
    ) -> None:

        if not self._is_fit:
            raise NotFittedError('fit or fit_transform must be called first.')


    def transform(
        self,
        text: str,
        *args,
        **kwargs,
    ) -> int:

        return len(text)


    async def atransform(
        self,
        text: str,
        *args,
        **kwargs,
    ) -> int:

        return self.transform(text)
        ...


    def transform_multiple(
        self,
        texts: Sequence[str],
        *args,
        **kwargs,
    ) -> NDArray[np.int64]:

        return np.fromiter(
            map(len, texts),
            dtype = 'float64',
            count = len(texts),
        )


    async def atransform_multiple(
        self,
        texts: list[str],
        *args,
        **kwargs,
    ):

        return self.transform_multiple(texts = texts)


    def score(
        self,
        text: str,
        document_indices: Sequence[int] | None = None,
        documents: NDArray[np.integer] | None = None,
        *args,
        **kwargs,
    ) -> NDArray[np.float64]:

        documents = documents or self.documents
        if document_indices:
            documents = documents[document_indices]
    
        document = self.transform(text)
        return (documents - document)**2 / document


    async def ascore(
        self,
        text: str,
        document_indices: Sequence[int] | None = None,
        documents: NDArray[np.integer] | None = None,
        *args,
        **kwargs,
    ) -> NDArray[np.float64]:

        return self.score(
            text=text, 
            document_indices=document_indices,
            documents = documents,
        )


    @classmethod
    def load_from_file(
        cls,
        file_path: Path,
        *args,
        **kwargs,
    ) -> Self:

        raise NotImplementedError('sorry why would you ever need this?')


    def save_to_file(
        self,
        save_path: Path,
        *args,
        **kwargs,
    ):

        raise NotImplementedError('sorry why would you ever need this?')