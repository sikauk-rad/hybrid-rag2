from pathlib import Path
import polars as pl
import numpy as np
import orjson as json
from typing import Self
from beartype import beartype
from collections.abc import Iterable, Sequence
from sklearn.metrics.pairwise import cosine_similarity
from numbers import Number
from dataclasses import dataclass
from numpy.typing import NDArray
from cachetools import LFUCache
from ..base import TextTransformer, EmbeddingModelInterface
from ..exceptions import SizeError


@beartype
@dataclass
class EmbeddingCache:
    """
    A class for caching text embeddings with a least-frequently-used (LFU) eviction policy. 
    Stores embedding for a single model.

    This class allows storing, retrieving, and managing text embeddings.
    """

    cache: dict[str, Sequence[float]]
    max_size: int
    model_name: str

    def __post_init__(self) -> None:

        """
        Initializes the LFU cache and checks if the initial cache size exceeds the maximum allowed size.

        Raises:
            SizeError: If the initial cache size exceeds the maximum allowed size.
        """

        cache_size = len(self.cache)
        if self.max_size < cache_size:
            raise SizeError(
                f'Given cache size {cache_size} exceeds max_size {self.max_size}.'
            )

        cache = LFUCache(maxsize=self.max_size)
        cache.update(self.cache)
        self.cache = cache


    def add(
        self,
        text: str,
        embedding: Sequence[float] | NDArray[np.floating],
    ) -> None:

        """
        Adds a text embedding to the cache.

        Args:
            text (str): The text associated with the embedding.
            embedding (list[float] | NDArray[np.floating]): The embedding vector.
        """

        self.cache[text] = embedding


    def update(
        self,
        /,
        new_items: dict[str, Sequence[float]],
    ) -> None:

        self.cache.update(new_items)


    def remove(
        self,
        text: str,
    ) -> None:

        """
        Removes a specific text embedding from the cache.

        Args:
            text (str): The text associated with the embedding.
        """

        self.cache.popitem(text)


    def retrieve(
        self,
        text: str,
    ) -> list[float] | None:

        """
        Retrieves a text embedding from the cache.

        Args:
            text (str): The text associated with the embedding.

        Returns:
            list[float] | None: The embedding vector if found, otherwise None.
        """

        return self.cache.get(text, None)


    @classmethod
    def from_file(
        cls,
        file_path: Path,
        max_size: int,
        model_name: str,
        fast_load: bool = True,
    ) -> Self:

        """
        Creates an EmbeddingCache instance from a JSON file.

        Args:
            file_path (Path): Path to the JSON file.
            max_size (int): Maximum size of the cache.
            fast_load (bool, optional): Whether to use a fast loading method. Defaults to True.

        Returns:
            EmbeddingCache: An instance of EmbeddingCache.

        Raises:
            TypeError: If the file is not a JSON file.
        """

        if file_path.suffix.lower() != '.json':
            raise TypeError('file must be a JSON.')

        if fast_load:
            print('using fast loader.')
            raw_json = pl.read_json(file_path).to_dicts()[0]
        else:
            print('using slow loader.')
            with open(file_path, 'rb') as file:
                raw_json = json.loads(file.read())

        print('file loaded.')
        return cls(
            cache = raw_json,
            max_size=max_size,
            model_name = model_name,
        )


    def __len__(self) -> int:

        """
        Returns the current size of the cache.

        Returns:
            int: The number of items currently in the cache.
        """

        return self.cache.currsize


    def __getitem__(self, key: str) -> list[float]:

        """
        Retrieves an embedding from the cache using a key.

        Args:
            key (str): The text for which to retrieve the embedding.

        Returns:
            list[float]: The embedding vector associated with the key.
        """
        return self.cache[key]


    def save(
        self,
        save_path: Path,
        fail_on_overwrite: bool = True,
        append_model_name: bool = False,
    ) -> None:

        """
        Saves the current cache to a JSON file.

        Args:
            save_path (Path): Path to save the JSON file.
            fail_on_overwrite (bool, optional): Whether to fail if the file already exists. Defaults to True.

        Raises:
            TypeError: If the save_path is not a JSON file.
            FileExistsError: If fail_on_overwrite is True and the file already exists.
        """

        if save_path.suffix.lower() != '.json':
            raise TypeError('save_path must be a JSON file.')
        if fail_on_overwrite and save_path.exists():
            raise FileExistsError('save_path already exists.')
        if append_model_name:
            save_path = save_path.with_stem(f'{save_path.stem} {self.model_name}')

        with open(save_path, 'wb') as file:
            file.write(json.dumps({**self.cache}))


@beartype
class TextEmbedder(TextTransformer):

    def __init__(
        self,
        embedding_model: EmbeddingModelInterface,
    ) -> None:

        self.text_encodings = None
        self._is_fit = False
        self.embedding_model = embedding_model


    def fit_transform(
        self,
        texts: Sequence[str],
        n_retries: int = 100,
        save_path: Path | None = None,
        fail_on_overwrite: bool = True,
        as_polars: bool = False,
        precomputed_embeddings: NDArray[np.number] | None = None,
    ) -> NDArray[np.float32] | pl.DataFrame:

        if precomputed_embeddings is not None:
            array = precomputed_embeddings
        else:
            array = np.array(
                self.transform_multiple(
                    texts, 
                    n_retries = n_retries,
                    save_path = save_path,
                    fail_on_overwrite = fail_on_overwrite,
                ),
                dtype = 'float32',
            )
        self.text_encodings = pl.DataFrame(
            [texts,array],
            schema = ['content', 'embedding'],
        )
        self._is_fit = True
        return self.text_encodings if as_polars else array


    def fit(
        self,
        texts: Sequence[str],
        n_retries: int = 100,
        save_path: Path | None = None,
        fail_on_overwrite: bool = True,
        precomputed_embeddings: Iterable[Iterable[Number]] | None = None,
    ) -> Self:

        self.fit_transform(
            texts, 
            n_retries = n_retries,
            save_path = save_path,
            fail_on_overwrite = fail_on_overwrite,
            precomputed_embeddings = precomputed_embeddings,
        )
        return self


    async def afit_transform(
        self,
        texts: Sequence[str],
        n_retries: int = 100,
        save_path: Path | None = None,
        fail_on_overwrite: bool = True,
        as_polars: bool = False,
        precomputed_embeddings: NDArray[np.number] | None = None,
    ) -> NDArray[np.float32] | pl.DataFrame:

        if precomputed_embeddings is not None:
            array = precomputed_embeddings
        else:
            array = np.array(
                await self.atransform_multiple(
                    texts, 
                    n_retries = n_retries,
                    save_path = save_path,
                    fail_on_overwrite = fail_on_overwrite,
                ),
                dtype = 'float32',
            )
        self.text_encodings = pl.DataFrame(
            [texts, array],
            schema = ['content', 'embedding'],
        )
        self._is_fit = True
        return self.text_encodings if as_polars else array


    async def afit(
        self,
        texts: Sequence[str],
        n_retries: int = 100,
        save_path: Path | None = None,
        fail_on_overwrite: bool = True,
        precomputed_embeddings: NDArray[np.number] | None = None,
    ) -> Self:

        await self.afit_transform(
            texts, 
            n_retries = n_retries,
            save_path = save_path,
            fail_on_overwrite = fail_on_overwrite,
        )
        return self


    def transform(
        self,
        text: str,
        n_retries: int = 100,
    ) -> list[float]:

        return self.embedding_model.transform(
            text = text,
            n_retries = n_retries,
        )


    async def atransform(
        self,
        text: str,
        n_retries: int = 100,
    ) -> list[float]:

        return await self.embedding_model.atransform(
            text = text,
            n_retries = n_retries,
        )


    def transform_multiple(
        self, 
        texts: Sequence[str], 
        n_retries: int = 100,
        save_path: Path | None = None,
        fail_on_overwrite: bool = True,
    ) -> list[list[float]]:

        return self.embedding_model.transform_multiple(
            texts, 
            n_retries = n_retries,
            save_path = save_path,
            fail_on_overwrite = fail_on_overwrite,
        )


    async def atransform_multiple(
        self, 
        texts: Sequence[str], 
        n_retries: int = 100,
        save_path: Path | None = None,
        fail_on_overwrite: bool = True,
    ) -> list[list[float]]:

        return await self.embedding_model.atransform_multiple(
            texts, 
            n_retries = n_retries,
            save_path = save_path,
            fail_on_overwrite = fail_on_overwrite,
        )


    def score(
        self,
        text: str,
        n_retries: int = 100,
        document_indices: list[int] | NDArray[np.integer] | None = None,
        documents: Iterable[Iterable[Number]] | None = None,
    ) -> NDArray[np.floating]:

        embedding = self.transform(
            text, 
            n_retries = n_retries,
        )

        if documents is None:
            self._check_fit()
            documents = self.text_encodings['embedding']

        return cosine_similarity(
            documents if document_indices is None else documents[document_indices], 
            [embedding],
        )[:,0]


    async def ascore(
        self,
        text: str,
        n_retries: int = 100,
        document_indices: list[int] | NDArray[np.integer] | None = None,
        documents: Iterable[Iterable[Number]] | None = None,
    ) -> NDArray[np.floating]:

        embedding = await self.atransform(
            text, 
            n_retries = n_retries,
        )

        if documents is None:
            self._check_fit()
            documents = self.text_encodings['embedding']

        return cosine_similarity(
            documents if document_indices is None else documents[document_indices], 
            [embedding],
        )[:,0]


    def save_cache(
        self,
        save_path: Path,
        fail_on_overwrite: bool = True,
    ) -> None:

        self.cache.save(save_path, fail_on_overwrite)


    def save_to_file(
        self,
        save_path: Path,
        fail_on_overwrite: bool = True,
    ) -> None:

        if save_path.suffix.lower() != '.parquet':
            raise TypeError('save_path must be a .parquet file.')
        if fail_on_overwrite and save_path.exists():
            raise FileExistsError('save_path already exists.')
        self.text_encodings.write_parquet(save_path)


    @classmethod
    def load_from_file(
        cls, 
        file_path: Path, 
        embedding_model: EmbeddingModelInterface,
    ) -> Self:

        self = cls(embedding_model)
        self.text_encodings = pl.read_parquet(file_path)
        self._is_fit = True
        return self