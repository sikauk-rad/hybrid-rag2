from collections import Counter
from collections.abc import Iterable, Sequence
import operator as op
from pathlib import Path
import re
from typing import Literal, Self

from beartype import beartype
# import bm25s
from llm_utilities.utilities import check_all_arguments_are_none_or_not, get_optimal_uintype
import orjson as json
from nltk.corpus import stopwords
import numpy as np
from numpy.typing import NDArray
import polars as pl
from scipy.sparse import csr_array, sparray, coo_array, save_npz, load_npz#, lil_array
from sklearn.feature_extraction.text import TfidfVectorizer#, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Stemmer import Stemmer
from tokenizers.normalizers import NFKD, StripAccents
from tokenizers.normalizers import Sequence as Normaliser

from ..base import TextTransformer


@beartype
class TextTDFIF(TextTransformer):

    def __init__(
        self,
        remove_accents: bool,
        stop_words: Iterable[str],
        dtype: Literal['float32', 'float64'],
    ) -> None:

        self.remove_accents = remove_accents
        self.stop_words = [*{*stop_words}]
        self.dtype = op.attrgetter(dtype)(np)
        self.model = TfidfVectorizer(
            strip_accents = 'unicode' if self.remove_accents else None,
            analyzer = 'word',
            stop_words = self.stop_words,
            dtype = self.dtype,
        )
        self._is_fit = False
        self.text_encodings = None


    def fit_transform(
        self,
        texts: Sequence[str],
    ) -> NDArray[np.floating]:

        self.text_encodings = self.model.fit_transform(texts).toarray()
        self._is_fit = True
        return self.text_encodings


    async def afit_transform(
        self,
        texts: Sequence[str],
    ) -> NDArray[np.floating]:

        return self.fit_transform(texts)


    def fit(
        self,
        texts: Sequence[str],
    ) -> Self:

        self.fit_transform(texts)
        return self


    async def afit(
        self,
        texts: Sequence[str],
    ) -> Self:

        self.fit_transform(texts)
        return self


    def transform(
        self,
        text: str,
    ) -> NDArray[np.floating]:

        self._check_fit()
        return self.model.transform([text]).toarray()[0]


    async def atransform(
        self,
        text: str,
    ) -> NDArray[np.floating]:

        return self.transform(text)


    def transform_multiple(
        self,
        texts: Sequence[str],
    ) -> NDArray[np.floating]:

        self._check_fit()
        return self.model.transform(texts).toarray()


    async def atransform_multiple(
        self,
        texts: Sequence[str],
    ) -> NDArray[np.floating]:

        return self.transform_multiple(texts)


    def score(
        self,
        text: str,
        document_indices: list[int] | NDArray[np.integer] | None = None,
        documents: Iterable[Iterable[int | float]] | None = None,
    ) -> NDArray[np.floating]:

        keywords = self.transform(text)
        if documents is None:
            self._check_fit()
            documents = self.text_encodings

        return cosine_similarity(
            documents if document_indices is None else documents[document_indices], 
            [keywords],
        )[:,0]


    async def ascore(
        self,
        text: str,
        document_indices: list[int] | NDArray[np.integer] | None = None,
        documents: Iterable[Iterable[int | float]] | None = None,
    ) -> NDArray[np.floating]:

        return self.score(
            text = text, 
            document_indices = document_indices, 
            documents = documents,
        )


# @beartype
# class TextBM25Original(TextTransformer):

#     def __init__(
#         self,
#         stemming_language: Literal[
#             'arabic',
#             'armenian',
#             'basque',
#             'catalan',
#             'danish',
#             'dutch',
#             'english',
#             'finnish',
#             'french',
#             'german',
#             'greek',
#             'hindi',
#             'hungarian',
#             'indonesian',
#             'irish',
#             'italian',
#             'lithuanian',
#             'nepali',
#             'norwegian',
#             'porter',
#             'portuguese',
#             'romanian',
#             'russian',
#             'serbian',
#             'spanish',
#             'swedish',
#             'tamil',
#             'turkish',
#             'yiddish'
#         ] | None = 'english',
#         stop_words: Iterable[str] | Literal[
#             'arabic',
#             'azerbaijani',
#             'basque',
#             'bengali',
#             'catalan',
#             'chinese',
#             'danish',
#             'dutch',
#             'english',
#             'finnish',
#             'french',
#             'german',
#             'greek',
#             'hebrew',
#             'hinglish',
#             'hungarian',
#             'indonesian',
#             'italian',
#             'kazakh',
#             'nepali',
#             'norwegian',
#             'portuguese',
#             'romanian',
#             'russian',
#             'slovene',
#             'spanish',
#             'swedish',
#             'tajik',
#             'turkish'
#         ] = 'english',
#         method: Literal['robertson', 'lucene', 'atire'] = 'lucene',
#         k1: int | float = 1.5, 
#         b: int | float = 0.75,
#         sparse: bool = True,
#     ) -> None:

#         if isinstance(stop_words, str):
#             self.stop_words = [*{*stopwords.words(stop_words)}]
#         else:
#             self.stop_words = [*{*stop_words}]

#         if stemming_language is None:
#             self.stemmer = None
#         else:
#             self.stemmer = Stemmer(stemming_language)
    
#         self.model = bm25s.BM25(method = method)
#         self._is_fit = False
#         self.k1 = k1
#         self.b = b
#         self.sparse = sparse
#         self.scorer = getattr(self, f'_{method}_score')


#     def tokenise(
#         self,
#         text: str,
#     ) -> list[str]:

#         return bm25s.tokenize(
#             text, 
#             stopwords = self.stop_words, 
#             stemmer = self.stemmer,
#             return_ids = False,
#         )[0]


#     def tokenise_multiple(
#         self,
#         texts: list[str],
#     ) -> list[list[str]]:

#         return bm25s.tokenize(
#             texts, 
#             stopwords = self.stop_words, 
#             stemmer = self.stemmer,
#             return_ids = False,
#         )


#     @staticmethod
#     def create_token_index(
#         document_tokens: list[list[int]],
#         n_unique_tokens: int,
#         n_documents: int | None = None,
#         sparse: bool = True,
#     ) -> NDArray[np.int64] | csr_array:

#         shape = (
#             len(document_tokens) if n_documents is None else n_documents,
#             n_unique_tokens,
#         )
#         if sparse:
#             token_counts = lil_array(shape, dtype = 'int64')
#         else:
#             token_counts = np.zeros(shape, dtype = 'int64')

#         column_range = np.arange(token_counts.shape[1], dtype = 'int64')
#         for row, words in enumerate(document_tokens):
#             counts = np.bincount(words)
#             counts_mask = counts > 0
#             columns = column_range[:counts.shape[0]][counts_mask]
#             counts = counts[counts_mask]
#             token_counts[row, columns] = counts

#         return token_counts.tocsr() if sparse else token_counts


#     # @staticmethod
#     # def create_token_map(
#         # token_index: dict[str, int],
#         # token_index_length: int | None = None,
#     # ) -> NDArray:

#         # count =  len(token_index.keys()) if not token_index_length else token_index_length
#         # tokens = np.fromiter(
#         #     token_index.keys(),
#         #     dtype = f'<U{max(map(len, token_index.keys()))+1}', 
#         #     count = count,
#         # )
#         # max_position = max(token_index.values())
#         # positions = np.fromiter(
#         #     token_index.values(),
#         #     dtype = 'int64',
#         #     count = count,
#         # )
#         # empty = np.full(max_position + 1, '', dtype = tokens.dtype)
#         # empty[positions] = tokens

#         # return empty


#     @staticmethod
#     def calculate_document_attributes(
#         term_frequencies: NDArray[np.integer] | csr_array,
#     ) -> tuple[
#         NDArray[np.int64], 
#         NDArray[np.float32],
#     ]:

#         document_frequencies = (term_frequencies > 0).sum(axis = 0).astype('int64')
#         document_lengths = term_frequencies.sum(axis = 1).astype('float32')
#         document_lengths /= document_lengths.mean(dtype = 'float32')
#         return document_frequencies, document_lengths


#     def fit_transform(
#         self,
#         texts: list[str],
#     ) -> tuple[NDArray[np.floating] | csr_array, dict[str, int]]:

#         document_tokens, self.token_map = bm25s.tokenize(
#             texts, 
#             stopwords = self.stop_words, 
#             stemmer = self.stemmer,
#             return_ids = True,
#         )
#         self.model.index((document_tokens, self.token_map))

#         self.n_documents = len(document_tokens)
#         self.n_unique_tokens = len(self.token_map)

#         term_frequencies = self.create_token_index(
#             document_tokens,
#             self.n_unique_tokens,
#             n_documents = self.n_documents,
#             sparse = self.sparse,
#         )
#         self.document_frequencies, self.document_lengths = self.calculate_document_attributes(
#             term_frequencies
#         )
#         self.term_scores = self.scorer(
#             term_frequencies.tocoo() if self.sparse else term_frequencies,
#             self.document_frequencies,
#             self.document_lengths,
#             self.n_documents,
#             k1 = self.k1,
#             b = self.b
#         )
#         if self.sparse:
#             self.term_scores = self.term_scores.tocsr()
#         self._is_fit = True
#         return self.term_scores, self.token_map


#     def fit(
#         self,
#         texts: list[str],
#     ) -> None:

#         self.fit_transform(texts)


#     async def afit_transform(
#         self,
#         texts: list[str]
#     )-> tuple[NDArray[np.floating], dict[str, int]]:

#         return self.fit_transform(texts)


#     async def afit(
#         self,
#         texts: list[str],
#     ) -> None:

#         self.fit_transform(texts)


#     def transform(
#         self,
#         text: str,
#         token_map: dict[str, int] | None = None,
#     ) -> NDArray[np.int64]:

#         text_tokenised = self.tokenise(text)
#         text_token_counter = Counter(text_tokenised)

#         token_map = self.token_map if token_map is None else token_map
#         token_intersection = text_token_counter.keys() & token_map.keys()
#         if not token_intersection:
#             return np.array([], dtype = 'int64')
#         fetcher = op.itemgetter(*token_intersection)

#         return np.array(
#             fetcher(token_map), 
#             dtype = 'int64',
#         ).repeat(fetcher(text_token_counter))


#     async def atransform(
#         self,
#         text: str,
#         token_map: dict[str, int] | None = None,
#     )-> NDArray[np.int64]:

#         return self.transform(text, token_map)


#     def transform_multiple(
#         self,
#         texts: list[str],
#         token_map: dict[str, int] | None = None,
#     ) -> list[NDArray[np.int64]]:

#         token_arrays = []
#         for text in self.tokenise_multiple(texts):
#             token_arrays.append(self.transform(text, token_map))
#         return token_arrays


#     async def atransform_multiple(
#         self,
#         texts: list[str],
#         token_map: dict[str, int] | None = None,
#     )-> list[NDArray[np.int64]]:

#         return self.transform_multiple(texts, token_map)


#     @staticmethod
#     def _robertson_score(
#         term_frequencies: NDArray[np.integer] | coo_array,
#         document_frequencies: NDArray[np.integer],
#         document_lengths: NDArray[np.floating],
#         n_documents: int, 
#         k1: int | float,
#         b: int | float,
#         **kwargs,
#     ) -> NDArray[np.floating] | coo_array:

#         tfc_term = term_frequencies / (
#             k1 * ((1 - b) + b * document_lengths.reshape(-1, 1)) + term_frequencies
#         )
#         idf_term = (
#             (n_documents - document_frequencies + 0.5)
#             /
#             (document_frequencies + 0.5)
#         )
#         idf_term[idf_term < 1] = 1
#         return tfc_term * np.log(idf_term)


#     @staticmethod
#     def _lucene_score(
#         term_frequencies: NDArray[np.integer] | coo_array,
#         document_frequencies: NDArray[np.integer],
#         document_lengths: NDArray[np.floating],
#         n_documents: int, 
#         k1: int | float,
#         b: int | float,
#         **kwargs,
#     ) -> NDArray[np.floating] | coo_array:

#         tfc_term = term_frequencies / (
#             k1 * ((1 - b) + b * document_lengths.reshape(-1, 1)) + term_frequencies
#         )
#         idf_term = 1 + (
#             (n_documents - document_frequencies + 0.5)
#             /
#             (document_frequencies + 0.5)
#         )
#         return tfc_term * np.log(idf_term)


#     @staticmethod
#     def _atire_score(
#         term_frequencies: NDArray[np.integer] | coo_array,
#         document_frequencies: NDArray[np.integer],
#         document_lengths: NDArray[np.floating],
#         n_documents: int, 
#         k1: int | float,
#         b: int | float,
#         **kwargs,
#     ) -> NDArray[np.floating] | coo_array:

#         tfc_term = (term_frequencies * (k1 + 1)) / (
#             term_frequencies + k1 * (1 - b + b * document_lengths.reshape(-1,1))
#         )
#         idf_term = n_documents / document_frequencies

#         return tfc_term * np.log(idf_term)


#     @staticmethod
#     def _bm25l_score(
#         text_term_frequencies: NDArray[np.integer] | coo_array,
#         document_frequencies: NDArray[np.integer],
#         document_lengths: NDArray[np.floating],
#         n_documents: int, 
#         k1: int | float,
#         b: int | float,
#         delta: int | float,
#     ) -> NDArray[np.floating]:

#         raise NotImplementedError('not yet finished.')
#         c_array = text_term_frequencies / (1 - b + b * document_lengths.reshape(-1,1))
#         tfc_term = ((k1 + 1) * (c_array + delta)) / (k1 + c_array + delta)
#         idf_term = (n_documents + 1) / (document_frequencies + 0.5)

#         return tfc_term * np.log(idf_term)


#     @staticmethod
#     def _bm25plus_score(
#         text_term_frequencies: NDArray[np.integer] | coo_array,
#         document_frequencies: NDArray[np.integer],
#         document_lengths: NDArray[np.floating],
#         n_documents: int, 
#         k1: int | float,
#         b: int | float,
#         delta: int | float,
#     ) -> NDArray[np.floating]:

#         raise NotImplementedError('not yet finished.')
#         tfc_term = (
#             ((k1 + 1) * text_term_frequencies)
#             /
#             (k1 * (1 - b + b * document_lengths.reshape(-1,1)) + text_term_frequencies)
#         ) + delta
#         idf_term = (n_documents + 1) / document_frequencies

#         return tfc_term * np.log(idf_term)


#     @staticmethod
#     def normalise_relevances(
#         document_relevances: NDArray[np.floating],
#     ) -> NDArray[np.floating]:

#         min_relevance = document_relevances.min()
#         relevance_range = document_relevances.max() - min_relevance
#         return (document_relevances - min_relevance) / relevance_range


#     def score(
#         self,
#         text: str,
#         normalise: bool = True,
#         document_indices: list[int] | NDArray[np.integer] | None = None,
#         documents: NDArray[np.floating] | sparray | None = None,
#         token_map: dict[str, int] | None = None,
#         use_base_model: bool = False,
#     ) -> NDArray[np.floating] | sparray:

#         document_relevances = None
#         if documents is None:
#             if use_base_model:
#                 document_relevances = self.model.get_scores(self.tokenise(text))
#             elif token_map is not None:
#                 raise TypeError('if token_map is provided, documents must be given too.')
#             else:
#                 documents, token_map = self.term_scores, self.token_map
#         else:
#             if token_map is not None:
#                 raise TypeError('if documents is provided, token_map must be given too.')
#             elif isinstance(documents, sparray) and not isinstance(documents, csr_array):
#                 documents = documents.tocsr()

#         if document_relevances is None:
#             text_token_idx = self.transform(text)
#             if not text_token_idx.size:
#                 return np.zeros(documents.shape[0], dtype = 'float32')
#             document_relevances = documents[:,text_token_idx].sum(axis = 1)

#         if document_indices is not None:
#             document_relevances = document_relevances[document_indices]

#         if normalise:
#             document_relevances = self.normalise_relevances(document_relevances)

#         return document_relevances


#     async def ascore(
#         self,
#         text: str,
#         normalise: bool = True,
#         document_indices: list[int] | NDArray[np.integer] | None = None,
#         documents: NDArray[np.floating] | sparray | None = None,
#         token_map: dict[str, int] | None = None,
#         use_base_model: bool = False,
#     ) -> NDArray[np.floating]:

#         return self.score(
#             text = text,
#             normalise = normalise,
#             document_indices = document_indices,
#             documents = documents,
#             token_map = token_map,
#             use_base_model = use_base_model,
#         )


# @beartype
# class TextBM25SKLearn(TextTransformer):

#     def __init__(
#         self,
#         stemming_language: Literal[
#             'arabic',
#             'armenian',
#             'basque',
#             'catalan',
#             'danish',
#             'dutch',
#             'english',
#             'finnish',
#             'french',
#             'german',
#             'greek',
#             'hindi',
#             'hungarian',
#             'indonesian',
#             'irish',
#             'italian',
#             'lithuanian',
#             'nepali',
#             'norwegian',
#             'porter',
#             'portuguese',
#             'romanian',
#             'russian',
#             'serbian',
#             'spanish',
#             'swedish',
#             'tamil',
#             'turkish',
#             'yiddish',
#         ] | None,
#         stop_words: Iterable[str] | Literal[
#             'arabic',
#             'azerbaijani',
#             'basque',
#             'bengali',
#             'catalan',
#             'chinese',
#             'danish',
#             'dutch',
#             'english',
#             'finnish',
#             'french',
#             'german',
#             'greek',
#             'hebrew',
#             'hinglish',
#             'hungarian',
#             'indonesian',
#             'italian',
#             'kazakh',
#             'nepali',
#             'norwegian',
#             'portuguese',
#             'romanian',
#             'russian',
#             'slovene',
#             'spanish',
#             'swedish',
#             'tajik',
#             'turkish'
#         ] = 'english',
#         strip_accents: Literal['ascii', 'unicode'] | None = 'unicode',
#         lowercase: bool = True,
#         find_pattern: str = r"(?u)\b\w\w+\b",
#         method: Literal['robertson', 'lucene', 'atire'] = 'lucene',
#         k1: int | float = 1.5, 
#         b: int | float = 0.75,
#     ) -> None:

#         self.stemmer = Stemmer(stemming_language) if stemming_language else None
#         self.find_pattern = re.compile(find_pattern)
#         self.k1 = k1
#         self.b = b
#         if isinstance(stop_words, str):
#             self.stop_words = stopwords.words(stop_words)
#         else:
#             self.stop_words = stop_words
#         self.stop_words = [*{*self.stemmer.stemWords(stop_words), *stop_words}]

#         self.model = CountVectorizer(
#             input = 'content',
#             strip_accents = strip_accents,
#             lowercase = lowercase,
#             tokenizer = self._internal_tokenise,
#             stop_words = self.stop_words,
#             analyzer = 'word',
#             binary = False,
#         )
#         self.scorer = getattr(self, f'_{method}_score')
#         self.tokenise = self.model.build_analyzer()
#         self._is_fit = False


#     def _internal_tokenise(
#             self, 
#             text: str,
#         ) -> list[str]:

#         tokens = re.findall(self.find_pattern, text)
#         return self.stemmer.stemWords(tokens) if self.stemmer else tokens


#     @staticmethod
#     def calculate_document_attributes(
#         term_frequencies: csr_array,
#     ) -> tuple[
#         NDArray[np.int64], 
#         NDArray[np.float32],
#     ]:

#         document_frequencies = (term_frequencies > 0).sum(axis = 0).astype('int64')
#         document_lengths = term_frequencies.sum(axis = 1).astype('float32')
#         document_lengths /= document_lengths.mean(dtype = 'float32')
#         return document_frequencies, document_lengths


#     @staticmethod
#     def _robertson_score(
#         term_frequencies: NDArray[np.integer] | coo_array,
#         document_frequencies: NDArray[np.integer],
#         document_lengths: NDArray[np.floating],
#         n_documents: int, 
#         k1: int | float,
#         b: int | float,
#         **kwargs,
#     ) -> NDArray[np.floating] | coo_array:

#         tfc_term = term_frequencies / (
#             k1 * ((1 - b) + b * document_lengths.reshape(-1, 1)) + term_frequencies
#         )
#         idf_term = (
#             (n_documents - document_frequencies + 0.5)
#             /
#             (document_frequencies + 0.5)
#         )
#         idf_term[idf_term < 1] = 1
#         return tfc_term * np.log(idf_term)


#     @staticmethod
#     def _lucene_score(
#         term_frequencies: NDArray[np.integer] | coo_array,
#         document_frequencies: NDArray[np.integer],
#         document_lengths: NDArray[np.floating],
#         n_documents: int, 
#         k1: int | float,
#         b: int | float,
#         **kwargs,
#     ) -> NDArray[np.floating] | coo_array:

#         tfc_term = term_frequencies / (
#             k1 * ((1 - b) + b * document_lengths.reshape(-1, 1)) + term_frequencies
#         )
#         idf_term = 1 + (
#             (n_documents - document_frequencies + 0.5)
#             /
#             (document_frequencies + 0.5)
#         )
#         return tfc_term * np.log(idf_term)


#     @staticmethod
#     def _atire_score(
#         term_frequencies: NDArray[np.integer] | coo_array,
#         document_frequencies: NDArray[np.integer],
#         document_lengths: NDArray[np.floating],
#         n_documents: int, 
#         k1: int | float,
#         b: int | float,
#         **kwargs,
#     ) -> NDArray[np.floating] | coo_array:

#         tfc_term = (term_frequencies * (k1 + 1)) / (
#             term_frequencies + k1 * (1 - b + b * document_lengths.reshape(-1,1))
#         )
#         idf_term = n_documents / document_frequencies

#         return tfc_term * np.log(idf_term)


#     @staticmethod
#     def _bm25l_score(
#         text_term_frequencies: NDArray[np.integer] | coo_array,
#         document_frequencies: NDArray[np.integer],
#         document_lengths: NDArray[np.floating],
#         n_documents: int, 
#         k1: int | float,
#         b: int | float,
#         delta: int | float,
#     ) -> NDArray[np.floating]:

#         raise NotImplementedError('not yet finished.')
#         c_array = text_term_frequencies / (1 - b + b * document_lengths.reshape(-1,1))
#         tfc_term = ((k1 + 1) * (c_array + delta)) / (k1 + c_array + delta)
#         idf_term = (n_documents + 1) / (document_frequencies + 0.5)

#         return tfc_term * np.log(idf_term)


#     @staticmethod
#     def _bm25plus_score(
#         text_term_frequencies: NDArray[np.integer] | coo_array,
#         document_frequencies: NDArray[np.integer],
#         document_lengths: NDArray[np.floating],
#         n_documents: int, 
#         k1: int | float,
#         b: int | float,
#         delta: int | float,
#     ) -> NDArray[np.floating]:

#         raise NotImplementedError('not yet finished.')
#         tfc_term = (
#             ((k1 + 1) * text_term_frequencies)
#             /
#             (k1 * (1 - b + b * document_lengths.reshape(-1,1)) + text_term_frequencies)
#         ) + delta
#         idf_term = (n_documents + 1) / document_frequencies

#         return tfc_term * np.log(idf_term)


#     def fit_transform(
#         self,
#         texts: list[str],
#     ) -> tuple[csr_array, dict[str, int]]:

#         term_frequencies = csr_array(self.model.fit_transform(texts))
#         self.token_map = self.model.vocabulary_
#         self.document_frequencies, self.document_lengths = self.calculate_document_attributes(
#             term_frequencies,
#         )
#         self.n_documents, self.n_unique_tokens = term_frequencies.shape
#         self.term_scores = self.scorer(
#             term_frequencies.tocoo(),
#             self.document_frequencies,
#             self.document_lengths,
#             self.n_documents,
#             k1 = self.k1,
#             b = self.b
#         ).tocsr()
#         self._is_fit = True
#         return self.term_scores, self.token_map


#     def fit(
#         self,
#         texts: list[str],
#     ) -> Self:

#         self.fit_transform(texts)
#         return self


#     async def afit_transform(
#         self,
#         texts: list[str],
#     ) -> Self:

#         return self.fit_transform(texts)


#     async def afit(
#         self,
#         texts: list[str],
#     ) -> Self:

#         return self.fit(texts)


#     def transform(
#         self,
#         text: str,
#         token_map: dict[str, int] | None = None,
#     ) -> NDArray[np.int64]:

#         text_tokenised = self.tokenise(text)
#         text_token_counter = Counter(text_tokenised)

#         token_map = self.token_map if token_map is None else token_map
#         token_intersection = text_token_counter.keys() & token_map.keys()
#         if not token_intersection:
#             return np.array([], dtype = 'int64')
#         fetcher = op.itemgetter(*token_intersection)

#         return np.array(
#             fetcher(token_map), 
#             dtype = 'int64',
#         ).repeat(fetcher(text_token_counter))


#     async def atransform(
#         self,
#         text: str,
#         token_map: dict[str, int] | None = None,
#     )-> NDArray[np.int64]:

#         return self.transform(text, token_map)


#     def transform_multiple(
#         self,
#         texts: list[str],
#         token_map: dict[str, int] | None = None,
#     ) -> list[NDArray[np.int64]]:

#         token_arrays = []
#         for text in self.tokenise_multiple(texts):
#             token_arrays.append(self.transform(text, token_map))
#         return token_arrays


#     async def atransform_multiple(
#         self,
#         texts: list[str],
#         token_map: dict[str, int] | None = None,
#     )-> list[NDArray[np.int64]]:

#         return self.transform_multiple(texts, token_map)


#     @staticmethod
#     def normalise_relevances(
#         document_relevances: NDArray[np.floating],
#     ) -> NDArray[np.floating]:

#         min_relevance = document_relevances.min()
#         relevance_range = document_relevances.max() - min_relevance
#         return (document_relevances - min_relevance) / relevance_range


#     def score(
#         self,
#         text: str,
#         normalise: bool = True,
#         document_indices: list[int] | NDArray[np.integer] | None = None,
#         documents: NDArray[np.floating] | sparray | None = None,
#         token_map: dict[str, int] | None = None,
#     ) -> NDArray[np.floating]:

#         if not check_all_arguments_are_none_or_not((documents, token_map)):
#             raise TypeError(
#                 'either both or neither of documents and token_map should be provided.'
#             )
#         elif documents is None:
#             documents, token_map = self.term_scores, self.token_map
#         elif not isinstance(documents, sparray) and not isinstance(documents, csr_array):
#             documents = documents.tocsr()

#         if document_indices is not None:
#             documents = documents[document_indices]

#         text_token_idx = self.transform(text)
#         if not text_token_idx.size:
#             return np.zeros(documents.shape[0], dtype = 'float32')

#         document_relevances = documents[:,text_token_idx].sum(axis = 1)

#         if normalise:
#             document_relevances = self.normalise_relevances(document_relevances)

#         return document_relevances


#     async def ascore(
#         self,
#         text: str,
#         normalise: bool = True,
#         document_indices: list[int] | NDArray[np.integer] | None = None,
#         documents: NDArray[np.floating] | sparray | None = None,
#         token_map: dict[str, int] | None = None,
#     ) -> NDArray[np.floating]:

#         return self.score(
#             text = text,
#             normalise = normalise,
#             document_indices = document_indices,
#             documents = documents,
#             token_map = token_map,
#         )


@beartype
class TextBM25(TextTransformer):

    def __init__(
        self,
        stemming_language: Literal[
            'arabic',
            'armenian',
            'basque',
            'catalan',
            'danish',
            'dutch',
            'english',
            'finnish',
            'french',
            'german',
            'greek',
            'hindi',
            'hungarian',
            'indonesian',
            'irish',
            'italian',
            'lithuanian',
            'nepali',
            'norwegian',
            'porter',
            'portuguese',
            'romanian',
            'russian',
            'serbian',
            'spanish',
            'swedish',
            'tamil',
            'turkish',
            'yiddish',
        ] | None,
        stop_words: Iterable[str] | Literal[
            'arabic',
            'azerbaijani',
            'basque',
            'bengali',
            'catalan',
            'chinese',
            'danish',
            'dutch',
            'english',
            'finnish',
            'french',
            'german',
            'greek',
            'hebrew',
            'hinglish',
            'hungarian',
            'indonesian',
            'italian',
            'kazakh',
            'nepali',
            'norwegian',
            'portuguese',
            'romanian',
            'russian',
            'slovene',
            'spanish',
            'swedish',
            'tajik',
            'turkish'
        ] = 'english',
        lowercase: bool = True,
        normalise: bool = True,
        strip_accents: bool = True,
        find_pattern: str = r"(?u)\b\w\w+\b",
        method: Literal['robertson', 'lucene', 'atire'] = 'lucene',
        k1: int | float = 1.5, 
        b: int | float = 0.75,
        stem_stop_words: bool = True,
    ) -> None:

        self.stemming_language = stemming_language
        self.stemmer = Stemmer(stemming_language) if stemming_language else None
        self.find_pattern = re.compile(find_pattern)
        self.k1 = k1
        self.lowercase = lowercase
        self.normalise = normalise
        self.method = method
        self.b = b
        self.strip_accents = strip_accents
        if isinstance(stop_words, str):
            self.stop_words = stopwords.words(stop_words)
        else:
            self.stop_words = stop_words
        if stem_stop_words:
            self.stop_words = [*{
                *self.stemmer.stemWords(self.stop_words), 
                *self.stop_words,
            }]
        self.scorer = getattr(self, f'_{method}_score')
        self._is_fit = False
        normaliser_sequence = []
        self.normalise = not not normalise

        if normalise:
            normaliser_sequence.append(NFKD())
        if strip_accents:
            normaliser_sequence.append(StripAccents())
        self._normaliser = Normaliser(normaliser_sequence) if normaliser_sequence else None


    @staticmethod
    def calculate_document_attributes(
        term_frequencies: csr_array,
    ) -> tuple[
        NDArray[np.int64], 
        NDArray[np.float32],
    ]:

        document_frequencies = (term_frequencies > 0).sum(axis = 0).astype('uint64')
        document_lengths = term_frequencies.sum(axis = 1).astype('float32')
        document_lengths /= document_lengths.mean(dtype = 'float32')
        return document_frequencies, document_lengths


    @staticmethod
    def _robertson_score(
        term_frequencies: NDArray[np.integer] | coo_array,
        document_frequencies: NDArray[np.integer],
        document_lengths: NDArray[np.floating],
        n_documents: int, 
        k1: int | float,
        b: int | float,
        **kwargs,
    ) -> NDArray[np.floating] | coo_array:

        tfc_term = term_frequencies / (
            k1 * ((1 - b) + b * document_lengths.reshape(-1, 1)) + term_frequencies
        )
        idf_term = (
            (n_documents - document_frequencies + 0.5)
            /
            (document_frequencies + 0.5)
        )
        idf_term[idf_term < 1] = 1
        return tfc_term * np.log(idf_term)


    @staticmethod
    def _lucene_score(
        term_frequencies: NDArray[np.integer] | coo_array,
        document_frequencies: NDArray[np.integer],
        document_lengths: NDArray[np.floating],
        n_documents: int, 
        k1: int | float,
        b: int | float,
        **kwargs,
    ) -> NDArray[np.floating] | coo_array:

        tfc_term = term_frequencies / (
            k1 * ((1 - b) + b * document_lengths.reshape(-1, 1)) + term_frequencies
        )
        idf_term = 1 + (
            (n_documents - document_frequencies + 0.5)
            /
            (document_frequencies + 0.5)
        )
        return tfc_term * np.log(idf_term)


    @staticmethod
    def _atire_score(
        term_frequencies: NDArray[np.integer] | coo_array,
        document_frequencies: NDArray[np.integer],
        document_lengths: NDArray[np.floating],
        n_documents: int, 
        k1: int | float,
        b: int | float,
        **kwargs,
    ) -> NDArray[np.floating] | coo_array:

        tfc_term = (term_frequencies * (k1 + 1)) / (
            term_frequencies + k1 * (1 - b + b * document_lengths.reshape(-1,1))
        )
        idf_term = n_documents / document_frequencies

        return tfc_term * np.log(idf_term)


    @staticmethod
    def _bm25l_score(
        text_term_frequencies: NDArray[np.integer] | coo_array,
        document_frequencies: NDArray[np.integer],
        document_lengths: NDArray[np.floating],
        n_documents: int, 
        k1: int | float,
        b: int | float,
        delta: int | float,
    ) -> NDArray[np.floating]:

        raise NotImplementedError('not yet finished.')
        c_array = text_term_frequencies / (1 - b + b * document_lengths.reshape(-1,1))
        tfc_term = ((k1 + 1) * (c_array + delta)) / (k1 + c_array + delta)
        idf_term = (n_documents + 1) / (document_frequencies + 0.5)

        return tfc_term * np.log(idf_term)


    @staticmethod
    def _bm25plus_score(
        text_term_frequencies: NDArray[np.integer] | coo_array,
        document_frequencies: NDArray[np.integer],
        document_lengths: NDArray[np.floating],
        n_documents: int, 
        k1: int | float,
        b: int | float,
        delta: int | float,
    ) -> NDArray[np.floating]:

        raise NotImplementedError('not yet finished.')
        tfc_term = (
            ((k1 + 1) * text_term_frequencies)
            /
            (k1 * (1 - b + b * document_lengths.reshape(-1,1)) + text_term_frequencies)
        ) + delta
        idf_term = (n_documents + 1) / document_frequencies

        return tfc_term * np.log(idf_term)


    def tokenise(
        self,
        text: str,
    ) -> list[str]:

        if self.lowercase:
            text = text.casefold()
        if self._normaliser:
            text = self._normaliser.normalize_str(text)
        tokens = [token for token in re.findall(
            self.find_pattern, 
            text,
        ) if token not in self.stop_words]
        if self.stemmer:
            tokens = [token for token in self.stemmer.stemWords(
                tokens
            ) if token not in self.stop_words]
        return tokens


    def tokenise_multiple(
        self,
        texts: Sequence[str],
        as_lazyframe: bool = False,
    ) -> list[list[str]] | pl.LazyFrame:

        texts_frame = pl.LazyFrame(
            [texts],
            schema = {'text': pl.String},
        ).with_row_index(
            name = 'index'
        )

        if self.lowercase:
            texts_frame = texts_frame.with_columns(
                pl.col('text').str.to_lowercase()
            )

        if self._normaliser:
            texts_frame = texts_frame.with_columns(
                pl.col('text').map_elements(
                    self._normaliser.normalize_str,
                    return_dtype = pl.String,
                )
            )

        tokens_frame = texts_frame.with_columns(
            pl.col('text').str.extract_all(
                self.find_pattern.pattern,
            )
        ).explode(
            'text'
        ).with_columns(
            pl.col(
                'text'
            ).str.strip_chars()
        )

        if self.stop_words:
            tokens_frame = tokens_frame.filter(
                pl.col(
                    'text'
                ).is_in(
                    self.stop_words
                ).not_()
            )

        tokens_frame = tokens_frame.group_by(
            'text',
            maintain_order = True,
        ).agg(
            'index'
        )

        if self.stemmer:
            tokens_frame = tokens_frame.with_columns(
            pl.col(
                'text'
            ).map_elements(
                self.stemmer.stemWord,
                return_dtype = pl.String,
            )
        )
            if self.stop_words:
                tokens_frame = tokens_frame.filter(
                    pl.col(
                        'text'
                    ).is_in(
                        self.stop_words
                    ).not_(),
                )

        tokens_frame = tokens_frame.filter(
            pl.col(
                'text'
            ).str.replace_all(
                '[^A-z0-9]',
                '',
            ).str.len_chars().gt(
                0
            )
        )

        if as_lazyframe:
            return tokens_frame
        else:
            return tokens_frame.explode(
                'index'
            ).group_by(
                'index',
                maintain_order = False,
            ).agg(
                'text'
            ).select(
                'text',
            ).collect()[:,0].to_list()


    def fit_transform(
        self,
        texts: Sequence[str],
    ) -> tuple[csr_array, dict[str, int]]:

        tokens_frame = self.tokenise_multiple(
            texts,
            as_lazyframe = True,
        ).select(
            'text',
            pl.col(
                'index'
            ).list.eval(
                pl.element().value_counts()
            )
        ).with_row_index(
            'column'
        )

        self.token_map = dict(tokens_frame.select(
            'text',
            'column',
        ).collect().rows())

        term_frequencies = tokens_frame.explode(
            'index'
        ).select(
            pl.col(
                'index'
            ).struct.field(
                'count'
            ),
            pl.col(
                'index'
            ).struct.field(
                ''
            ),
            pl.col(
                'column',
            ),
        ).collect().to_numpy()

        self.count_dtype = get_optimal_uintype(term_frequencies[:,0].max())

        term_frequencies = csr_array(
            (
                term_frequencies[:,0],
                (
                    term_frequencies[:,1],
                    term_frequencies[:,2],
                ),
            ),
            dtype = self.count_dtype,
        )

        self.document_frequencies, self.document_lengths = self.calculate_document_attributes(
            term_frequencies,
        )
        self.n_documents, self.n_unique_tokens = term_frequencies.shape
        self.term_scores = self.scorer(
            term_frequencies.tocoo(),
            self.document_frequencies,
            self.document_lengths,
            self.n_documents,
            k1 = self.k1,
            b = self.b
        ).tocsr()
        self._is_fit = True
        return self.term_scores, self.token_map


    def fit(
        self,
        texts: Sequence[str],
    ) -> Self:

        self.fit_transform(texts)
        return self


    async def afit_transform(
        self,
        texts: Sequence[str],
    ) -> Self:

        return self.fit_transform(texts)


    async def afit(
        self,
        texts: Sequence[str],
    ) -> Self:

        return self.fit(texts)


    def transform(
        self,
        text: str,
        token_map: dict[str, int] | None = None,
    ) -> NDArray[np.uint64]:

        text_tokenised = self.tokenise(text)
        text_token_counter = Counter(text_tokenised)

        token_map = self.token_map if token_map is None else token_map
        token_intersection = text_token_counter.keys() & token_map.keys()
        if not token_intersection:
            return np.array([], dtype = 'uint64')
        fetcher = op.itemgetter(*token_intersection)

        return np.array(
            fetcher(token_map), 
            dtype = 'uint64',
        ).repeat(
            fetcher(text_token_counter)
        )


    async def atransform(
        self,
        text: str,
        token_map: dict[str, int] | None = None,
    )-> NDArray[np.int64]:

        return self.transform(text, token_map = token_map)


    def transform_multiple(
        self,
        texts: Sequence[str],
        token_map: dict[str, int] | None = None,
    ) -> list[NDArray[np.int64]]  | list[list[int]]:

        tokenised = self.tokenise_multiple(texts, as_lazyframe = True)
        token_map = self.token_map if token_map is None else token_map

        transformed = (
            tokenised
            .filter(
                pl.col(
                    'text'
                ).is_in(
                    token_map.keys()
                )
            )
            .with_columns(
                pl.col(
                    'text'
                ).replace_strict(
                    token_map,
                    return_dtype = pl.UInt64,
                    default = None,
                )
            )
            .explode(
                'index'
            )
            .group_by(
                'index'
            )
            .agg(
                'text'
            )
            .sort(
                by = 'index'
            )
            .select(
                'text'
            )
            .collect()
            [:,0]
            .to_list()
        )
        return transformed


    async def atransform_multiple(
        self,
        texts: Sequence[str],
        token_map: dict[str, int] | None = None,
    )-> list[NDArray[np.int64]]:

        return self.transform_multiple(texts, token_map = token_map)


    @staticmethod
    def normalise_relevances(
        document_relevances: NDArray[np.floating],
    ) -> NDArray[np.floating]:

        min_relevance = document_relevances.min()
        relevance_range = document_relevances.max() - min_relevance
        return (document_relevances - min_relevance) / relevance_range


    def score(
        self,
        text: str,
        normalise: bool = False,
        document_indices: list[int] | NDArray[np.integer] | None = None,
        documents: NDArray[np.floating] | sparray | None = None,
        token_map: dict[str, int] | None = None,
    ) -> NDArray[np.floating]:

        if not check_all_arguments_are_none_or_not((documents, token_map)):
            raise TypeError(
                'either both or neither of documents and token_map should be provided.'
            )
        elif documents is None:
            documents, token_map = self.term_scores, self.token_map
        elif (isinstance(documents, sparray)) and not isinstance(documents, csr_array):
            documents = documents.tocsr()

        if document_indices is not None:
            documents = documents[document_indices]

        text_token_idx = self.transform(text)
        if not text_token_idx.size:
            return np.zeros(documents.shape[0], dtype = 'float32')

        document_relevances = documents[:,text_token_idx].sum(axis = 1)

        if normalise:
            document_relevances = self.normalise_relevances(document_relevances)

        return document_relevances.flatten()


    async def ascore(
        self,
        text: str,
        normalise: bool = True,
        document_indices: list[int] | NDArray[np.integer] | None = None,
        documents: NDArray[np.floating] | sparray | None = None,
        token_map: dict[str, int] | None = None,
    ) -> NDArray[np.floating]:

        return self.score(
            text,
            normalise = normalise,
            document_indices = document_indices,
            documents = documents,
            token_map = token_map,
        )


    def save_to_file(
        self, 
        json_save_path: Path, 
        array_save_path: Path,
        fail_on_overwrite: bool = True,
    ) -> None:

        if json_save_path.suffix.lower() != '.json':
            raise TypeError('json_save_path must be a .json file.')
        elif fail_on_overwrite and json_save_path.exists():
            raise FileExistsError('json_save_path already exists.')
        elif array_save_path.suffix.lower() != '.npz':
            raise TypeError('array_save_path must be a .npz file.')
        elif fail_on_overwrite and array_save_path.exists():
            raise FileExistsError('array_save_path already exists.')

        with open(json_save_path, 'wb') as file:
            file.write(json.dumps({
                'kwargs': {
                    'stemming_language': self.stemming_language,
                    'stop_words': self.stop_words,
                    'lowercase': self.lowercase,
                    'normalise': self.normalise,
                    'strip_accents': self.strip_accents,
                    'find_pattern': self.find_pattern.pattern,
                    'method': self.method,
                    'k1': self.k1,
                    'b': self.b,
                },
                'token_map': self.token_map,
            }))
        save_npz(
            file = array_save_path,
            matrix = self.term_scores,
            compressed = True,
        )


    @classmethod
    def load_from_file(
        cls,
        json_file_path: Path,
        array_file_path: Path,
    ) -> Self:

        if json_file_path.suffix.lower() != '.json':
            raise TypeError('json_file_path must be a .json file.')
        elif array_file_path.suffix.lower() != '.npz':
            raise TypeError('array_file_path must be a .npz file.')
        with open(json_file_path, 'rb') as file:
            kwargs, token_map = json.loads(file.read()).values()
            self = cls(**kwargs, stem_stop_words = False)
        self.token_map = token_map
        self.term_scores = csr_array(load_npz(array_file_path))
        self._is_fit = True
        return self