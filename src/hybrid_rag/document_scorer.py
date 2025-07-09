from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import polars as pl
import numpy as np
from typing import Any
from beartype import beartype
from collections.abc import Iterable, Sequence
from asyncio import gather
from numbers import Number
from numpy.typing import NDArray
from polars.exceptions import ColumnNotFoundError
from .base import TextTransformer
from .exceptions import SizeError


@beartype
class DocumentScorer:

    def __init__(
        self,
        documents: Sequence[str],
        document_sizes: Sequence[int] | NDArray[np.integer],
        transformers: dict[str, TextTransformer],
        *,
        rank_weights: dict[str, Number] = {},
        transform_arguments: dict[str, dict[str, Any]] = {},
        score_arguments: dict[str, dict[str, Any]] = {},
        metadata: dict[str, Sequence[str | Number]] = {},
        reranker_name: str = 'ms-marco-MiniLM-L-6-v2',
    ) -> None:

        from sentence_transformers import CrossEncoder

        if len(documents) != len(document_sizes):
            raise SizeError(
                f'number of documents {len(documents)} does not match number of '\
                f'document sizes {len(document_sizes)}.'
            )

        self.document_table = pl.DataFrame(
            {
                'content': documents,
                'content size': document_sizes,
            } | metadata,
        ).with_row_index(
            name = '__index__',
        )

        self.transform_arguments, self.score_arguments = {},{}
        self.score_rank_rename = {}
        self.rank_reciprocal_rename = {}
        self.weighted_rank_rename = {}
        self.rank_weights = {}
        for column_name in transformers.keys():
            self.transform_arguments[column_name] = transform_arguments.get(
                column_name, 
                {},
            )
            self.score_arguments[column_name] = score_arguments.get(
                column_name,
                {}
            )
            rank_name = f'{column_name} rank'
            reciprocal_name = f'{column_name} reciprocal rank'
            weighted_name = f'{column_name} weighted reciprocal rank'
            self.score_rank_rename[f'{column_name} score'] = rank_name
            self.rank_reciprocal_rename[rank_name] = reciprocal_name
            self.weighted_rank_rename[reciprocal_name] = weighted_name
            self.rank_weights[weighted_name] = rank_weights.get(
                column_name,
                1.,
            )
        self.transformers = transformers
        self.reranker_name = reranker_name
        self.reranker = CrossEncoder(f'cross-encoder/{reranker_name}')


    # def save_to_file(
    #     self,
    #     save_path: Path,
    #     fail_on_overwrite: bool = True,
    # ) -> None:

    #     save_path.mkdir(parents = True, exist_ok = not fail_on_overwrite)
    #     save_path


    def check_column(
        self,
        column_name: str,
        dtype = None,
    ) -> None:

        if column_name not in self.document_table.columns:
            raise ColumnNotFoundError(f'Column {column_name} not found in document_table.')
        if dtype and not isinstance(self.document_table[column_name].dtype, dtype):
            raise TypeError(f'Column {column_name} is not of expected datatype {dtype}.')


    def transform_query(
        self,
        query: str,
    ) -> dict[str, list[list[float]] | NDArray[np.floating]]:
        
        return {column_name: text_transformer.transform(
            query, 
            **self.transform_arguments[column_name],
        ) for column_name, text_transformer in self.transformers.items()}


    async def atransform_query(
        self,
        query: str,
    ) -> dict[str, list[list[float]] | NDArray[np.floating]]:
        
        return dict(zip(
            self.transformers.keys(),
            await gather(*[
                text_transformer.atransform(
                    query,
                    **self.transform_arguments[column_name],
                ) for column_name, text_transformer in self.transformers.items()
            ]),
        ))


    def _calculate_weighted_score(
        self,
        score_table: pl.DataFrame,
        fusion_factor: Number,
    ) -> pl.DataFrame:

        rank_table = score_table.with_columns(
            pl.col(
                *self.score_rank_rename.keys()
            ).fill_null(
                0
            ).fill_nan(
                0
            ).rank(
                method = 'min',
                descending = True,
            ).name.map(
                self.score_rank_rename.get
            )
        )
        reciprocal_rank_table = rank_table.with_columns(
            (
                1 / (pl.col(
                    *self.rank_reciprocal_rename.keys()
                ) + fusion_factor)
            ).name.map(
                self.rank_reciprocal_rename.get
            )
        )
        weighted_rank_table = reciprocal_rank_table.with_columns(
            (pl.col(
                reciprocal_col
            ) * self.rank_weights[weight_col]).alias(
                weight_col
            ) for reciprocal_col, weight_col in self.weighted_rank_rename.items()
        )
        return weighted_rank_table.with_columns(
            pl.sum_horizontal(
                pl.col(
                    *self.rank_weights.keys()
                )
            ).alias(
                'fused weighted reciprocal rank'
            )
        ).sort(
            by = 'fused weighted reciprocal rank',
            descending = True,
        )


    def _filter_documents(
        self,
        document_table: pl.DataFrame,
        filters: Iterable[pl.Expr],
    ) -> pl.DataFrame:

        if not filters:
            return document_table
        else:
            return document_table.filter(*filters)


    def score_documents(
        self,
        query: str,
        fusion_factor: Number,
        filters: Iterable[pl.Expr] = [],
    ) -> pl.DataFrame:

        filtered_documents = self._filter_documents(self.document_table, filters)

        if not filtered_documents.shape[0]:
            return filtered_documents

        index = filtered_documents['__index__'].to_numpy()

        scored_documents = filtered_documents.with_columns(
            pl.Series(
                score_column,
                text_transformer.score(
                    query, 
                    document_indices = index,
                    **self.transform_arguments[column_name],
                ),
                dtype = pl.Float32,
            ) for (column_name, text_transformer), score_column in zip(
                self.transformers.items(),
                self.score_rank_rename.keys(),
            )
        )

        return self._calculate_weighted_score(
            scored_documents,
            fusion_factor = fusion_factor,
        )


    async def ascore_documents(
        self,
        query: str,
        fusion_factor: Number,
        filters: Iterable[pl.Expr] = [],
    ) -> pl.DataFrame:

        filtered_documents = self._filter_documents(self.document_table, filters)

        if not filtered_documents.shape[0]:
            return filtered_documents

        index = filtered_documents['__index__'].to_numpy()

        document_scores = await gather(*[text_transformer.ascore(
            query,
            document_indices = index,
            **self.transform_arguments[column_name],
        ) for column_name, text_transformer in self.transformers.items()])

        scored_documents = filtered_documents.with_columns(
            pl.Series(
                score_column,
                document_score,
                dtype = pl.Float32,
            ) for score_column, document_score in zip(
                self.score_rank_rename.keys(),
                document_scores,
            )
        )

        return self._calculate_weighted_score(
            scored_documents,
            fusion_factor = fusion_factor,
        )


    def rank_documents(
        self,
        query: str,
        documents: Iterable[str],
    ) -> NDArray[np.floating]:

        return self.reranker.predict(
            [(query, document) for document in documents]
        )


    def rank_sort_filter_documents(
        self,
        query: str,
        document_frame: pl.DataFrame,
        rerank_score_threshold: Number,
    ) -> pl.DataFrame:

        return document_frame.with_columns(
            pl.Series(
                'rerank score',
                self.rank_documents(query, document_frame['content'].to_list()),
                dtype = pl.Float32,
            ),
        ).filter(
            pl.col('rerank score') >= rerank_score_threshold
        ).sort(
            by = 'rerank score',
            descending = True,
        )


    def get_top_k_documents(
        self,
        query: str,
        k: int,
        content_size_limit: Number,
        filters: Iterable[pl.Expr] = [],
        k_multiplier: Number = 2.,
        weighted_rank_threshold: Number = 0.,
        fusion_factor: Number = 1,
        rerank: bool = True,
        rerank_score_threshold: Number = -np.inf,
        verbose: bool = False,
    ) -> pl.DataFrame:

        if k_multiplier < 1:
            raise ValueError(f'k_multipler must exceed 1.')

        relevant_documents = self.score_documents(
            query, 
            filters = filters, 
            fusion_factor = fusion_factor,
        )
        filtered_documents = relevant_documents.filter(
            pl.col('fused weighted reciprocal rank') > weighted_rank_threshold
        )[:round(k_multiplier * k)]
        if rerank and filtered_documents.shape[0]:
            reranked_documents = self.rank_sort_filter_documents(
                query,
                filtered_documents,
                rerank_score_threshold,
            )
        else:
            reranked_documents = filtered_documents
        final_documents = reranked_documents[:k].filter(
            pl.col('content size').cum_sum() < content_size_limit,
        )
        if verbose:
            print(f'{relevant_documents.shape[0]} relevant_documents.')
            print(f'{filtered_documents.shape[0]} filtered_documents.')
            print(f'{reranked_documents.shape[0]} reranked_documents.')
            print(f'{final_documents.shape[0]} final_documents.')
        return final_documents


    async def aget_top_k_documents(
        self,
        query: str,
        k: int,
        content_size_limit: Number,
        filters: Iterable[pl.Expr] = [],
        k_multiplier: Number = 2.,
        weighted_rank_threshold: Number = 0.,
        fusion_factor: Number = 1,
        rerank: bool = True,
        rerank_score_threshold: Number = -np.inf,
        verbose: bool = False,
    ) -> pl.DataFrame:

        if k_multiplier < 1:
            raise ValueError(f'k_multipler must exceed 1.')

        relevant_documents = await self.ascore_documents(
            query, 
            filters = filters, 
            fusion_factor = fusion_factor,
        )
        filtered_documents = relevant_documents.filter(
            pl.col('fused weighted reciprocal rank') > weighted_rank_threshold
        )[:round(k_multiplier * k)]
        if rerank and filtered_documents.shape[0]:
            reranked_documents = self.rank_sort_filter_documents(
                query,
                filtered_documents,
                rerank_score_threshold,
            )
        else:
            reranked_documents = filtered_documents
        final_documents = reranked_documents[:k].filter(
            pl.col('content size').cum_sum() < content_size_limit,
        )
        if verbose:
            print(f'{relevant_documents.shape[0]} relevant_documents.')
            print(f'{filtered_documents.shape[0]} filtered_documents.')
            print(f'{reranked_documents.shape[0]} reranked_documents.')
            print(f'{final_documents.shape[0]} final_documents.')
        return final_documents