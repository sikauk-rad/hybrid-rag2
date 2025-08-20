
from collections.abc import Sequence
from operator import itemgetter
from warnings import warn

from beartype import beartype
from llm_utilities.datatypes import AzureMessageCountType, AzureMessageType
from llm_utilities.utilities import strip_token_counts
import numpy as np

from .base import TokeniserInterface


@beartype
def add_token_count(
    message: AzureMessageType,
    tokeniser: TokeniserInterface,
) -> AzureMessageCountType:

    token_count = tokeniser.get_token_length(message['content'])
    return AzureMessageCountType(
        role = message['role'],
        content = message['content'],
        tokens = token_count,
    )


@beartype
def add_token_counts(
    messages: Sequence[AzureMessageCountType],
    tokeniser: TokeniserInterface,
) -> list[AzureMessageCountType]:

    token_counts = tokeniser.get_token_lengths([*map(itemgetter('content'), messages)])
    return [AzureMessageCountType(
        role = message['role'],
        content = message['content'],
        tokens = token_count,
    ) for message, token_count in zip(
        messages,
        token_counts,
        strict = True,
    )]


@beartype
def get_allowed_history(
    messages: Sequence[AzureMessageCountType],
    token_limit: int,
    strip_counts: bool = True,
    message_preservation_indices: Sequence[int] | None = None,
) -> list[AzureMessageCountType] | list[AzureMessageType]:

    """
    Retrieve a list of messages that fit within a specified token limit.

    Args:
        messages (list[AzureMessageCountType]): A list of messages, each containing a token count.
        token_limit (int): The maximum number of tokens allowed in the returned messages.
        strip_counts (bool): If True, the 'tokens' key will be removed from the returned messages (default is True).

    Returns:
        list[AzureMessageCountType]: A filtered list of messages that fit within the token limit.
    """

    if not messages:
        return []

    n_messages = len(messages)
    token_counts = np.fromiter(
        map(itemgetter('tokens'), messages),
        dtype = 'int64',
        count = n_messages,
    )
    message_indices = np.arange(n_messages, dtype = 'int64')

    if message_preservation_indices:
        message_indices_ordered_by_priority = np.append(
            message_preservation_indices,
            values=np.delete(message_indices, message_preservation_indices)[::-1],
        )
    else:
        message_indices_ordered_by_priority = message_indices[::-1]

    token_counts_ordered_by_priority = token_counts[message_indices_ordered_by_priority]
    within_token_limit_mask = token_counts_ordered_by_priority.cumsum() <= token_limit
    message_indices_within_token_limit = message_indices_ordered_by_priority[within_token_limit_mask]
    message_indices_within_token_limit.sort()

    if not message_indices_within_token_limit.size:
        return []

    elif message_indices_within_token_limit[-1] != (n_messages - 1):
        warn(
            'preserved messages are larger than token limit. Last user message not '\
            'sent to chat model.',
            category = UserWarning,
        )

    messages_within_token_limit = itemgetter(*message_indices_within_token_limit)(messages)
    if message_indices_within_token_limit.shape[0] < 2:
        messages_within_token_limit = [messages_within_token_limit]

    if not strip_counts:
        return messages_within_token_limit
    else:
        return strip_token_counts(messages_within_token_limit)