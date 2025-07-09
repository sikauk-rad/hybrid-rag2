from beartype import beartype
from .datatypes import AzureMessageCountType, AzureMessageType
from .base import TokeniserInterface
from operator import itemgetter
import numpy as np
from warnings import warn
from collections.abc import Sequence, Iterable
from numbers import Number
from typing import Literal, TypeVar
from hashlib import sha256

str_or_None = TypeVar("T", bound= str | None)


def check_all_arguments_are_none_or_not(
    *args,
) -> bool:

    """
    Check if all provided arguments are either None or not None.

    Args:
        *args: A variable number of arguments to check.

    Returns:
        bool: True if all arguments are None or all are not None; False otherwise.
    """

    all_none = [arg is None for arg in args]
    return not (any(all_none) and (not all(all_none)))


@beartype
def hash_repr_deterministic(
   string: str_or_None,
) -> str_or_None:

   if object is None:
      return None
   else:
      return sha256(f'{object}'.encode('utf-8')).hexdigest()


@beartype
def get_optimal_uintype(
    number: Number,
) -> Literal['uint8', 'uint16', 'uint32', 'uint64', 'float32', 'float64']:

    if number < 255:
        return 'uint8'
    elif number < 65535:
        return 'uint16'
    elif number < 4294967295:
        return 'uint32'
    elif number < 18446744073709551615:
        return 'uint64'
    elif number <= 3.4028235e+38:
        return 'float32'
    else:
        return 'float64'


@beartype
def strip_token_count(
    message: AzureMessageCountType,
) -> AzureMessageType:

    return {key: value for key, value in message.items() if key != 'tokens'}


@beartype
def strip_token_counts(
    messages: Iterable[AzureMessageCountType],
) -> list[AzureMessageType]:

    return [{
        'role': message['role'], 
        'content': message['content'],
    } for message in messages]


@beartype
def add_token_count(
    message: AzureMessageType,
    tokeniser: TokeniserInterface,
) -> AzureMessageCountType:

    token_count = tokeniser.get_token_length(message['content'])
    return message | {'tokens': token_count}


@beartype
def add_token_counts(
    messages: Sequence[AzureMessageCountType],
    tokeniser: TokeniserInterface,
) -> list[AzureMessageCountType]:

    token_counts = tokeniser.get_token_lengths([*map(itemgetter('content'), messages)])
    return [message | {'tokens': token_count} for message, token_count in zip(
        messages,
        token_counts,
    )]


@beartype
def get_allowed_history(
    messages: Sequence[AzureMessageCountType],
    token_limit: int,
    strip_counts: bool = True,
    message_preservation_indices: Sequence[int] | None = None,
) -> list[AzureMessageCountType]:

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
        return messages

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