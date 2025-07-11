"""Module"""
import typing

import datasets


class Master(typing.NamedTuple):
    """
    The data type class ⇾ Master
    """

    id2label: dict
    label2id: dict
    data: datasets.DatasetDict
