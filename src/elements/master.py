"""Module"""
import typing

import datasets


class Master(typing.NamedTuple):
    """
    The data type class ⇾ Master

    id2label: dict
        A dictionary wherein (a) the keys are the identification codes of text labels, and (b) the values are the labels.
    label2id: dict
        A dictionary wherein (a) the values are the labels, and (b) the keys are the identification codes of text labels.
    data: datasets: DatasetDict
        The training, validation, and testing data; named `train`, `validation`, and `test`, respectively.
    """

    id2label: dict
    label2id: dict
    data: datasets.DatasetDict
