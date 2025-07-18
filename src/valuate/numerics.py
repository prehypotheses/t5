"""Module numerics.py"""
import dask
import numpy as np
import sklearn.metrics as sm


class Numerics:
    """
    Calculates the error matrix labels per unique label
    """

    def __init__(self, originals: list[str], predictions: list[str]):
        """

        :param originals: The list of original labels
        :param predictions: The list of predicted labels
        """

        self.__originals: np.ndarray = np.array(originals)
        self.__predictions: np.ndarray = np.array(predictions)

    # noinspection PyUnresolvedReferences
    def __measures(self, name: str) -> dict:
        """

        :param name: The name of one of the labels
        :return:
        """

        _true: np.ndarray = (self.__originals == name).astype(int)
        _prediction: np.ndarray =  (self.__predictions == name).astype(int)

        tn, fp, fn, tp = sm.confusion_matrix(
            y_true=_true, y_pred=_prediction).ravel()

        return {name: {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}}

    def exc(self) -> dict:
        """

        :return:
        """

        # The set of unique labels
        names = np.unique(self.__originals)

        # The error matrix measures per unique label
        objects = [dask.delayed(self.__measures)(name) for name in names]
        calculations = dask.compute(objects)[0]

        # The error matrix measures per label in dict form
        structure = {str(k): v for c in calculations for k, v in c.items()}

        return structure
