"""Module estimates.py"""
import logging
import typing

import datasets
import numpy as np
import transformers


class Estimates:
    """
    Determines the predictions w.r.t. (with respect to) a given data set.
    """

    def __init__(self, blob: datasets.Dataset, id2label: dict):
        """

        :param blob:
        """

        self.__blob = blob
        self.__id2label = id2label

        # Logging
        logging.basicConfig(level=logging.INFO, format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H-%M-%S')
        self.__logger = logging.getLogger(__name__)

    def exc(self, model: transformers.Trainer) -> typing.Tuple[list, list]:
        """

        :param model: The model; trained using the best set of hyperparameters.
        :return:
            labels: The codes of the original labels<br>
            predictions: The predicted codes
        """

        # The outputs bucket
        bucket = model.predict(self.__blob)
        __labels: np.ndarray = bucket.label_ids
        __predictions: np.ndarray = bucket.predictions
        self.__logger.info('Labels: %s', __labels.shape)
        self.__logger.info('Predictions: %s', __predictions.shape)

        # Reshaping
        ref = __labels.reshape(-1)
        matrix = __predictions.reshape(-1, model.model.config.num_labels)
        est = np.argmax(matrix, axis=1)

        # Active
        self.__logger.info('Determining active labels & predictions')
        active = np.not_equal(ref, -100)
        labels = ref[active]
        predictions = est[active]

        # Code -> tag
        labels_: list[str] = [self.__id2label[code.item()] for code in labels]
        predictions_: list[str] = [self.__id2label[code.item()] for code in predictions]

        return labels_, predictions_
