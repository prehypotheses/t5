import logging

import numpy as np
import pandas as pd
import sklearn.metrics as sm

import src.modelling.derivations


class Lineage:

    def __init__(self, id2label: dict):
        """

        :param id2label:
        """

        self.__id2label = id2label
        self.__labels = list(id2label.values())
        self.__fields = ['label', 'N', 'precision', 'sensitivity', 'fnr', 'f-score', 'matthews', 'b-accuracy']


    def __cases(self, originals: list[str], predictions: list[str]):
        """

        :param originals:
        :param predictions:
        :return:
        """

        measures = sm.confusion_matrix(y_true=originals, y_pred=predictions, labels=self.__labels)

        tp = np.diag(measures, k=0)
        fp = np.sum(measures, axis=0) - tp
        fn = np.sum(measures, axis=1) - tp

        tn = [int(measures.sum() - measures[:,k].sum() - measures[k,:].sum() + measures[k,k])
              for k in range(measures.shape[0])]

        frame = pd.DataFrame(
            data={'label': self.__labels, 'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn, 'N': np.sum(measures, axis=1)})

        return frame

    def __publish(self, derivations: pd.DataFrame) -> dict:
        """

        :param derivations: A data frame of error measures & metrics
        :return:
        """

        logging.info(derivations)

        frame = derivations.copy().loc[derivations['label'].isin(self.__labels), self.__fields]

        elements = frame.melt(id_vars='label', var_name='metric', value_name='score')
        elements['valuation'] = elements['label'] + '-' + elements['metric']
        elements.sort_values(by='valuation', ascending=True, inplace=True)

        dictionary = elements[['valuation', 'score']].set_index(
            keys='valuation').to_dict(orient='dict')['score']

        return dictionary

    def exc(self, originals: list[str], predictions: list[str]):
        """

        :param originals: The true values, a simple, i.e., un-nested, list.<br>
        :param predictions: The predictions, a simple list, i.e., un-nested, list.<br>
        """

        cases = self.__cases(originals=originals, predictions=predictions)
        derivations = src.modelling.derivations.Derivations(cases=cases).exc()
        elements = self.__publish(derivations=derivations)

        return elements
