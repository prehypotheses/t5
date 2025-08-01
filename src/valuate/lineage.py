"""Module lineage.py"""
import datetime
import logging
import os
import time

import mlflow
import numpy as np
import pandas as pd
import sklearn.metrics as sm

import src.modelling.derivations


class Lineage:
    """
    Lineage
    """

    def __init__(self, id2label: dict, experiment: dict):
        """

        :param id2label: A dictionary wherein (a) the keys are the identification codes of text labels,
                         and (b) the values are the labels.
        :param experiment:
        """

        self.__labels = list(id2label.values())
        self.__fields = ['label', 'N', 'precision', 'sensitivity', 'fnr', 'f-score', 'matthews', 'b-accuracy']

        # Experiment
        self.__experiment = experiment
        mlflow.set_tracking_uri(uri=self.__experiment.get('uri'))
        mlflow.set_experiment(experiment_name=self.__experiment.get('experiment_name'))

    def __cases(self, originals: list[str], predictions: list[str]):
        """

        :param originals: The true values; a simple, un-nested, list.<br>
        :param predictions: The predictions; a simple, un-nested, list.<br>
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

    def __structure(self, derivations: pd.DataFrame) -> dict:
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

    def exc(self, originals: list[str], predictions: list[str], stage: str):
        """


        :param originals: The true values; a simple, un-nested, list.<br>
        :param predictions: The predictions; a simple, un-nested, list.<br>
        :param stage: Either training, testing, or validation
        """

        # A unique run identification code
        today: datetime.datetime = datetime.datetime.now()

        # Calculate
        cases = self.__cases(originals=originals, predictions=predictions)
        derivations = src.modelling.derivations.Derivations(cases=cases).exc()
        elements = self.__structure(derivations=derivations)

        # Log: artifact_path == artifact_location + stage ... model_output_directory optimal client
        with mlflow.start_run(run_name=str(int(time.mktime(today.timetuple()))),
                              experiment_id=self.__experiment.get('experiment_id')):

            mlflow.set_experiment_tags(tags={'stage': stage})
            mlflow.log_metrics(elements)
            mlflow.log_artifact(
                local_path=os.path.join(self.__experiment.get('model_output_directory'), 'optimal', 'store', stage),
                artifact_path=self.__experiment.get('artifact_location') + '/' + stage)
