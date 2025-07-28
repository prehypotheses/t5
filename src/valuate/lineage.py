import datetime
import logging
import time

import mlflow
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

        # A unique run identification code
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        pattern = datetime.datetime.strptime(f'{today} 00:00:00', '%Y-%m-%d %H:%M:%S')
        self.__seconds = int(time.mktime(pattern.timetuple()))

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

    def exc(self, originals: list[str], predictions: list[str], name: str, tags: dict, stage: str, artefacts_: str):
        """

        :param originals: The true values, a simple, i.e., un-nested, list.<br>
        :param predictions: The predictions, a simple list, i.e., un-nested, list.<br>
        :param name: MlFlow experiment name
        :param tags: MlFlow overarching/common experiment tags
        :param stage: Either training, testing, or validation
        :param artefacts_: The location for the artefacts
        """

        client = mlflow.MlflowClient()
        register = client.create_experiment(name=name, artifact_location='', tags=tags)

        mlflow.set_experiment(experiment_name=name)
        mlflow.set_experiment_tag(key='stage', value=stage)

        # Calculate
        cases = self.__cases(originals=originals, predictions=predictions)
        derivations = src.modelling.derivations.Derivations(cases=cases).exc()
        elements = self.__structure(derivations=derivations)

        # Log
        with mlflow.start_run(run_name=str(self.__seconds)):
            mlflow.log_metrics(elements)
