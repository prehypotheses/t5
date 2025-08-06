"""Module lineage.py"""
import datetime
import time

import mlflow
import numpy as np
import pandas as pd
import sklearn.metrics as sm

import src.functions.directories
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
        self.__experiment_id = self.__get_experiment_id()

        # Instances
        self.__directories = src.functions.directories.Directories()

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

        frame = derivations.copy().loc[derivations['label'].isin(self.__labels), self.__fields]

        elements = frame.melt(id_vars='label', var_name='metric', value_name='score')
        elements['valuation'] = elements['label'] + '-' + elements['metric']
        elements.sort_values(by='valuation', ascending=True, inplace=True)

        dictionary = elements[['valuation', 'score']].set_index(
            keys='valuation').to_dict(orient='dict')['score']

        return dictionary

    def __get_experiment_id(self) -> str:
        """

        :return:
        """

        try:
            experiment = mlflow.get_experiment_by_name(self.__experiment.get('experiment_name'))
            return experiment.experiment_id
        except AttributeError:
            return mlflow.create_experiment(
                name=self.__experiment.get('experiment_name'),
                artifact_location=self.__experiment.get('artifact_location'),
                tags=self.__experiment.get('experiment_tags'))

    def exc(self, originals: list[str], predictions: list[str], stage: str):
        """
        local_path = os.path.join(self.__experiment.get('model_output_directory'), 'optimal', 'store', stage)
        self.__directories.create(local_path)
        mlflow.log_artifact(local_path=local_path, artifact_path=stage)


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

        # Logging
        mlflow.set_experiment(experiment_id=self.__experiment_id)
        with mlflow.start_run(experiment_id=self.__experiment_id, run_name=str(int(time.mktime(today.timetuple())))):
            mlflow.set_experiment_tag(key='stage', value=stage)
            mlflow.log_metrics(elements)
