"""Module experiment.py"""
import logging
import boto3
import mlflow

import src.elements.arguments as ag
import src.functions.secret


class Experiment:
    """

    <a href="https://mlflow.org/docs/latest/ml/tracking/tracking-api/#setup--configuration" target="_blank">
        set-up configurations</a><br>
    <a href="https://mlflow.org/docs/latest/genai/getting-started/connect-environment" target="_blank">
        connecting</a><br>
    <a href="https://mlflow.org/docs/latest/ml/tracking/artifact-stores" target="_blank">
        artefact stores</a><br>
    <a href="https://mlflow.org/docs/latest/ml/deep-learning/transformers/guide/#logging-a-components-based-model"
        target="_blank">logging a components based model</a><br>
    <a href="https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html#mlflow.start_run" target="_blank">
        mlflow.start_run()</a><br>
    <a href="https://mlflow.org/docs/latest/ml/tracking/backend-stores/#supported-store-types" target="_blank">
        store types</a><br>
    <a href="https://mlflow.org/docs/latest/ml/getting-started/logging-first-model/step6-logging-a-run/
        #using-mlflow-tracking-to-keep-track-of-training">tracking</a>
    """

    def __init__(self, connector: boto3.session.Session, arguments: ag.Arguments):
        """
        :param connector: A boto3 session instance, it retrieves the developer's <default> Amazon
                          Web Services (AWS) profile details, which allows for programmatic interaction with AWS.
        :param arguments: A set of arguments for the model development classes/modules.
        """

        self.__arguments = arguments

        # Secrets
        self.__secret = src.functions.secret.Secret(connector=connector)

    def __get_tracking_uri(self) -> str:
        """

        :return:
        """

        t_secret  = self.__secret.exc(secret_id='FNTC', node='tracking-secret')
        t_endpoint = self.__secret.exc(secret_id='FNTC', node='tracking-endpoint')
        t_database = self.__secret.exc(secret_id='FNTC', node='tracking-database')
        t_port = self.__secret.exc(secret_id='FNTC', node='tracking-port')

        username = self.__secret.exc(secret_id=t_secret, node='username')
        password = self.__secret.exc(secret_id=t_secret, node='password')

        uri: str = f"postgresql://{username}:{password}@{t_endpoint}:{t_port}/{t_database}"

        return uri

    def __get_backend_details(self) -> str:
        """

        :return:
        """

        bucket = self.__secret.exc(secret_id='FNTC', node='tracking-bucket')
        architecture = self.__arguments.architecture.upper()

        return f's3://{bucket}/{architecture}/{self.__arguments.experiment_segment}/'

    def exc(self) -> dict:
        """

        :return:
        """

        return {'experiment_name': self.__arguments.experiment_name,
                'experiment_tags': self.__arguments.experiment_tags,
                'artifact_location': self.__get_backend_details(),
                'uri': self.__get_tracking_uri(),
                'model_output_directory': self.__arguments.model_output_directory}
