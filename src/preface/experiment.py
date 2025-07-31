import mlflow
import datetime
import time
import boto3

import config
import src.functions.secret


class Experiment:

    def __init__(self, connector: boto3.session.Session):
        """
        :param connector: A boto3 session instance, it retrieves the developer's <default> Amazon
                          Web Services (AWS) profile details, which allows for programmatic interaction with AWS.
        """

        self.__configurations = config.Config()
        self.__secret = src.functions.secret.Secret(connector=connector)

        # Tracking Resource
        self.__tracking_uri = self.__get_tracking_uri()

        # A unique run identification code
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        pattern = datetime.datetime.strptime(f'{today} 00:00:00', '%Y-%m-%d %H:%M:%S')
        self.__seconds = int(time.mktime(pattern.timetuple()))

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

    def __client(self):
        """
        https://mlflow.org/docs/latest/ml/tracking/backend-stores/#supported-store-types
        https://mlflow.org/docs/latest/ml/getting-started/logging-first-model/step6-logging-a-run/#using-mlflow-tracking-to-keep-track-of-training
        mlflow.set_tracking_uri()

        t_bucket = self.__secret.exc(secret_id='FNTC', node='tracking-bucket')

        :return:
        """



        client = mlflow.MlflowClient()
        experiment_id = client.create_experiment(
            name=self.__configurations.experiment_name, artifact_location='',)


        dictionary = {'run_name': self.__seconds, 'experiment_id': ''}
