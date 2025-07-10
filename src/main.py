"""Module main.py"""
import datetime
import logging
import os
import sys

import boto3
import ray
import torch


def main():
    """

    :return:
    """

    logger: logging.Logger = logging.getLogger(__name__)
    logger.info('Starting: %s', datetime.datetime.now().isoformat(timespec='microseconds'))

    # Device Selection: Setting a graphics processing unit as the default device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('device: %s', device)

    # Ray
    ray.init(dashboard_host='172.17.0.2', dashboard_port=8265)

    # Grid
    grid = src.modelling.steps.Steps(s3_parameters=s3_parameters, arguments=arguments, hyperspace=hyperspace).exc()
    logger.info(grid.get_dataframe())
    logger.info(grid.get_best_result())

    # Cache
    src.functions.cache.Cache().exc()


if __name__ == '__main__':

    root = os.getcwd()
    sys.path.append(root)
    sys.path.append(os.path.join(root, 'src'))

    # Logging
    logging.basicConfig(level=logging.INFO,
                        format='\n\n%(message)s\n%(asctime)s.%(msecs)03d\n',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # Activate graphics processing units
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    os.environ['TOKENIZERS_PARALLELISM']='true'
    os.environ['RAY_USAGE_STATS_ENABLED']='0'
    os.environ['HF_HOME']='/tmp'

    # Classes
    import src.elements.arguments as ag
    import src.elements.hyperspace as hp
    import src.elements.s3_parameters as s3p
    import src.elements.service as sr
    import src.functions.cache
    import src.modelling.steps
    import src.preface.interface

    connector: boto3.session.Session
    s3_parameters: s3p.S3Parameters
    service: sr.Service
    arguments: ag.Arguments
    hyperspace: hp.Hyperspace
    connector, s3_parameters, service, arguments, hyperspace = src.preface.interface.Interface().exc()

    main()
