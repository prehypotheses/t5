"""Module script.py"""
import datetime
import logging
import os
import sys

import boto3
import ray
import torch


# noinspection DuplicatedCode
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

    # Data
    master: mr.Master = src.data.interface.Interface(s3_parameters=s3_parameters, arguments=arguments).exc()
    logger.info(master.id2label)
    logger.info(master.data)

    # Best, etc
    src.modelling.interface.Interface(
        arguments=arguments, hyperspace=hyperspace, experiment=experiment).exc(master=master)

    # Transfer
    messages = src.transfer.interface.Interface(
        service=service, s3_parameters=s3_parameters, arguments=arguments).exc(data=master.data)
    logger.info(messages)

    # Cache
    src.functions.cache.Cache().exc()

# noinspection DuplicatedCode
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
    import src.data.interface
    import src.elements.arguments as ag
    import src.elements.hyperspace as hp
    import src.elements.master as mr
    import src.elements.s3_parameters as s3p
    import src.elements.service as sr
    import src.functions.cache
    import src.modelling.interface
    import src.preface.interface
    import src.transfer.interface

    connector: boto3.session.Session
    s3_parameters: s3p.S3Parameters
    service: sr.Service
    arguments: ag.Arguments
    hyperspace: hp.Hyperspace
    experiment: dict
    connector, s3_parameters, service, arguments, hyperspace, experiment = src.preface.interface.Interface().exc()

    main()
