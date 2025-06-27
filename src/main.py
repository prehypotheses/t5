"""Module main.py"""
import datetime
import logging
import os
import sys

import boto3


def main():
    """

    :return:
    """

    logger: logging.Logger = logging.getLogger(__name__)
    logger.info('Starting: %s', datetime.datetime.now().isoformat(timespec='microseconds'))

    # Background
    logger.info(s3_parameters)
    logger.info(arguments)
    logger.info(hyperspace)

    # Modelling
    src.modelling.interface.Interface(s3_parameters=s3_parameters).exc()

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

    # Classes
    import src.elements.arguments as ag
    import src.elements.hyperspace as hp
    import src.elements.s3_parameters as s3p
    import src.elements.service as sr
    import src.functions.cache
    import src.modelling.interface
    import src.preface.interface

    connector: boto3.session.Session
    s3_parameters: s3p.S3Parameters
    service: sr.Service
    arguments: ag.Arguments
    hyperspace: hp.Hyperspace
    connector, s3_parameters, service, arguments, hyperspace = src.preface.interface.Interface().exc()

    main()
