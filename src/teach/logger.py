# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0


import logging
import sys

from teach.settings import get_settings


def create_logger(name: str = None, level=logging.DEBUG):
    teach_settings = get_settings()
    if teach_settings.GUNICORN_LOGGING:
        logger = logging.getLogger(name if name else __name__)
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.NullHandler())
        return logger
    else:
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(name if name else __name__)
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("[%(threadName)s-%(process)s-%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
