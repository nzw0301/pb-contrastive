import logging


def get_logger(terminator=''):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.terminator = terminator
    logger.addHandler(stream_handler)
    return logger
