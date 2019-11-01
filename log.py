import logging
from enum import Enum


class Logger(Enum):
    test = 'test',
    feature_norm = 'feature_norm',
    kernel = 'kernel'


class Log:
    def __init__(self):
        logging.basicConfig(level=logging.INFO, filename='/kw_resources/pg.log',
                            filemode='w', format='%(asctime)s-%(process)d-%(name)s\n%(levelname)s\n%(message)s')

        self.feature_norm = logging.getLogger(str(Logger.feature_norm))
        self.kernel = logging.getLogger(str(Logger.kernel))
        self.test = logging.getLogger(str(Logger.test))

    def print_log(self):
        pass
        # for item in self.printed_items
