from logging.handlers import TimedRotatingFileHandler
import logging
import os
import inspect
from config import LOG_DIR


# 日志输出类
class Logger(object):
    _LOG_FORMAT = '%(asctime)-15s | %(levelname)s | %(message)s'

    _is_initialized = False

    def __init__(self, logger_name):
        if not Logger._is_initialized:
            logging.basicConfig(level=logging.INFO, format=Logger._LOG_FORMAT)
            Logger._is_initialized = True

        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(logging.INFO)

        log_file_path = os.path.join(LOG_DIR, '%s.log' % logger_name)
        file_handler = TimedRotatingFileHandler(log_file_path, when="d", interval=1, encoding='utf-8')  # 每周一个日志文件
        file_handler.setFormatter(logging.Formatter(Logger._LOG_FORMAT))
        file_handler.suffix = '%s.log' % file_handler.suffix
        self._logger.addHandler(file_handler)

    def _log_format(self, caller, message: str) -> str:
        file_name = caller[1].split('/')[-1]  # 日志记录所在的文件
        line_no = caller[2]                   # 日志记录所在的行数
        func_name = caller[3]                 # 日志记录所在的函数
        return '%s#%s:%s | %s\n' % (file_name, func_name, line_no, message)

    def debug(self, message: str) -> None:
        caller = inspect.stack()[1]
        self._logger.debug(self._log_format(caller, message))

    def info(self, message: str) -> None:
        caller = inspect.stack()[1]
        self._logger.info(self._log_format(caller, message))

    def warning(self, message: str) -> None:
        caller = inspect.stack()[1]
        self._logger.warning(self._log_format(caller, message))

    def error(self, message: str) -> None:
        caller = inspect.stack()[1]
        self._logger.error(self._log_format(caller, message))

    def exception(self, ex) -> None:
        self._logger.exception(ex)

    def get_logger(self) -> logging.Logger:
        return self._logger


_logger_map = dict()


def get_logger(logger_name=None) -> Logger:
    """返回一个Logger对象"""
    if logger_name is None:
        logger_name = os.path.split(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])[1]
    if logger_name not in _logger_map.keys():
        _logger_map[logger_name] = Logger(logger_name)
    return _logger_map.get(logger_name, None)
