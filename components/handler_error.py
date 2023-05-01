from config import error_codes
from werkzeug.exceptions import MethodNotAllowed, BadRequest, NotFound, Forbidden, Unauthorized
from components import get_logger
from config import MyBaseException
from flask import request, jsonify

HTTP_EXCEPTION_ERROR_CODE_MAP = {
    MethodNotAllowed: error_codes.SERVER_METHOD_NOT_ALLOWED,
    BadRequest: error_codes.SERVER_BAD_REQUEST,
    NotFound: error_codes.SERVER_PATH_NOT_FOUND,
    Unauthorized: error_codes.SERVER_LOGIN_REQUIRED,
    Forbidden: error_codes.SERVER_NO_PERMISSION
}

error_logger = get_logger("err")


def error_handler(e):
    """
    全局异常处理
    """
    if isinstance(e, MyBaseException):  # 自定义的一些错误
        error_code = (getattr(e, 'code'), getattr(e, 'msg'))
        data = getattr(e, 'data', None)
        return jsonify({
            'msg': error_code[1],
            'data': data
        }), error_code[0]

    error_code = HTTP_EXCEPTION_ERROR_CODE_MAP.get(type(e))
    if isinstance(e, AssertionError):  # 某些地方 assert 出错
        error_code = (400, str(e))
    if not error_code:  # 未知服务器内部代码错误
        error_code = error_codes.SERVER_INTERVAL_ERROR
        error_logger.exception(e)

    return jsonify({
        'msg': error_code[1]
    }), error_code[0]
