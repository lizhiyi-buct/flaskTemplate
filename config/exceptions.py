class MyBaseException(Exception):
    """
    异常基类，所有自定义异常均继承于此类
    用于区分是自定义异常，还是未知异常
    """
    pass


class BusinessException(MyBaseException):
    """业务相关异常，传对应错误码即可"""
    def __init__(self, error_code):
        self.code = error_code[0]
        self.msg = error_code[1]


class CommonException(MyBaseException):
    """通用异常 """
    def __init__(self, error_code, data=None, msg=None):
        self.code = error_code[0]
        self.msg = error_code[1]
        self.data = data
        if msg:
            self.msg = msg