# common error

SERVER_BAD_REQUEST = (400, '请求语法不正确')

SERVER_LOGIN_REQUIRED = (401, '请登录')

SERVER_NO_PERMISSION = (403, '无权限')

SERVER_PATH_NOT_FOUND = (404, '请求路径不存在')

SERVER_METHOD_NOT_ALLOWED = (405, '请求方法不正确')

SERVER_PARAM_INVALID = (412, '请求参数不合法或缺失')

SERVER_HEADERS_REQUIRED = (417, '缺少必要请求头或者不正确')

SERVER_REQUEST_EXPIRED = (418, '请求已过期')

SERVER_INTERVAL_ERROR = (500, '服务器内部错误')


# business error

USER_REPEAT = (400, '用户名重复')

USER_NOT_FOUND = (400, '用户不存在')