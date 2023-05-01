def resDTO(code=200, msg="请求成功", data=None):
    if data is not None:
        return {"status_code": code, "message": msg, "data": data}
    else:
        return {"status_code": code, "message": msg}


def errDTO(code=500, msg="参数异常"):
    return {"status_code": code, "message": msg}
