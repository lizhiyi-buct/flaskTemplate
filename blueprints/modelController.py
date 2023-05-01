from flask import Blueprint, request, current_app
from concurrent.futures import ThreadPoolExecutor
from torch_model import trainAndSaveRecord
from components.resFormat import resDTO, errDTO
from models import CNNModel

modelBlueprint = Blueprint('model', __name__, url_prefix='/model')

# 异步执行工具
executor = ThreadPoolExecutor()


# 开启模型训练
@modelBlueprint.get("/train")
def train_model():
    # 异步处理,需要将当前app以及上下文作为参数传给子进程
    app = current_app._get_current_object()
    executor.submit(trainAndSaveRecord, (app, ))
    return resDTO()


# 获得所有训练记录
@modelBlueprint.get("/all")
def get_process():
    models = CNNModel.query.all()
    res = []
    for item in models:
        item_data = {
            "uuid": item.id,
            "name": item.name,
            "addr": item.addr,
            "is_completed": item.is_completed,
            "create_time": item.create_time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        res.append(item_data)
    return resDTO(data=res)


# 测试训练
@modelBlueprint.post("/test")
def test_model():
    data = request.get_json()
    # 预处理文件UUID
    pre = data['pre']
    # 模型UUID
    model = data['model']
    # 测试方法
    # 返回记录
