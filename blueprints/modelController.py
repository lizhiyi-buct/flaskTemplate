from flask import Blueprint, request, current_app, send_file
from concurrent.futures import ThreadPoolExecutor
from torch_model import trainAndSaveRecord
from components.resFormat import resDTO, errDTO
from models import CNNModel, Record, Predict
from torch_model import evaTest
from utils import UUIDGenerator
from config import DATA_ADDR
import os
from components import db
from flask_paginate import Pagination

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


# 获得所有模型训练记录
@modelBlueprint.get("/all")
def get_process():
    # 当前的页码，默认为1
    page = request.args.get("page", default=1, type=int)
    # 每页的数量
    page_size = request.args.get("page_size", default=15, type=int)
    models = CNNModel.query.paginate(page=page, per_page=page_size)
    res = []
    for item in models.items:
        item_data = {
            "uuid": item.id,
            "name": item.name,
            "addr": item.addr,
            "is_completed": item.is_completed,
            "create_time": item.create_time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        res.append(item_data)
    result = {
        "total": models.total,
        "page": models.page,
        "page_size": models.per_page,
        "total_pages": models.pages,
        "data": res
    }
    return resDTO(data=result)


# 根据uuid查询模型
@modelBlueprint.get("/queryModel")
def queryModel():
    model_id = request.args.get("model_id")
    item = CNNModel.query.filter_by(id=model_id).first()
    item_data = {
        "uuid": item.id,
        "name": item.name,
        "addr": item.addr,
        "is_completed": item.is_completed,
        "create_time": item.create_time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    return resDTO(data=item_data)


# 选择测试数据和模型进行测试
@modelBlueprint.post("/test")
def test_model():
    data = request.get_json()
    # 预处理文件UUID
    pre_uuid = data['pre_uuid']
    # 模型UUID
    model_uuid = data['model_uuid']
    query = Predict.query.filter_by(data_id=pre_uuid, model_id=model_uuid).first()
    if query:
        return resDTO()
    item = Record.query.filter_by(id=pre_uuid).first()
    # 预处理文件的地址
    pre_addr = item.addr
    model = CNNModel.query.filter_by(id=model_uuid).first()
    model_addr = model.addr
    # 生成测试结果的uuid
    res_uuid = UUIDGenerator.generate_uuid()
    res_addr = DATA_ADDR['processed'] + os.path.sep + res_uuid + ".csv"
    # 数据入库
    pre_item = Predict()
    pre_item.data_id = pre_uuid
    pre_item.model_id = model_uuid
    pre_item.res_addr = res_addr
    pre_item.is_completed = 0
    db.session.add(pre_item)
    db.session.commit()
    # 测试方法
    app = current_app._get_current_object()
    executor.submit(evaTest, model_addr, pre_addr, res_addr, pre_uuid, model_uuid, (app, ))
    return resDTO()


# 获得所有分类记录
@modelBlueprint.get("/all_test_records")
def getAllTestRecords():
    # 当前的页码，默认为1
    page = request.args.get("page", default=1, type=int)
    # 每页的数量
    page_size = request.args.get("page_size", default=15, type=int)
    pres = Predict.query.paginate(page=page, per_page=page_size)
    res = []
    for item in pres.items:
        item_data = {
            "data_id": item.data_id,
            "model_id": item.model_id,
            "res_addr": item.res_addr,
            "is_completed": item.is_completed,
            "create_time": item.create_time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        res.append(item_data)
    result = {
        "total": pres.total,
        "page": pres.page,
        "page_size": pres.per_page,
        "total_pages": pres.pages,
        "data": res
    }
    return resDTO(data=result)


# 查询测试记录,通过uuid
@modelBlueprint.get("/queryTestByID")
def queryTestRecord():
    data_id = request.args.get("data_id")
    item_list = Predict.query.filter_by(data_id=data_id).all()
    res = []
    for item in item_list:
        item_data = {
            "data_id": item.data_id,
            "model_id": item.model_id,
            "res_addr": item.res_addr,
            "is_completed": item.is_completed,
            "create_time": item.create_time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        res.append(item_data)
    return resDTO(data=res)


# 根据数据id和模型id下载分类好的文件
@modelBlueprint.get("/downloadPredict")
def downloadPredict():
    data_id = request.args.get("data_id")
    model_id = request.args.get("model_id")
    item = Predict.query.filter_by(data_id=data_id, model_id=model_id).first()
    # 生成文件名称 ids-model.csv
    record = Record.query.filter_by(id=data_id).first()
    model = CNNModel.query.filter_by(id=model_id).first()
    file_name = record.ids + "-" + model.name + ".csv"
    return send_file(item.res_addr, as_attachment=True, attachment_filename=file_name)
