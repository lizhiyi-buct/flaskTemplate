from flask import Blueprint, request, send_file, current_app
from config.env import DATA_ADDR, FILE_HASH
from utils.file_utils import dfs_files
from components.resFormat import resDTO, errDTO
from torch_model import data_process
from components import db
from models import Record
from concurrent.futures import ThreadPoolExecutor
from flask_paginate import Pagination

filesBlueprint = Blueprint('files', __name__, url_prefix='/files')
# 异步执行工具
executor = ThreadPoolExecutor()


# 获得所有文件, 已分页，只用传页码就行
@filesBlueprint.get("/getOriginList")
def get_origin_list():
    # 当前的页码，默认为1
    page = request.args.get("page", default=1, type=int)
    # 每页的数量
    page_size = request.args.get("page_size", default=15, type=int)
    # 总页数
    page_sum = 0
    # 总数据条数
    data_size = 0
    # 获取当前路径的上层路径
    data = dfs_files(DATA_ADDR['addr'])
    data = list(filter(lambda i: i.endswith('atr'), data))
    # 获得总条数
    data_size = len(data)
    # 计算总页数
    page_sum = (data_size + page_size - 1) // page_size
    offset = (page - 1) * page_size
    limit = page_size
    data_page = data[offset:offset + limit]
    res = []
    for item in data_page:
        id = item.split("\\")[-1].split('.')[0]
        item_data = {
            "id": id,
            "file_addr": item
        }
        res.append(item_data)
    result = {
      "total": data_size,
      "page": page,
      "page_size": page_size,
      "total_pages": page_sum,
      "data": res
    }
    return resDTO(data=result)


# 选择要处理的序号，可以传多个序号，进行处理，异步返回，如果已存在则不执行
@filesBlueprint.post("/process_array")
def chose_file():
    data = request.get_json()
    # 从小到大排序
    chose = data['files'].sort()
    ids = '-'.join([str(elem) for elem in chose])
    query = Record.query.filter_by(ids=ids).first()
    if query:
        # 已存在，直接返回
        return resDTO()
    # 异步处理,需要将当前app以及上下文作为参数传给子进程
    app = current_app._get_current_object()
    executor.submit(data_process, (chose, ), (app, ))
    return resDTO()


# 查询所有文件处理结果
@filesBlueprint.get("/process_progress")
def get_progress():
    # 当前的页码，默认为1
    page = request.args.get("page", default=1, type=int)
    # 每页的数量
    page_size = request.args.get("page_size", default=15, type=int)
    records = Record.query.paginate(page=page, per_page=page_size)
    res = []
    for item in records.items:
        item_data = {
            "uuid": item.id,
            "ids": item.ids,
            "addr": item.addr,
            "is_completed": item.is_completed,
            "create_time": item.create_time.strftime('%Y-%m-%d %H:%M:%S')
        }
        res.append(item_data)
    result = {
        "total": records.total,
        "page": records.page,
        "page_size": records.per_page,
        "total_pages": records.pages,
        "data": res
    }

    return resDTO(data=result)


# 根据uuid查询文件处理结果
@filesBlueprint.get("/query_process_progress_uuid")
def queryRecordByUUID():
    record_id = request.args.get("uuid")
    item = Record.query.filter_by(id=record_id)
    item_data = {
        "uuid": item.id,
        "ids": item.ids,
        "addr": item.addr,
        "is_completed": item.is_completed,
        "create_time": item.create_time.strftime('%Y-%m-%d %H:%M:%S')
    }
    return resDTO(data=item_data)


# 根据文件序号查询文件处理结果,只支持传入一个序号,分页完成
@filesBlueprint.get("/query_process_progress_id")
def queryRecordByID():
    # 当前的页码，默认为1
    page = request.args.get("page", default=1, type=int)
    # 每页的数量
    page_size = request.args.get("page_size", default=15, type=int)
    record_id = request.args.get("id")
    records = Record.query.filter(Record.ids.like('%'+record_id+"%")).paginate(page=page, per_page=page_size)
    res = []
    for item in records.items:
        item_data = {
            "uuid": item.id,
            "ids": item.ids,
            "addr": item.addr,
            "is_completed": item.is_completed,
            "create_time": item.create_time.strftime('%Y-%m-%d %H:%M:%S')
        }
        res.append(item_data)
    result = {
        "total": records.total,
        "page": records.page,
        "page_size": records.per_page,
        "total_pages": records.pages,
        "data": res
    }
    return resDTO(data=result)


# 下载预处理后的文件
@filesBlueprint.post("/download")
def download_files():
    data = request.get_json()
    uuid = data.get("uuid")
    record = Record.query.filter_by(id=uuid).first()
    file_name = record.ids + ".csv"
    return send_file(record.addr, as_attachment=True, attachment_filename=file_name)
