from flask import Blueprint, request, send_file, current_app
from config.env import DATA_ADDR, FILE_HASH
from utils.file_utils import dfs_files
from components.resFormat import resDTO, errDTO
from torch_model import data_process
from components import db
from models import Record
from concurrent.futures import ThreadPoolExecutor

filesBlueprint = Blueprint('files', __name__, url_prefix='/files')
# 异步执行工具
executor = ThreadPoolExecutor()


# 获得所有文件
@filesBlueprint.get("/getOriginList")
def get_origin_list():
    # 获取当前路径的上层路径
    data = dfs_files(DATA_ADDR['addr'])
    data = list(filter(lambda i: i.endswith('atr'), data))
    # 转为树状结构,用于前端展示
    # 在内存中更新维护一个hash表，表key为文件名，value为文件地址
    FILE_HASH.clear()
    for item in data:
        key = item.split("\\")[-1].split('.')[0]
        FILE_HASH[key] = item
    return resDTO(data=FILE_HASH)


# 选择要处理的序号，进行处理，异步返回
@filesBlueprint.post("/process_array")
def chose_file():
    data = request.get_json()
    chose = data['files']
    # 异步处理,需要将当前app以及上下文作为参数传给子进程
    app = current_app._get_current_object()
    executor.submit(data_process, (chose, ), (app, ))
    return resDTO()


# 查询所有文件处理结果
@filesBlueprint.get("/process_progress")
def get_progress():
    records = Record.query.all()
    res = []
    for item in records:
        item_data = {
            "uuid": item.id,
            "ids": item.ids,
            "addr": item.addr,
            "is_completed": item.is_completed,
            "create_time": item.create_time.strftime('%Y-%m-%d %H:%M:%S')
        }
        res.append(item_data)
    return resDTO(data=res)


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


# 根据文件序号查询文件处理结果,只支持传入一个序号
@filesBlueprint.get("/query_process_progress_id")
def queryRecordByID():
    record_id = request.args.get("id")
    records = Record.query.all()
    res = []
    for item in records:
        if record_id in item.ids:
            item_data = {
                "uuid": item.id,
                "ids": item.ids,
                "addr": item.addr,
                "is_completed": item.is_completed,
                "create_time": item.create_time.strftime('%Y-%m-%d %H:%M:%S')
            }
            res.append(item_data)
    return resDTO(data=res)


# 下载预处理后的文件
@filesBlueprint.post("/download")
def download_files():
    data = request.get_json()
    uuid = data.get("uuid")
    record = Record.query.filter_by(id=uuid).first()
    file_name = record.ids + ".csv"
    return send_file(record.addr, as_attachment=True, attachment_filename=file_name)
