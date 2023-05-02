import pandas as pd
import wfdb
import pywt
from tensorflow.python.keras.layers.core import *
from tqdm import tqdm
from config import DATA_ADDR
import os
from utils import UUIDGenerator
from components import db, logger
from models import Record

# 日志类
log = logger.get_logger("process")


# 数组转字符串
def array_to_string(arr):
    return '-'.join([str(elem) for elem in arr])


# 小波去噪预处理
def denoise(data):
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # 阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata


# 读取心电数据和对应标签,并对数据进行小波去噪
def getDataSet(number, X_data, Y_data):
    ecgClassSet = ['N', 'A', 'V', 'L', 'R']
    # 读取心电数据记录
    log.info("正在读取 " + number + " 号心电数据...")
    record = wfdb.rdrecord(
        DATA_ADDR['addr'] + os.path.sep + number,
        channel_names=['MLII'])
    data = record.p_signal.flatten()
    rdata = denoise(data=data)
    # 获取心电数据记录中R波的位置和对应的标签
    annotation = wfdb.rdann(
        DATA_ADDR['addr'] + os.path.sep + number,
        'atr')
    Rlocation = annotation.sample
    Rclass = annotation.symbol
    # 去掉前后的不稳定数据
    start = 10
    end = 5
    i = start
    j = len(annotation.symbol) - end
    # 因为只选择NAVLR五种心电类型,所以要选出该条记录中所需要的那些带有特定标签的数据,舍弃其余标签的点
    # X_data在R波前后截取长度为140的数据点
    # Y_data将NAVLR按顺序转换为01234
    while i < j:
        try:
            # Rclass[i] 是标签
            lable = ecgClassSet.index(Rclass[i])
            x_train = rdata[Rlocation[i] - 49:Rlocation[i] + 91]
            X_data.append(x_train)
            Y_data.append(lable)
            i += 1
        except ValueError:
            i += 1
    return


# 数据处理子进程
def data_process(file_ids, current_app):
    file_ids = file_ids[0]
    current_app = current_app[0]
    if file_ids is not None and file_ids != []:
        # 生成文件名和存储地址
        # 生成UUID作为文件名和数据库主键
        uuid = UUIDGenerator.generate_uuid()
        sava_addr = DATA_ADDR['processed'] + os.path.sep + uuid + '.csv'
        # 在数据库中生成记录
        record = Record()
        record.id = uuid
        record.ids = array_to_string(file_ids)
        record.addr = sava_addr
        record.is_completed = 0
        with current_app.app_context():
            db.session.add(record)
            db.session.commit()

        dataSet = []
        labelSet = []
        log.info("开始读取数据")
        for n in tqdm(file_ids):
            getDataSet(n, dataSet, labelSet)
        dataset = None
        log.info("开始处理数据")
        for data in tqdm(dataSet):
            if dataset is None:
                dataset = data.reshape((1, -1))
                continue
            dataset = np.concatenate([dataset, data.reshape((1, -1))], axis=0)
        log.info("数据处理完成")
        pd.DataFrame(dataset).to_csv(sava_addr, header=False, index=False)
        # 记录更新
        with current_app.app_context():
            update = Record.query.filter_by(id=uuid).first()
            update.is_completed = 1
            db.session.commit()
    else:
        return None
