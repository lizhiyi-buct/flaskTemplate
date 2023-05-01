from sqlalchemy import PrimaryKeyConstraint

from components.exts import db
from datetime import datetime


# 记录表
class Predict(db.Model):
    __tablename__ = 'predict'
    # 数据uuid
    data_id = db.Column(db.String(100), primary_key=True)
    # 模型uuid
    model_id = db.Column(db.String(100), primary_key=True)
    # 预测结果的地址
    res_addr = db.Column(db.String(1000), nullable=False)
    # 创建时间
    create_time = db.Column(db.DateTime, default=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    # 是否处理完成,0未完成,1已完成
    is_completed = db.Column(db.Integer, nullable=False)
    # 联合主键
    __table_args__ = (
        PrimaryKeyConstraint('data_id', 'model_id'),
        {},
    )
