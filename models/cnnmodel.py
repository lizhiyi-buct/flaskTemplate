from components.exts import db
from datetime import datetime


# 记录表
class CNNModel(db.Model):
    __tablename__ = 'cnnmodel'
    # uuid主键
    id = db.Column(db.String(100), primary_key=True)
    # 模型名称
    name = db.Column(db.String(100), nullable=False)
    # 模型保存地址
    addr = db.Column(db.String(1000), nullable=False)
    # 是否训练完成,0未完成,1已完成
    is_completed = db.Column(db.Integer, nullable=False)
    # 创建时间
    create_time = db.Column(db.DateTime, default=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
