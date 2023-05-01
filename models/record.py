from components.exts import db
from datetime import datetime


# 记录表
class Record(db.Model):
    __tablename__ = 'record'
    # uuid主键
    id = db.Column(db.String(100), primary_key=True)
    # 处理id，用-分隔
    ids = db.Column(db.String(100), nullable=False)
    # 预处理完成文件保存地址
    addr = db.Column(db.String(1000), nullable=False)
    # 是否处理完成,0未完成,1已完成
    is_completed = db.Column(db.Integer, nullable=False)
    # 创建时间
    create_time = db.Column(db.DateTime, default=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
