from exts import db
from datetime import datetime


class StudentModel(db.Model):
    __tablename__ = 'student'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(100), nullable=False)
    sex = db.Column(db.String(1000), nullable=False)
    email = db.Column(db.String(100), nullable=False, unique=True)
    create_time = db.Column(db.DateTime, default=datetime.now)
