import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据库配置
HOSTNAME = '192.168.12.170'
PORT = 5432
USERNAME = 'postgres'
PASSWORD = '981224'
DATABASE = 'classification'
DB_URI = f"postgresql://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}"
SQLALCHEMY_DATABASE_URI = DB_URI

# 日志目录
LOG_DIR = os.getenv('LOG_DIR', os.path.join(BASE_DIR, 'logs'))

# 当前项目的目录
DATA_ADDR = {
    "addr": None,
    "processed": None,
    "model_save": None
}

# 维护的文件表
FILE_HASH = {}
