from flask import Flask
from flask_cors import CORS
from flask_migrate import Migrate
from blueprints import blueprint_list
from config import env, DATA_ADDR
from models import *
from components import error_handler
import os

app = Flask(__name__)
# 开启跨域支持
cors = CORS(app)
# 导入自定义配置
app.config.from_object(env)
# 初始化数据库
db.init_app(app)
# 数据库迁移
# 数据库映射三步,DRM改变只需要执行23
# flask db init 只需执行一次
# flask db migrate 识别DRM改变，生成迁移脚本
# flask db upgrade 运行脚本，同步数据库
migrate = Migrate(app, db)
# 注册所有蓝图
[app.register_blueprint(blueprint) for blueprint in blueprint_list]
# 注册异常
app.register_error_handler(Exception, error_handler)
# 训练集文件存储地址初始化
DATA_ADDR['addr'] = app.root_path + os.path.sep + "data"
# 预处理完成文件存储地址初始化
DATA_ADDR['processed'] = app.root_path + os.path.sep + "processed"
# 模型保存地址初始化
DATA_ADDR['model_save'] = app.root_path + os.path.sep + "torch_model" + os.path.sep + "model_save"

if __name__ == '__main__':
    # dev
    app.run(host="localhost", port=5000)

