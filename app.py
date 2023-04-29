from flask import Flask
from config import env
from exts import db
from flask_cors import CORS
from flask_migrate import Migrate
from models import *
from blueprints import blueprint_list

app = Flask(__name__)
# 开启跨域
cors = CORS(app, supports_credentials=True)
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


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


if __name__ == '__main__':
    app.run()
