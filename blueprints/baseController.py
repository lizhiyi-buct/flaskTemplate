from flask import Blueprint

base = Blueprint('news', __name__, url_prefix='/news')


@base.route("/society/")
def society_news():
    return "社会新闻版块"


@base.route("/tech/")
def tech_news():
    return "IT 新闻板块"
