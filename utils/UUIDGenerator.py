import uuid


# 用于生成UUID的工具类
class UUIDGenerator:

    @staticmethod
    def generate_uuid():
        return str(uuid.uuid4())


