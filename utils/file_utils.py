# 文件、文件夹工具类
import os


def dfs_files(root_path):
    """
    使用深度优先搜索遍历文件夹，返回所有文件的绝对路径
    """
    files = []
    stack = [root_path]
    while stack:
        current_path = stack.pop()
        s = os.listdir(current_path)
        for item in os.listdir(current_path):
            item_path = os.path.join(current_path, item)
            if os.path.isfile(item_path):
                files.append(os.path.abspath(item_path))
            elif os.path.isdir(item_path):
                stack.append(item_path)
    return files


if __name__ == '__main__':
   files = dfs_files("D:\\test")
   print(files)
