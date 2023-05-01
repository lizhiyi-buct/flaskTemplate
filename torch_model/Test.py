import torch
import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from components import logger, db
from models import Record, predict, Predict
import pandas as pd
import numpy as np
from torch_model import CNN1


# 日志类
log = logger.get_logger("test")


def get_batch(X,y,batch_size):
    index = np.random.randint(0, len(y), batch_size)
    return X[index, :], y[index]


class ECGDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        return x, y

    def __len__(self):
        return len(self.data)


def testByModel(cnn1, X_test, y_test):
    dataset = ECGDataset(X_test, y_test)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    right_cnt = 0
    total = 0
    testclass_total = [0] * 5
    testclass_correct = [0] * 5
    with torch.no_grad():
        for i, (x, y) in tqdm.tqdm(enumerate(dataloader)):
            x = x.cuda()
            y = y.cuda()
            y_hat = cnn1(x)  # 标签概率
            total += len(y)
            y_hat_idx = torch.argmax(y_hat, dim=1)  # 取概率最大的标签作为最后的标签
            right_cnt += torch.sum(y_hat_idx == y)
            for j in range(len(y)):
                testclass_total[y[j]] += 1
                testclass_correct[y[j]] += (y_hat_idx[j] == y[j]).item()
        right_cnt = right_cnt.item()
        print('right', right_cnt)
        print('total', total)
        print('test acc', right_cnt / total)
    for i in range(5):
        print('TestAccuracy of class %d : %f %%' % (i, 100 * testclass_correct[i] / testclass_total[i]))


# 传入预处理完成的文件uuid，生成测试数据
def generateTestSet(addr):
    all_data = pd.read_csv(addr).values.astype('float32')
    X = all_data[:, :].astype('float32')
    X_test = torch.tensor(X)
    temp = X_test.numpy()
    X_test = X_test.unsqueeze(1)
    return X_test, temp


# 加载模型通过模型文件保存地址
def loadModelByUUID(addr):
    cnn = CNN1()
    cnn.load_state_dict(torch.load(addr))
    # 设置评估模式
    cnn.eval()
    return cnn


# 模型分类
def evaTest(model_addr, set_addr, save_addr, pre_uuid, model_uuid, current_app):
    current_app = current_app[0]
    X_test, temp = generateTestSet(set_addr)
    model = loadModelByUUID(model_addr)
    y_hat = model(X_test)
    # 取概率最大的标签作为最后的标签
    y_hat_idx = torch.argmax(y_hat, dim=1).numpy()
    y_hat_idx = y_hat_idx.reshape(-1, 1)
    df = np.concatenate((y_hat_idx, temp), axis=1)
    pd.DataFrame(df).to_csv(save_addr, header=False, index=False)
    with current_app.app_context():
        # 更新数据
        update = Predict.query.filter_by(data_id=pre_uuid, model_id=model_uuid).first()
        update.is_completed = 1
        db.session.commit()
    return save_addr
