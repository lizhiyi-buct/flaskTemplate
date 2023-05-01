import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch_model import CNN, CNN1
from collections import defaultdict
from components import logger, db
from datetime import datetime
import matplotlib.pyplot as plt
import tqdm
from torch.utils.data import DataLoader, Dataset
from models import CNNModel
from utils import UUIDGenerator
from config import DATA_ADDR

np.random.seed(7)
choose_index = np.random.randint(1, 100, 100)
num_epochs = 5
batch_size = 32
learning_rate = 0.001

# 日志类
log = logger.get_logger("train")


def get_batch(X, y, batch_size):
    index = np.random.randint(0, len(y), batch_size)
    return X[index, :], y[index]


class ECGDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        if self.label is None:
            return self.data[index]
        x = self.data[index]
        y = self.label[index]
        return x, y

    def __len__(self):
        return len(self.data)


# 加载数据第一步
def loadData():
    log.info("load data...")
    xtest = pd.read_csv('D:\Project\Money\\flaskTemplate\data\ECG5000\ECG5000_TEST.tsv', sep='\t').values
    xtrain = pd.read_csv('D:\Project\Money\\flaskTemplate\data\ECG5000\ECG5000_TRAIN.tsv', sep='\t').values
    all_data = np.concatenate([xtest, xtrain])
    X = all_data[:, 1:].astype('float32')
    Y = all_data[:, 0].astype('int32') - 1
    xnolabel = pd.read_csv('D:\Project\Money\\flaskTemplate\data\\test\\test.csv').values.astype('float32')
    ss = StandardScaler()  # 创建一个 StandardScaler 对象 ss，用于对数据进行标准化处理。
    std_data = ss.fit_transform(X)  # 使用 StandardScaler 对象对 X 进行标准化处理，并将处理结果存储在 std_data 变量中
    X = np.expand_dims(X, axis=2)
    std_data1 = ss.fit_transform(xnolabel)
    xnolabel = np.expand_dims(xnolabel, axis=2)
    print("begin StratifiedShuffleSplit...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, train_size=0.9,
                                 random_state=0)  # n_split=1就只有二八分，如果需要交叉验证，把训练和测试的代码放到for循环里面就可以
    sss.get_n_splits(X, Y)
    for train_index, test_index in sss.split(X, Y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        y_train = np.eye(5)[y_train]  # 将标签 y_train 转换成 one-hot 编码格式。
    X_train = torch.tensor(X_train)
    X_train = X_train.transpose(1, 2)
    X_test = torch.tensor(X_test)
    X_test = X_test.transpose(1, 2)
    y_train = torch.tensor(y_train)
    y_train = torch.argmax(y_train, dim=1)
    xnolabel = torch.tensor(xnolabel)
    xnolabel = xnolabel.transpose(1, 2)
    log.info("load data over...")
    return X_train, y_train, X_test, y_test, xnolabel


# 加载数据第二步，在训练CNN之后进行
def loadData2(xnolabel, cnn, X_train, y_train, X_test, y_test):
    log.info("load data2...")
    datasetnolabel = ECGDataset(xnolabel, None)
    dataloadernolabel = DataLoader(datasetnolabel, batch_size=32, shuffle=False)
    ynolabel_label = []
    for i, (x) in tqdm.tqdm(enumerate(dataloadernolabel)):
        # x = x.cuda()
        y_hat = cnn(x)  # 标签概率
        y_hat_idx = torch.argmax(y_hat, dim=1)  # 取概率最大的标签作为最后的标签
        y_hat_p, _ = torch.max(y_hat, dim=1)
        y_hat_idx = torch.where(y_hat_p > 0.8, y_hat_idx, torch.ones_like(y_hat_idx) * (-1))
        idx = np.argwhere(np.logical_or(y_hat_idx.cpu().numpy() == 0, y_hat_idx.cpu().numpy() == 1)).squeeze()
        # 修复了一个bug
        if idx.ndim == 0:
            continue
        tmp = idx.shape[0]
        if int(tmp) == 0:
            continue
        y_train = torch.cat([y_train, y_hat_idx.cpu()[idx]])
        X_train = torch.cat([X_train, x.cpu()[idx]])
    log.info("load data2 over...")
    return X_train, y_train


# 训练CNN
def trainModelCNN(model_name):
    log.info("train cnn...")
    cnn = CNN()
    opt = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    trainclass_correct = defaultdict(int)
    trainclass_total = defaultdict(int)
    loss_list = []
    X_train, y_train, X_test, y_test, xnolabel = loadData()
    # cnn = cnn.cuda()
    for epoch in range(num_epochs):
        print(f'===========epoch {epoch}=============')
        num_batches = int(len(y_train) // batch_size)
        right_cnt = 0
        total = 0
        loss_sum = 0
        for batch_index in tqdm.tqdm(range(num_batches)):
            X, y = get_batch(X_train, y_train, batch_size)
            # X = X.cuda()
            # y = y.cuda()
            y_hat = cnn(X)
            loss = loss_fn(y_hat, y)
            y_hat_idx = torch.argmax(y_hat, dim=1)
            for i in range(len(y)):
                label = y[i].item()
                prediction = y_hat_idx[i].item()
                trainclass_correct[label] += int(prediction == label)
                trainclass_total[label] += 1
            right_cnt += torch.sum(y_hat_idx == y)
            total += len(y)
            opt.zero_grad()
            loss.backward()
            loss_sum += loss.item()
            opt.step()
        loss_list.append(loss_sum / num_batches)

        right_cnt = right_cnt.item()
        print('right', right_cnt)
        print('total', total)
        print('train acc', right_cnt / total)
        for i in trainclass_correct:
            print('TrainAccuracy of %5s : %f %%' % (
                str(i), 100 * trainclass_correct[i] / trainclass_total[i]
            ))
    # plt.figure()
    # plt.plot(np.arange(0, len(loss_list)), loss_list)
    # plt.savefig('loss.png')
    # 开始第二步训练
    log.info("train cnn over...")
    trainModelCNN1(xnolabel, cnn, X_train, y_train, X_test, y_test, model_name)


# 训练CNN1
def trainModelCNN1(xnolabel, cnn, X_train, y_train, X_test, y_test, model_name):
    log.info("train cnn2...")
    X_train, y_train = loadData2(xnolabel, cnn, X_train, y_train, X_test, y_test)
    cnn1 = CNN1()
    opt1 = torch.optim.Adam(cnn1.parameters(), lr=learning_rate)
    loss_list2 = []
    trainclass2_correct = defaultdict(int)
    trainclass2_total = defaultdict(int)
    loss_fn = nn.CrossEntropyLoss()
    # cnn1 = cnn1.cuda()
    for epoch in range(num_epochs):
        print(f'===========epoch {epoch}=============')
        num_batches = int(len(y_train) // batch_size)
        right_cnt2 = 0
        total2 = 0
        loss_sum2 = 0
        for batch_index in tqdm.tqdm(range(num_batches)):
            X2, y2 = get_batch(X_train, y_train, batch_size)
            # X2 = X2.cuda()
            # y2 = y2.cuda()
            y_hat2 = cnn1(X2)
            loss2 = loss_fn(y_hat2, y2)
            y_hat_idx2 = torch.argmax(y_hat2, dim=1)
            for i in range(len(y2)):
                label2 = y2[i].item()
                prediction2 = y_hat_idx2[i].item()
                trainclass2_correct[label2] += int(prediction2 == label2)
                trainclass2_total[label2] += 1
            right_cnt2 += torch.sum(y_hat_idx2 == y2)
            total2 += len(y2)
            opt1.zero_grad()
            loss2.backward()
            loss_sum2 += loss2.item()
            opt1.step()
        loss_list2.append(loss_sum2 / num_batches)

        right_cnt2 = right_cnt2.item()
        print('right2', right_cnt2)
        print('total2', total2)
        print('train2 acc', right_cnt2 / total2)
        for i in trainclass2_correct:
            print('TrainAccuracy2 of %5s : %f %%' % (
                str(i), 100 * trainclass2_correct[i] / trainclass2_total[i]
            ))
    # plt.figure()
    # plt.plot(np.arange(0, len(loss_list2)), loss_list2)
    # plt.savefig('loss2.png')

    log.info("模型训练完成")
    torch.save(cnn1.state_dict(), model_name)


def trainAndSaveRecord(current_app):
    current_app = current_app[0]
    now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    model_addr = DATA_ADDR['model_save'] + os.path.sep + "cnn1-" + now + ".pth"
    # 新建模型训练记录
    uuid = UUIDGenerator.generate_uuid()
    model = CNNModel()
    model.id = uuid
    model.name = "cnn1-" + now + ".pth"
    model.addr = model_addr
    model.is_completed = 0
    with current_app.app_context():
        db.session.add(model)
        db.session.commit()
    # 模型训练
    trainModelCNN(model_addr)
    # 更新模型训练记录
    with current_app.app_context():
        update = CNNModel.query.filter_by(id=uuid).first()
        update.is_completed = 1
        db.session.commit()


