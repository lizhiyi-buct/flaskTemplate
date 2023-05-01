import torch
import tqdm
from torch.utils.data import DataLoader, Dataset
from components import logger


# 日志类
log = logger.get_logger("test")


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

