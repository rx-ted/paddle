# coding: utf-8

# 项目4-语句分类


# 用来过滤警告
from paddle.io import BatchSampler, DataLoader
import paddle.nn as nn
from paddle.io import Dataset
from gensim.models import Word2Vec
from gensim.models import word2vec
import argparse
import os
import pandas as pd
import numpy as np
import paddle
import warnings
warnings.filterwarnings('ignore')
# utils.py
# 这个块用来先定义一些等等常用到的函数
paddle.disable_static()


def load_training_data(path='training_label.txt'):
    # 把训练时需要的数据读进来
    # 如果是 'training_label.txt'，需要读取标签，如果是 'training_nolabel.txt'，不需要读取标签
    if 'training_label' in path:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.strip('\n').split(' ') for line in lines]
        x = [line[2:] for line in lines]
        y = [line[0] for line in lines]
        return x, y
    else:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            x = [line.strip('\n').split(' ') for line in lines]
        return x


def load_testing_data(path='testing_data'):
    # 把测试时需要的数据读进来
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        X = ["".join(line.strip('\n').split(",")[1:]).strip()
             for line in lines[1:]]
        X = [sen.split(' ') for sen in X]
    # print("X", X[:10])
    return X


def evaluation(outputs, labels):
    # outputs => probability (float)
    # labels => labels

    outputs = paddle.to_tensor(
        [1.0 if element >= 0.5 else 0.0 for element in outputs])
    labels = labels.squeeze(1)
    correct = paddle.sum(paddle.cast(paddle.equal(
        outputs, labels), dtype="int64")).numpy()
    return correct


def train_word2vec(x):
    # 训练word to vector的词向量
    model = word2vec.Word2Vec(x, vector_size=250, window=5,
                              min_count=5, workers=12, epochs=10, sg=1)
    return model


# preprocess.py
# 这个块用来做数据的预处理
# 实现了dataset所需要的 '__init__', '__getitem__', '__len__'
# 好让 dataloader 能使用
class Preprocess():
    def __init__(self, sentences, sen_len, w2v_path="./w2v.model"):
        self.w2v_path = w2v_path
        self.sentences = sentences
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []

    def get_w2v_model(self):
        # 把之前训练好的 word to vec 模型读进来
        self.embedding = word2vec.Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size

    def add_embedding(self, word):
        # 把词加进 embedding，并赋予他一个随机生成的表示向量
        # 词只会是 "<PAD>" 或 "<UNK>"
        vector = np.random.uniform(size=(1, self.embedding_dim))
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = np.concatenate(
            [self.embedding_matrix, vector], 0)

    def make_embedding(self, load=True):
        print("Get embedding ...")
        # 取得训练好的 Word2vec词向量
        if load:
            print("loading word to vec model ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError
        # 制作一个 word2idx 的 字典
        # 制作一个 idx2word 的 列表
        # 制作一个 word2vector 的 列表

        for i, word in enumerate(self.embedding.wv.index_to_key):
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding.wv[word])
        # print('')
#        self.embedding_matrix = paddle.to_tensor(self.embedding_matrix)
        self.embedding_matrix = np.array(self.embedding_matrix)
        # 将 "<PAD>" 跟 "<UNK>" 加进 embedding 里面
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        # print("total words: {}".format(len(self.embedding_matrix)))
        self.embedding_matrix = self.embedding_matrix.astype(np.float32)
        return self.embedding_matrix

    def pad_sequence(self, sentence):
        # 将每个句子变成一样的长度
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        assert len(sentence) == self.sen_len
        return sentence

    def sentence_word2idx(self):
        # 把句子里面的字转成相对应的索引
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            # print('sentence count #{}'.format(i+1), end='\r')
            sentence_idx = []
            for word in sen:
                if (word in self.word2idx.keys()):
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx["<UNK>"])
            # 将每个句子变成一样的长度
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return paddle.to_tensor(sentence_list)

    def labels_to_tensor(self, y):
        # 把标签转成张量
        y = [float(label) for label in y]
        return paddle.to_tensor(y)


class TwitterDataset(Dataset):
    """
    Expected data shape like:(data_num, data_len)
    Data can be a list of numpy array or a list of lists
    input data shape : (data_num, seq_len, feature_dim)
    __len__ will return the number of data
    """

    def __init__(self, X, y):
        self.data = X
        self.label = y

    def __getitem__(self, idx):
        if self.label is None:
            return self.data[idx], paddle.to_tensor([1.])
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


class LSTM_Net(nn.Layer):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_embedding=True):
        super(LSTM_Net, self).__init__()
        # 制作 embedding layer
        # self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        # self.embedding.weight = torch.nn.Parameter(embedding)
#         if fix_embedding:
#             w_param_attrs = paddle.ParamAttr(trainable=False)
#         else:
#             w_param_attrs = paddle.ParamAttr(trainable=True)
#         self.embedding = nn.Embedding((embedding.shape[0],embedding.shape[1]), param_attr= w_param_attrs)
        self.embedding = nn.Embedding(
            embedding.shape[0], embedding.shape[1], sparse=True)
        self.embedding.weight.set_value(embedding)
        self.embedding.weight.requires_grad = False if fix_embedding else True

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers)
        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(hidden_dim, 1),
                                        nn.Sigmoid())

    def forward(self, inputs):
        inputs = self.embedding(inputs)
#         print("embedding",inputs)
        x, _ = self.lstm(inputs, None)
        # x 的 维度 (batch, seq_len, hidden_size)
        # 取用 LSTM 最后一层的隐藏状态
        x = x[:, -1, :]
        x = self.classifier(x)
        return x


# 这个块是用来训练模型的

def training(batch_size, n_epoch, lr, model_dir, train, valid, model):
    model.train()  # 将模型的模式设为 train，这样优化器就可以更新模型的参数
    criterion = paddle.nn.loss.BCELoss()  # 定义损失函数，这裡我们使用 二元交叉熵损失
    t_batch = len(train)
    v_batch = len(valid)
    optimizer = paddle.optimizer.Adam(
        learning_rate=lr, parameters=model.parameters())  # 将模型的参数给优化器，并给予适当的学习率
    total_loss, total_acc, best_acc = 0, 0, 0
    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0
        # 这段做训练
        for i, (inputs, labels) in enumerate(train):
            optimizer.clear_grad()  # 由于 loss.backward() 的梯度会累加，所以每次喂完一个 batch 后需要归零
            outputs = model(inputs)  # 将输入喂给模型
            loss = criterion(outputs, labels)  # 计算此时模型的训练损失
            loss.backward()  # 算损失的梯度
            optimizer.step()  # 更新训练模型的参数
            correct = evaluation(outputs, labels)  # 计算此时模型的训练准确率

            total_acc += (correct / batch_size)
            total_loss += loss.numpy()

            print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
                epoch+1, i+1, t_batch, loss.numpy()[0], correct[0]*100/batch_size), end='\r')
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(
            total_loss[0]/t_batch, total_acc[0]/t_batch*100))

        # 这段做验证
        model.eval()  # 将模型的模式设为eval，这样模型的参数就会固定住
#         with torch.no_grad():
        total_loss, total_acc = 0, 0
        for i, (inputs, labels) in enumerate(valid):
            outputs = model(inputs)  # 将输入喂给模型

            loss = criterion(outputs, labels)  # 计算此时模型的验证损失
            correct = evaluation(outputs, labels)  # 计算此时模型的验证准确率
            total_acc += (correct / batch_size)
            total_loss += loss.numpy()

        print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(
            total_loss[0]/v_batch, total_acc[0]/v_batch*100))
        if total_acc > best_acc:
            # 如果验证的结果优于之前所有的结果，就把当下的模型存下来以备之后做预测时使用
            best_acc = total_acc
            #torch.save(model, "{}/val_acc_{:.3f}.model".format(model_dir,total_acc/v_batch*100))
            paddle.save(model.state_dict(), "lstm_crf.pdparams")
            paddle.save(optimizer.state_dict(),  "lstm_crf.pdopt")
            print('saving model with acc {:.3f}'.format(
                total_acc[0]/v_batch*100))
        print('-----------------------------------------------')
        model.train()  # 将模型的模式设为 train，这样优化器就可以更新模型的参数（因为刚刚转成 eval 模式）


# 测试
def testing(batch_size, test_loader, model):
    model.eval()
    ret_output = []
    with paddle.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            outputs = model(inputs)
            outputs = [1 if element >= 0.5 else 0 for element in outputs]
            ret_output += outputs
    return ret_output


# 主函数
path_prefix = './'
# 处理好各个数据的路径
train_with_label = os.path.join(path_prefix, 'training_label.txt')
train_no_label = os.path.join(path_prefix, 'training_nolabel.txt')
testing_data = os.path.join(path_prefix, 'testing_data.txt')

# 处理 word to vec model 的路径
w2v_path = os.path.join(path_prefix, 'w2v_all.model')

# 定义句子长度、要不要固定 embedding、批次大小、要训练几个 epoch、学习率的值、模型的资料夹路径
sen_len = 20
fix_embedding = True  # 保持训练时的嵌入不变
batch_size = 128*8
epoch = 5
lr = 0.001
# model_dir = os.path.join(path_prefix, 'model/') # 检查点模型的模型目录
model_dir = path_prefix  # 检查点模型的模型目录

print("loading data ...")  # 把 'training_label.txt' 跟 'training_nolabel.txt' 读进来
train_x, y = load_training_data(train_with_label)
train_x_no_label = load_training_data(train_no_label)
test_x = load_testing_data(testing_data)
# keep /w2v.model
# model = train_word2vec(train_x+test_x)
# model.save(w2v_path)

# 对 输入 跟 标签 做预处理
preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
train_x = preprocess.sentence_word2idx()
y = preprocess.labels_to_tensor(y)

# 制作一个模型的对象
model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=150,
                 num_layers=1, dropout=0.5, fix_embedding=fix_embedding)

# 把 数据 分为 训练数据 跟 验证数据
X_train, X_val, y_train, y_val = train_x[:
                                         180000], train_x[180000:], y[:180000], y[180000:]

# 把 数据 做成 dataset 供 dataloader 取用
train_dataset = TwitterDataset(X=X_train, y=y_train)
val_dataset = TwitterDataset(X=X_val, y=y_val)

# 把 数据 转成 batch of tensors
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          places=paddle.CPUPlace())

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        places=paddle.CPUPlace())

# 开始训练
training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model)


# Predict and Write to csv file
# 开始测试模型并做预测
print("loading testing data ...")
test_x = load_testing_data(testing_data)
preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
test_x = preprocess.sentence_word2idx()
# print("test_x", test_x[0])
test_dataset = TwitterDataset(X=test_x, y=None)
# print("test_dataset", test_dataset[0])
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                         places=paddle.CPUPlace())

print('\nload model ...')

param_state_dict = paddle.load("lstm_crf.pdparams")
opt_state_dict = paddle.load("lstm_crf.pdopt")
model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=150,
                 num_layers=1, dropout=0.5, fix_embedding=fix_embedding)
model.set_state_dict(param_state_dict)
model.set_state_dict(opt_state_dict)

optimizer = paddle.optimizer.Adam(
    learning_rate=lr, parameters=model.parameters())  # 将模型的参数给优化器，并给予适当的学习率
optimizer.set_state_dict(opt_state_dict)

outputs = testing(batch_size, test_loader, model)

# 写到 csv 档案供上传 Kaggle
tmp = pd.DataFrame({"id": [str(i)
                   for i in range(len(test_x))], "label": outputs})
print("save csv ...")
tmp.to_csv('predict.csv', index=False)
print("Finish Predicting")
