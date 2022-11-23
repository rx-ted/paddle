'''
Author: rx-ted
Date: 2022-11-23 21:17:58
LastEditors: rx-ted
LastEditTime: 2022-11-23 22:44:35
'''
import codecs
import jieba
from libsvm import svm
from libsvm.svmutil import svm_train, svm_predict, svm_save_model, svm_load_model
from libsvm.commonutil import svm_read_problem


train_path = "cnews.train.txt"
label_vocab_path = "cnews.category.txt"
test_path = 'cnews.test.txt'


def process_line(idx: int, line: str):
    data = tuple(line.strip('\r\t').split('\t'))
    if len(data) != 2:
        return None
    content_segged = jieba.cut(data[1])
    return data[0], content_segged


def load_data(filename):
    with codecs.open(train_path, encoding='utf-8')as fp:
        lines = fp.readlines()
    data_records = [process_line(idx, line)
                    for idx, line in enumerate(lines)]
    data_records = [data for data in data_records if data is not None]
    return data_records


def build_label_vocab(path: str):
    label = {}
    with codecs.open(path, encoding='utf-8')as fp:
        lines = fp.readlines()
    for line in lines:
        l = line.strip('\r\t').split('\t')
        label[l[0]] = int(l[1])
    # print(label)
    return label


def build_vocab(train_data, thresh):
    word_count = {}
    for i, j in enumerate(train_data):
        for word in j:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1

    word_list = [(k, v) for k, v in word_count.items()]
    word_list.sort(key=lambda x: x[1], reverse=True)
    word_list_filtered = [word for word in word_list if word[1] > thresh]
    vocab = {}
    for word in word_list_filtered:
        vocab[word[0]] = len(vocab)
    return vocab


def construct_trainable_matrix(corpus, vocab, label_vocab, output_path):
    records = []
    for idx, data in enumerate(corpus):
        label = str(label_vocab[data[0]])
        token_dict = {}
        for token in data[1]:
            token_id = vocab.get(token, 0)
            if token_id in token_dict:
                token_dict[token_id] += 1
            else:
                token_dict[token_id] = 1
        feature = [str(int(k)+1)+':'+str(v) for k, v in token_dict.items()]
        feature_text = ' '.join(feature)
        records.append(label + ' ' + feature_text)
    with open(output_path, 'w') as f:
        f.write('\n'.join(records))


# train_data = load_data(train_path)
# test_data = load_data(test_path)
# label_vocab = build_label_vocab(label_vocab_path)
# vocab = build_vocab(train_data, 1)
# construct_trainable_matrix(train_data, vocab, label_vocab, "train_svm.txt")
# construct_trainable_matrix(train_data, vocab, label_vocab, "test_svm.txt")

# train svm
train_label, train_feature = svm_read_problem('train_svm.txt')
print(train_label[0], train_feature[0])
model=svm_train(train_label,train_feature,'-s 0 -c 5 -t 0 -g 0.5 -e 0.1')

# predict
test_label, test_feature = svm_read_problem('test_svm.txt')
print(test_label[0], test_feature[0])
p_labs, p_acc, p_vals = svm_predict(test_label, test_feature, model)

print('accuracy: {}'.format(p_acc))
