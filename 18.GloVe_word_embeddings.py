import os
import sys
import numpy as np
from keras.utils import to_categorical
from keras.layers import Embedding
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Flatten,MaxPooling1D,Conv1D
from keras.models import Model
texts = []  # list of text samples
labels_index = {}  # 由标签名称映射到ID的字典
labels = []  # list of label ids
TEXT_DATA_DIR='/news20/20_newsgroup'
GLOVE_DIR='/glove.6B'
MAX_NB_WORDS=20000
MAX_SEQUENCE_LENGTH=1000
VALIDATION_SPLIT=0.2
EMBEDDING_DIM=100

# 遍历下语料文件下的所有文件夹，获得不同类别的新闻以及对应的类别标签
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    # 文件夹名称即为类别标签
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)# 随着循环，id由0递增
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                if sys.version_info < (3,):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='latin-1')
                # 打开新闻样本
                t = f.read()
                i = t.find('\n\n')
                if 0 < i:
                    t = t[i:]
                texts.append(t)
                # 保存新闻样本的列表
                f.close()
                labels.append(label_id)
                # 保存新闻样本对应的类别ID的标签
print('Found %s texts.' % len(texts))

# 将新闻样本转换为张量张量
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
# 只保留新闻文本最常见的20000个词汇
tokenizer.fit_on_texts(texts)
# 根据文本列表更新内部词汇表
sequences = tokenizer.texts_to_sequences(texts)
# 将文本转为序列
word_index = tokenizer.word_index
# 字典，将词汇表的词汇映射为下标
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
#  每个新闻文本最多保留1000个词，用这1000个词来判断新闻文本的类别
labels = to_categorical(np.asarray(labels))
# 将类别向量(从0到nb_classes的整数向量)映射为二值类别矩阵
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# 将新闻样本分割成一个训练集和一个验证集。
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
# 打乱新闻样本
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

# 从GloVe文件中解析出每个词和它所对应的词向量，并用字典的方式存储
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'),'r',encoding='UTF-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
    # 每一行的第一个元素作为键，后面的元素全部作为值
f.close()
print('Found %s word vectors.' % len(embeddings_index))
print('Found %s unique tokens.' % len(word_index))

# 我们可以根据GloVe文件解析得到的字典生成上文所定义的词向量矩阵，第i列表示词索引为i的词的词向量。
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
# 词向量矩阵初始化为0，词汇数*词汇维度
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# 词向量矩阵加载到Embedding层
embedding_layer = Embedding(len(word_index) + 1, # 字典长度，即输入数据最大下标+1
                            EMBEDDING_DIM, #全连接嵌入的维度
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

# 训练一个一维的卷积网络
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=20, batch_size=128)
# 可以通过一些正则化机制（如退出）或通过微调Embedding层来更长时间地训练更高的精度。

#     若不使用预先训练的词语: Embedding从头开始初始化我们的层，并在训练期间学习它的权重。
#     这只需要替换Embedding图层（实际上，预训练的词向量引入了外部语义信息，往往对模型很有帮助）：
#embedding_layer = Embedding(len(word_index) + 1,
                            #EMBEDDING_DIM,
                            #input_length=MAX_SEQUENCE_LENGTH)