#说明：该程序演示将一个预训练好的模型在新数据集上重新fine-tuning的过程。我们冻结卷积层，只调整全连接层。
#在MNIST数据集上使用前五个数字[0...4]训练一个卷积网络。
#在后五个数字[5...9]用卷积网络做分类，冻结卷积层并且微调全连接层
import datetime
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
now = datetime.datetime.now

batch_size = 128
nb_classes = 5
nb_epoch = 5

# 输入图像的维度
img_rows, img_cols = 28, 28
# 使用卷积滤波器的数量
nb_filters = 32
# 用于max pooling的pooling面积的大小
pool_size = 2
# 卷积核的尺度
kernel_size = (3,3)
input_shape = (img_rows, img_cols, 1)

# 数据，在训练和测试数据集上混洗和拆分
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train_lt5 = X_train[y_train < 5]
y_train_lt5 = y_train[y_train < 5]
X_test_lt5 = X_test[y_test < 5]
y_test_lt5 = y_test[y_test < 5]

X_train_gte5 = X_train[y_train >= 5]
#使标签从0~4，故-5
y_train_gte5 = y_train[y_train >= 5] - 5
X_test_gte5 = X_test[y_test >= 5]
y_test_gte5 = y_test[y_test >= 5] - 5

# 模型的训练函数
def train_model(model, train, test, nb_classes):
    #train[0]是图片，train[1]是标签
    X_train = train[0].reshape((train[0].shape[0],) + input_shape)#1D+3D=4D
    X_test = test[0].reshape((test[0].shape[0],) + input_shape)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    Y_train = np_utils.to_categorical(train[1], nb_classes)
    Y_test = np_utils.to_categorical(test[1], nb_classes)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    t = now()
    model.fit(X_train, Y_train,
              batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1,
              validation_data=(X_test, Y_test))
    print('Training time: %s' % (now() - t))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

# 建立模型，构建卷积层（特征层）和全连接层（分类层）
feature_layers = [
    Convolution2D(nb_filters, kernel_size,
                  padding='valid',
                  input_shape=input_shape),
    Activation('relu'),
    Convolution2D(nb_filters, kernel_size),
    Activation('relu'),
    MaxPooling2D(pool_size=(pool_size, pool_size)),
    Dropout(0.25),
    Flatten(),
]
classification_layers = [
    Dense(128),
    Activation('relu'),
    Dropout(0.5),
    Dense(nb_classes),
    Activation('softmax')
]
model = Sequential(feature_layers + classification_layers)

#训练预训练模型用于5个数字[0...4]的分类任务
train_model(model,
            (X_train_lt5, y_train_lt5),
            (X_test_lt5, y_test_lt5), nb_classes)

#冻结特征层(用的是同一个模型，只是冻结了部分层)
for l in feature_layers:
    l.trainable = False

#fine-tuning分类层
train_model(model,
            (X_train_gte5, y_train_gte5),
            (X_test_gte5, y_test_gte5), nb_classes)