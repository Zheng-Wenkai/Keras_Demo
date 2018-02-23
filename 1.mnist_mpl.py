#coding=utf-8
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
batch_size = 128
nb_classes = 10
nb_epoch = 20
# 准备数据
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#将3D转化为2D
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# 对数据进行归一化到0-1 因为图像数据最大是255
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
# 将类别向量(从0到nb_classes的整数向量)映射为二值类别矩阵
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print(Y_train.shape)
print(Y_test.shape)
# 建立模型
model = Sequential()
model.add(Dense(512, input_shape=(784,)))#Dense的第一个参数是输出维度
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(10))
model.add(Activation('softmax'))
print(model.summary())
plot_model(model, to_file='model.png')
# 训练和评估
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
history = model.fit(X_train,Y_train,batch_size=batch_size,epochs=nb_epoch,verbose=1,validation_data=(X_test,Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
#模型保存
model.save('mnist-mpl.h5')
