from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.utils.vis_utils import plot_model
batch_size = 128
nb_classes = 10
nb_epoch = 12

# 输入数据的维度
img_rows, img_cols = 28, 28
# 使用的卷积滤波器的数量
nb_filters = 32
# 用于 max pooling 的池化面积
pool_size = (2, 2)
# 卷积核的尺寸
kernel_size = (3, 3)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#同样要reshape,只是现在图片是三维矩阵
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size,strides=(1, 1),
                        padding='valid',
                        input_shape=input_shape))#用作第一层时，需要输入input_shape参数
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))#Dense()的前面要减少连接点，防止过拟合，故通常要Dropout层或池化层
model.add(Flatten())#Dense()层的输入通常是2D张量，故应使用Flatten层或全局平均池化
model.add(Dense(128))
model.add(Activation('relu'))#Dense( )层的后面通常要加非线性化函数
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))#分类
model.summary()
plot_model(model, to_file='model-cnn.png')
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])