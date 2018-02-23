# 使用sklearn wrapper做参数搜索
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.grid_search import GridSearchCV
nb_classes = 10
img_rows, img_cols = 28, 28
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)
def make_model(dense_layer_sizes, nb_filters, nb_conv, nb_pool):
    '''Creates model comprised of 2 convolutional layers followed by dense layers
        dense_layer_sizes: List of layer sizes. This list has one number for each layer
        nb_filters: Number of convolutional filters in each convolutional layer
        nb_conv: Convolutional kernel size
        nb_pool: Size of pooling area for max pooling
        '''
    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(img_rows, img_cols, 1)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    for layer_size in dense_layer_sizes:
        model.add(Dense(layer_size))#加入多个全连接层
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    return model#对模型进行编译，但还没有训练

#[32]是一个输出维度为32的Dense，[32,32]是两个输出维度为32的Dense
dense_size_candidates = [[32], [64], [32, 32], [64, 64]]
#使用sklearn的分类器接口：第一个参数是可调用的函数或类对象，第二个参数是模型参数和训练参数
my_classifier = KerasClassifier(make_model, batch_size=32)
#sklearn中的GridSearchCV函数
#说明：对估计器的指定参数值进行穷举搜索。
validator = GridSearchCV(my_classifier,
                         param_grid={'dense_layer_sizes': dense_size_candidates,
                                     # nb_epoch可用于调整，即使不是模型构建函数的参数
                                     'nb_epoch': [3, 6],
                                     'nb_filters': [8],
                                     'nb_conv': [3],
                                     'nb_pool': [2]},
                         scoring='log_loss',
                         n_jobs=1)
validator.fit(X_train, y_train)
#打印最好模型的参数
print('The parameters of the best model are: ')
print(validator.best_params_)
#返回模型
best_model = validator.best_estimator_.model
metric_names = best_model.metrics_names
metric_values = best_model.evaluate(X_test, y_test)
print('\n')
#返回名称和数值
for metric, value in zip(metric_names, metric_values):
    print(metric, ': ', value)
