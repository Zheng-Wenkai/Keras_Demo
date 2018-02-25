from keras.datasets import cifar10
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
batch_size = 32
nb_classes = 10
nb_epoch = 20
kernel_size = (3, 3)
(x_train, y_train), (x_val, y_val) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, nb_classes)
y_val = np_utils.to_categorical(y_val, nb_classes)

model = Sequential()
model.add(Convolution2D(32, kernel_size, padding='same',
                        input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(32, kernel_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Convolution2D(64,kernel_size, padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, kernel_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.summary()
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    rescale=1./255)

# 计算依赖于数据的变换所需要的统计信息(均值方差等),
# 只有使用featurewise_center，featurewise_std_normalization或zca_whitening时需要此函数。
datagen.fit(x_train)

'''fit_generator接收numpy数组和标签为参数时使用flow'''
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0]//batch_size+1,
                    validation_data=(x_val,y_val),
                    epochs=nb_epoch)

'''fit_generator以文件夹路径为参数时使用flow_from_directory'''
# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)
# train_generator = train_datagen.flow_from_directory(
#         'data/train',
#         target_size=(150, 150),
#         batch_size=32,
#         class_mode='binary')
# val_datagen = ImageDataGenerator(rescale=1./255)
# validation_generator = val_datagen.flow_from_directory(
#         'data/validation',
#         target_size=(150, 150),
#         batch_size=32,
#         class_mode='binary')
# model.fit_generator(
#         train_generator,
#         steps_per_epoch=x_train.shape[0]//batch_size+1,
#         epochs=nb_epoch,
#         validation_data=validation_generator,)

'''evalute_geneator，主要用于评价模型之前对图片做一些处理'''
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = datagen.flow(
            x_val, y_val,
            batch_size=batch_size,
)
score=model.evaluate_generator(val_generator,steps=x_val.shape[0] // batch_size+1)