from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.utils.vis_utils import plot_model#绘制模型
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import keras
batch_size = 32
nb_classes = 10
nb_epoch = 20
img_channels = 3
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
#可将数据归一化
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

base_model = InceptionV3(weights='imagenet', include_top=False)
plot_model(base_model, to_file='base_model_inceptionV3.png')
base_model.summary()
x = base_model.output
#全局平均池化
x = GlobalAveragePooling2D()(x)
# 比原来的inceptionV3多了一个全连接层
x = Dense(1024, activation='relu')(x)
predictions = Dense(nb_classes, activation='softmax')(x)
model = Model(input=base_model.input, output=predictions)
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)
plot_model(model, to_file='change_model_inceptionV3.png')
# 图片生成器，数据扩增
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images
# Compute quantities required for feature-wise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(X_train)
# 首先训练顶层分类器
for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit_generator(datagen.flow(X_train, Y_train,
                                 batch_size=batch_size),
                    steps_per_epoch=X_train.shape[0]//batch_size+1,
                    epochs=nb_epoch,
                    validation_data=(X_test,Y_test),
                    validation_steps=X_test[0]//batch_size+1
                    )
# 然后再基于它进行fine-tune。
# Fine-tune以一个预训练好的网络为基础，在新的数据集上重新训练一小部分权重。fine-tune应该在很低的学习率下进行，通常使用SGD优化
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)
for layer in model.layers[:248]:
   layer.trainable = False
for layer in model.layers[248:]:
   layer.trainable = True
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
model.fit_generator(datagen.flow(X_train, Y_train,
                                 batch_size=batch_size),
                    steps_per_epoch=X_train.shape[0]//batch_size+1,
                    epochs=nb_epoch,
                    validation_data=(X_test, Y_test),
                    validation_steps=X_test[0]//batch_size+1
                    )

