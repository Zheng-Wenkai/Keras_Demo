import os
import sys
import glob
from keras.applications.inception_v3 import InceptionV3, preprocess_input

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

'''获取文件的个数'''

def get_nb_files(directory):
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


IM_WIDTH, IM_HEIGHT = 299, 299  # InceptionV3指定的图片尺寸
FC_SIZE = 1024  # 全连接层的节点个数
NB_IV3_LAYERS_TO_FREEZE = 172  # 冻结层的数量

train_dir = 'E:/data/flower_photos/train_split'  # 训练集数据
val_dir = 'E:/data/flower_photos/val_split'  # 验证集数据
nb_epoch = 3
batch_size = 16
nb_train_samples = get_nb_files(train_dir)  # 训练样本个数
nb_classes = len(glob.glob(train_dir + "/*"))  # 分类数
nb_val_samples = get_nb_files(val_dir)  # 验证集样本个数


def image_preprocess():
    #   图片生成器
    # 　训练集的图片生成器，通过参数的设置进行数据扩增
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    #   验证集的图片生成器，不进行数据扩增，只进行数据预处理
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    )
    # 训练数据与测试数据
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size, class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size, class_mode='categorical')
    return train_generator, validation_generator


# 添加顶层分类器
def add_new_last_layer(base_model, nb_classes):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    return model


# 冻结base_model所有层，这样就可以正确获得bottleneck特征
def setup_to_transfer_learn(model, base_model):
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# 冻结部分层，对顶层分类器进行Fine-tune
def setup_to_finetune(model):
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


if __name__ == "__main__":

    '''图片预处理（包括数据扩增）'''
    train_generator,validation_generator=image_preprocess()

    '''加载base_model'''
    # 使用带有预训练权重的InceptionV3模型，但不包括顶层分类器
    base_model = InceptionV3(weights='imagenet', include_top=False)  # 预先要下载no_top模型

    '''添加顶层分类器'''
    model = add_new_last_layer(base_model, nb_classes)  # 从基本no_top模型上添加新层

    '''训练顶层分类器'''
    setup_to_transfer_learn(model, base_model)
    history_tl = model.fit_generator(
        train_generator,
        epochs=nb_epoch,
        steps_per_epoch=nb_train_samples // batch_size,
        validation_data=validation_generator,
        validation_steps=nb_val_samples // batch_size,
        class_weight='auto')

    '''对顶层分类器进行Fine-tune'''
    # Fine-tune以一个预训练好的网络为基础，在新的数据集上重新训练一小部分权重。fine-tune应该在很低的学习率下进行，通常使用SGD优化
    setup_to_finetune(model)
    history_ft = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=nb_val_samples // batch_size,
        class_weight='auto')
