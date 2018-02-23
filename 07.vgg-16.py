from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

#加载并显示图片
img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
plt.imshow(img)
plt.show()

#将图片转成numpy array
x = image.img_to_array(img)
#扩展维度,因为preprocess_input需要4D的格式
x = np.expand_dims(x, axis=0)
#对张量进行预处理
x = preprocess_input(x)

#加载VGG16模型（包含全连接层）
model = VGG16(include_top=True, weights='imagenet')
#对模型进行评估，函数返回值是预测值的numpy array
scores = model.predict(x)
class_table = open('synset_words.txt', 'r')
lines = class_table.readlines()
#找出给定张量最大值的位置
print(np.argmax(scores))
print('result is ', lines[np.argmax(scores)])
class_table.close()
del model

# 加载VGG16模型（不包含包含全连接层）
model = VGG16(weights='imagenet', include_top=False)
features = model.predict(x)
print(features.shape)

#提取block5_pool层特征
model_extractfeatures = Model(input=model.input, output=model.get_layer('block5_pool').output)
block5_pool_features = model_extractfeatures.predict(x)
feature_image = block5_pool_features[:,:,:,0].reshape(7,7)
plt.imshow(feature_image)
plt.show()