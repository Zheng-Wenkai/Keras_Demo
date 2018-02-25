from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input,decode_predictions
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

'''加载并显示图片'''
img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
plt.imshow(img)
plt.show()

'''图片预处理'''
#将图片转成numpy array
x = image.img_to_array(img)
#扩展维度,因为preprocess_input需要4D的格式
x = np.expand_dims(x, axis=0)
#对张量进行预处理
x = preprocess_input(x)

#加载VGG16模型（包含全连接层）
model = VGG16(include_top=True, weights='imagenet')
preds = model.predict(x)
# 将结果解码成一个元组列表（类、描述、概率）（批次中每个样本的一个这样的列表）
print('Predicted:', decode_predictions(preds, top=3)[0])
