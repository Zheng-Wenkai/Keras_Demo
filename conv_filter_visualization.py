#说明：通过在输入空间的梯度上升，可视化VGG16的滤波器。
from scipy.misc import imsave
import numpy as np
import matplotlib.pyplot as plt
import time
from keras.applications import vgg16
from keras import backend as K
img_width = 128
img_height = 128
layer_name = 'block5_conv1'

#将张量转换成有效图像
def deprocess_image(x):
    # 对张量进行规范化
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)
    # 转化到RGB数组
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x
def normalize(x):
    # 效用函数通过其L2范数标准化张量
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

model = vgg16.VGG16(weights='imagenet', include_top=False)
print('Model loaded.')
model.summary()

input_img = model.input
#用一个字典layer_dict存放Vgg16模型的每一层
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
print(layer_dict.keys())
print(layer_dict.values())

#提取滤波器
#通过梯度上升，修改输入图像，使得滤波器的激活最大化。这是的输入图像就是我们要可视化的滤波器。
kept_filters = []
for filter_index in range(0, 200):
    # 我们只扫描前200个滤波器，
    # 但实际上有512个
    print('Processing filter %d' % filter_index)
    start_time = time.time()
    # 我们构建一个损耗函数，使所考虑的层的第n个滤波器的激活最大化
    #由字典索引输出
    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # 我们计算输入图像的梯度与这个损失
    grads = K.gradients(loss, input_img)[0]

    # 归一化技巧：我们规范化梯度
    grads = normalize(grads)

    # 此函数返回给定输入图像的损耗和梯度
    # inputs: List of placeholder tensors.
    # outputs: List of output tensors.
    iterate = K.function([input_img], [loss, grads])

    # 梯度上升的步长
    step = 1.

    # 我们从带有一些随机噪声的灰色图像开始
    input_img_data = np.random.random((1, img_width, img_height, 3))
    input_img_data = (input_img_data - 0.5) * 20 + 128

    # 我们运行梯度上升20步
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        #每次上升一步，逐次上升
        input_img_data += grads_value * step

        print('Current loss value:', loss_value)
        if loss_value <= 0.:
            # 一些过滤器陷入0，我们可以跳过它们
            break

    # 解码所得到的输入图像
    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        #将解码后的图像和损耗值加入列表
        kept_filters.append((img, loss_value))
    end_time = time.time()
    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))


# 我们将在8 x 8网格上选择最好的64个滤波器。
n = 8
# 具有最高损失的过滤器被假定为更好看。
# 我们将只保留前64个过滤器。
#Lambda:本函数用以对上一层的输出施以任何Theano/TensorFlow表达式
kept_filters.sort(key=lambda x: x[1], reverse=True)
kept_filters = kept_filters[:n * n]

# 构建一张黑色的图片，有足够的空间
# 我们的尺寸为128 x 128的8 x 8过滤器，中间有5px的边距
margin = 5
width = n * img_width + (n - 1) * margin
height = n * img_height + (n - 1) * margin
stitched_filters = np.zeros((width, height, 3))

# 使用我们保存的过滤器填充图片
for i in range(n):
    for j in range(n):
        img, loss = kept_filters[i * n + j]
        stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                         (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

#显示滤波器
plt.imshow(img)
plt.show()
plt.imshow(stitched_filters)
plt.show()
# 保存结果,将数组保存为图像
imsave('stitched_filters_%dx%d.png' % (n, n), stitched_filters)