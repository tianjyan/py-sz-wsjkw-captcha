__author__ = 'Tianjie Yang'
__email__ = 'tianjyan@qq.com'

import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
from PIL import Image
import numpy as np
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

img = Image.open(sys.stdin.buffer.raw).convert('L') # 读取图片并灰度化

img = img.crop((0, 10, 100, 36)) # 裁掉边变成 64x21

# 分离数字
img1 = img.crop((0, 0, 19, 26))
img2 = img.crop((30, 0, 49, 26))
img3 = img.crop((60, 0, 79, 26))
img4 = img.crop((80, 0, 99, 26))

img1 = np.array(img1).flatten()
img1 = list(map(lambda x: 1 if x <= 180 else 0, img1))
img2 = np.array(img2).flatten()
img2 = list(map(lambda x: 1 if x <= 180 else 0, img2))
img3 = np.array(img3).flatten()
img3 = list(map(lambda x: 1 if x <= 180 else 0, img3))
img4 = np.array(img4).flatten()
img4 = list(map(lambda x: 1 if x <= 180 else 0, img4))


x = tf.placeholder(tf.float32, [None, 19*26])

W = tf.Variable(tf.zeros([19*26, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

save_path="model/model"

saver.restore(sess, save_path)

correct_prediction = tf.argmax(y, 1)
result = sess.run(correct_prediction, feed_dict={x: [img1, img2, img3, img4]})
sess.close()

for num in result:
    print(num, end='')
