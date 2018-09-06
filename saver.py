import tensorflow as tf
import numpy as np


# # save to file
# remember to define the same dtype and shape when restore
# W = tf.Variable([[1,2,3],[3,4,5]],dtype= tf.float32,name = 'weights')
# b = tf.Variable([[1,2,3]],dtype=tf.float32,name='biases' )
#
# init = tf.initialize_all_variables()
#
# 定义一个保存器
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)
#     #将 session 中的所有东西保存下来
#     save_path = saver.save(sess,'my_net/save_net.ckpt')
#     print("Save to path:",save_path)

'''
导入变量
新变量的shape和type必须要一样
'''
# np.arange(6) 输出[0,1,2,3,4,5] 从0开始的6个数
W = tf.Variable(np.arange(6).reshape(2,3),dtype=tf.float32,name='weights')
b = tf.Variable(np.arange(3).reshape(1,3),dtype=tf.float32,name='biases')

#not need init
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,'my_net/save_net.ckpt')
    print("weights:",sess.run(W))
    print("biases:",sess.run(b))