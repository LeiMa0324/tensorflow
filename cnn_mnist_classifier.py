import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

'''
计算准确率
'''
def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

'''
存入shape 就可以返回 weight 和 bias 的变量
'''
def weight_variable(shape):
    #截断的产生正态分布的函数，产生的值和均值的差距不会超过两倍的标准差
    #(shape,mean=0,stddev=0.1) shape，均值，标准差 standard deviation
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)


def conv2d(x,W):
    #二维的 CNN,x 为图片的所有信息,W为神经层的权重，stride 为步长
    #stride [1,x_movement,y_movement,1]
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

def max_pool_2x2(x):
    pass


'''
conv1 layer
'''
'''
conv2 layer
'''
'''
func1 layer
'''
'''
func2 layer
'''