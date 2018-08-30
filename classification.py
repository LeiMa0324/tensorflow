import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
'''
使用一层神经网络，十个神经元
输入为784维的x
'''

'''
深度学习神经元的基本结构Wx+b
'''
def add_layer(inputs,in_size,out_size,activation_function=None,):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs,Weights)+biases

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs

'''
输出一轮计算后的正确率
'''
def compute_accuracy(v_xs,v_ys):
    global prediction
    #输出层的输出，记录为y_pre
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    #得到y_pre数组中，概率值最大的下标，和真实值ys为1的下标进行比较，下标相同则判断正确，返回True
    #返回的维度和比较的数据维度一致，一个true和false的向量或者矩阵
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    # tf.cast 将bool型转化为float型，计算一次比较后所有的实例有多少是正确的多少是错误的
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    #计算得到正确率百分比
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

#define placebolder
xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])

# add output layer，一层，使用softmax作为激励函数
prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax)

# loss
#使用交叉熵作为损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


sess = tf.Session()
sess.run(tf.initialize_all_variables())

'''
学习1000次
'''
for i in range(1000):
    #使用batch的方法学习
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i %50 :
        print(compute_accuracy(mnist.test.images,mnist.test.labels))
