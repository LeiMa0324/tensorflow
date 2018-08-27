import tensorflow as tf
import numpy as np
'''
variables：用于一些可训练变量，Weights&biases，其值在训练中会改变，必须提供初始值
placeholder：用于传递进来的真实的训练样本，不必指定初始值，
必须用feed_dict{}的方式给sess.run喂参数
'''


'''
定义层的函数
'''
def add_layers(inputs,in_size,out_size,activation_function=None):
    #使用random_normal给weights赋初始值
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    #使用0.1给偏置赋初始值
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

'''
定义数据
'''
#linspace matlab中的指令，生成x1和x2之间指定n个点（等差数列）
x_data = np.linspace(-1,1,300)[:,np.newaxis]
#输出x_data.shape个符合(0,0.05)正态分布的高斯随机数
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) -0.5 +noise

#定义placeholder
xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

'''
定义隐藏层，一个输入，10个输出，使用relu作为激励函数
'''
l1 = add_layers(xs,1,10,activation_function=tf.nn.relu)

'''
定义输出层，十个输入，1个输出，无激励函数
'''
prediction = add_layers(l1,10,1,activation_function=None)


'''
开始预测
'''
#定义损失函数
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),1))
#定义训练步骤,以0.1的速率，梯度下降求解loss的最小值
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()

'''
定义session
'''
sess = tf.Session()
sess.run(init)

#学习1000步 trainstep
for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})