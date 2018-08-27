import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
'''
with tf.name_scope("xx")给某个框架或者变量命名

'''
def add_layers(inputs,in_size,out_size,activation_function=None):
    #层框架命名
    with tf.name_scope("layer"):
        with tf.name_scope("weights"):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name="W")
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.matmul(inputs,Weights)+biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs



#linspace matlab中的指令，生成x1和x2之间指定n个点（等差数列）
x_data = np.linspace(-1,1,300)[:,np.newaxis]
#输出x_data.shape个符合(0,0.05)正态分布的高斯随机数
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) -0.5 +noise

#输入框架
with tf.name_scope("inputs"):
    xs = tf.placeholder(tf.float32,[None,1],name="x_input")
    ys = tf.placeholder(tf.float32,[None,1],name="y_input")



'''
定义隐藏层，一个输入，10个输出，使用relu作为激励函数
'''
l1 = add_layers(xs,1,10,activation_function=tf.nn.relu)

'''
定义输出层，十个输入，1个输出，无激励函数
'''
prediction = add_layers(l1,10,1,activation_function=None)

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),1))

with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


'''
定义session
'''
init = tf.initialize_all_variables()
sess = tf.Session()

#定义完session后，定义writer和文件
writer = tf.summary.FileWriter("logs/",sess.graph)
sess.run(init)


#学习1000步 trainstep
for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i %50:

        # print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        prediction_value = sess.run(prediction,feed_dict={xs:x_data,ys:y_data})

