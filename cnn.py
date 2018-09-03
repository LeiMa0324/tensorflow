import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot= True)

'''
计算准确率
'''
def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
    return result

'''
输入shape，得到weight和bias的初始值
'''
def weight_variable(shape):
    #产生随机变量
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # bias初始值为0.1
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

'''
定义卷积网络神经层
输入图片信息和Weights
'''
def conv2d(x,W):
    # stride = [1,x_movement,y_movement,1]
    # PADDING ='VALID' OR 'SAME'
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding= 'SAME')

'''
定义pooling层
'''
def max_pool_2x2(x):
    # stride = [1,x_movement,y_movement,1]
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#define placeholder
xs = tf.placeholder(tf.float32,[None,784])#28*28
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)

# -1为不管有多少个样本保持不变，样本内为28*28,1为channel
# 将xs的数据reshape为conv2d接受的input=[batch,in_heigh,in_weight,in_channels]的形式
x_image = tf.reshape(xs,[-1,28,28,1])

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

'''
error
'''
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=1))
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()

'''
important step
重要步骤
'''
sess.run(tf.initialize_all_variables)

'''
start training
开始训练
'''
for i in range(1000):
     batch_xs,batch_ys=mnist.train.next_batch(100)
     sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:1})
     if i%50 == 0:
         print(compute_accuracy(mnist.test.images,mnist.test.labels))