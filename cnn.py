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
    pass

def bias_variable(shape):
    pass

def conv2d(x,W):
    pass

def max_pool_2x2(x):
    pass

#define placeholder
xs = tf.placeholder(tf.float32,[None,784])#28*28
ys = tf.placeholder(tf.float32,[None,10])

keep_prob = tf.placeholder(tf.float32)

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
'''
sess.run(tf.initialize_all_variables)

'''
start training
'''
for i in range(1000):
     batch_xs,batch_ys=mnist.train.next_batch(100)
     sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:1})
     if i%50 == 0:
         print(compute_accuracy(mnist.test.images,mnist.test.labels))