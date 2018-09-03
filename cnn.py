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
#shape = [kernel height,kernel width, kernel channels,kernel number]
def weight_variable(shape):
    #产生随机变量
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

#shape =[kernel number]
def bias_variable(shape):
    # bias初始值为0.1
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

'''
定义卷积网络神经层
输入图片信息和Weights，步长为1
'''
def conv2d(x,W):
    # stride = [1,x_movement,y_movement,1]
    # PADDING ='VALID' OR 'SAME'
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding= 'SAME')

'''
定义pooling层，步长为2，SAME PADDING方式
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
patch(kernel/filter) 长款为5*5,图片原本的厚度insize=1,outsize=32，卷积过后图片的厚度(kernel的个数)
定义32个，厚度为1（与原图片厚度相同），大小为5*5的kernel(filter)，其中的weight由函数随机生成
'''

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
#给conv2d传入数据和权重，卷积过后每个kernel的结果+bias
#再对结果进行relu处理
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
#对结果进行pooling处理
h_pool1 = max_pool_2x2(h_conv1)

'''
conv2 layer
'''
#传入高度为32，传出为64
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

'''
fully connected layer 全连接层
'''
#输入的为conv2的输出的shape
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])

#将conv2输出的数据转换为1维数据
#[n_samples,7,7,64]->[n_samples,7*7*64] 将feature value拉平
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
#防止过拟合，引入dropout
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

'''
func2 layer 输出层
输出10位，0-9的值
'''
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

'''
error
'''
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=1))
#系统过于庞大，梯度下降法很慢
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()

'''
important step
重要步骤
'''
init = tf.initialize_all_variables()
sess.run(init)

'''
start training
开始训练
'''
for i in range(1000):
     batch_xs,batch_ys=mnist.train.next_batch(100)
     sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:1})
     if i%50 == 0:
         print(compute_accuracy(mnist.test.images,mnist.test.labels))