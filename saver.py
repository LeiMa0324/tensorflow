import tensorflow as tf


## save to file

W = tf.Variable([[1,2,3],[3,4,5]],dtype= tf.float32,name = 'weights')
b = tf.Variable([[1,2,3]],dtype=tf.float32,name='biases' )

init = tf.initialize_all_variables()

#定义一个保存器
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    #将 session 中的所有东西保存下来
    save_path = saver.save(sess,'my_net/save_net.ckpt')
    print("Save to path:",save_path)