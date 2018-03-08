import tensorflow as tf

m1 = tf.constant([[2, 2]])
m2 = tf.constant([[3],[3]])

dot_opertion = tf.matmul(m1, m2)

print(dot_opertion)

#method 1 use session
sess = tf.session()
result = sess.run(dot_opertion)
print(result)
sess.close()

#method2 use session
with tf.Session() as sess:
    result_ = sess.run(dot_opertion)
    print(result_)
