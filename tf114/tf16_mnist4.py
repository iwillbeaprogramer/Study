import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(10000, 28*28).astype('float32')/255

# 2. 모델 구성
x = tf.placeholder('float', [None,784])
y = tf.placeholder('float', [None,10])

# w1 = tf.Variable(tf.random_normal([784,100]), name='weight1')
w1 = tf.get_variable('w1',shape=[784,100],initializer = tf.contrib.layers.xavier_initializer())
print(w1)
# <tf.Variable 'w1:0' shape=(784, 100) dtype=float32_ref>
b1 = tf.Variable(tf.random_normal([100]), name='bias1')
print(b1) 
# <tf.Variable 'bias1:0' shape=(100,) dtype=float32_ref>
# 
# layer1 = tf.nn.softmax(tf.matmul(x,w) + b)
# layer1 = tf.nn.relu(tf.matmul(x,w) + b)
# layer1 = tf.nn.selu(tf.matmul(x,w) + b)
layer1 = tf.nn.elu(tf.matmul(x,w1) + b1)
print(layer1)
# Tensor("Elu:0", shape=(?, 100), dtype=float32)
layer1 = tf.nn.dropout(layer1, keep_prob=0.3) # dropout(0.3)
print(layer1)
# Tensor("dropout/mul_1:0", shape=(?, 100), dtype=float32)


w2 = tf.get_variable('weight2',shape = [100,50])
b2 = tf.Variable(tf.random_normal([50]), name='bias2')
layer2 = tf.nn.selu(tf.matmul(layer1,w2) + b2)
layer2 = tf.nn.dropout(layer2, keep_prob=0.3) # dropout(0.3)

w3 = tf.Variable(tf.random_normal([50,10]), name='weight3')
b3 = tf.Variable(tf.random_normal([10]), name='bias3')
hypothesis = tf.nn.softmax(tf.matmul(layer2,w3) + b3)

# 3. 컴파일, 훈련
loss = tf.reduce_mean(- tf.reduce_sum(y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)


training_epochs = 15
batch_size=100
total_batch = int(len(x_train)/batch_size)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost = 0

    for i in range(total_batch): #600번 돈다
        start = i*batch_size
        end = start+batch_size

        batch_x, batch_y = x_train[start:end],y_train[start:end]

        feed_dict = {x:batch_x , y:batch_y}
        c,_ = sess.run([loss,optimizer],feed_dict = feed_dict)
        avg_cost+= c/total_batch

    print('Epoch : ',"%04d"%(epoch+1),"\ncost = {:.9f}".format(avg_cost))
print("훈련끝")

prediction = tf.equal(tf.arg_max(hypothesis,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(prediction,tf.float32))
print(sess.run(accuracy,feed_dict={x:x_test,y:y_test}))





# with tf.Session() as sess:
#     sess.run(tf.compat.v1.global_variables_initializer())

#     for step in range(2001):
#         sess.run(optimizer, feed_dict={x:x_train, y:y_train})

#         if step % 100 == 0: 
#             print(step, '\tloss :', sess.run(loss, feed_dict={x:x_train, y:y_train}))
