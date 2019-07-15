# learn-tensorflow

## define const

```python
const = tf.constant(2.0, name='const')
```

## define variable

```python
c = tf.Variable(1.0, dtype=tf.float32, name='c')
```

## define operation

```python
e = tf.add(c, const, name='e')
```

## init & creat session

```python
# 1. 定义init operation
init_op = tf.global_variables_initializer()
# session
with tf.Session() as sess:
	# 2. 运行init operation
	sess.run(init_op)
	# 计算
	a_out = sess.run(a)
	print("Variable a is {}".format(a_out))
```

## placeholder

```python
# 创建placeholder
b = tf.placeholder(tf.float32, [None, 1], name='b')
a_out = sess.run(a, feed_dict={b: np.arange(0, 10)[:, np.newaxis]})
```

## neural network

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 超参数
learning_rate = 0.5
epochs = 10
batch_size = 100

# placeholder
# 输入图片为28 x 28 像素 = 784
x = tf.placeholder(tf.float32, [None, 784])
# 输出为0-9的one-hot编码
y = tf.placeholder(tf.float32, [None, 10])

# hidden layer => w, b
W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([300]), name='b1')
# output layer => w, b
W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([10]), name='b2')

# hidden layer
hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.relu(hidden_out)

# 计算输出
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))

# 创建优化器，确定优化目标
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimizer(cross_entropy)

# init operator
init_op = tf.global_variables_initializer()

# 创建准确率节点
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 创建session
with tf.Session() as sess:
	# 变量初始化
	sess.run(init_op)
	total_batch = int(len(mnist.train.labels) / batch_size)
	for epoch in range(epochs):
		avg_cost = 0
		for i in range(total_batch):
			batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
			_, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
			avg_cost += c / total_batch
		print("Epoch:", (epoch + 1), "cost = ", "{:.3f}".format(avg_cost))
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
```













