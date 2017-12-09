import numpy as np
import tensorflow as tf
from EmnistDataLoader import *

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def init_bias(shape):
    return tf.Variable(tf.zeros(shape))

def cnn(x, keep_dropout):
    weights = {
            'wc1': init_weights([5, 5, 1, 32]),  # 5x5x1 conv, 32 outputs
            'wc2': init_weights([5, 5, 32, 64]),          # 5x5x32 conv, 64 outputs
            'wf3': init_weights([7*7*64, 1024]),         # FC 7*7*64 inputs, 1024 outputs
            'wo': init_weights([1024, 62]),         # FC 1024 inputs, 10 outputs
    }
    biases = {
            'bc1': init_bias(32),
            'bc2': init_bias(64),
            'bf3': init_bias(1024),
            'bo': init_bias(62),
    }

    # Conv + ReLU + Pool
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(tf.nn.bias_add(conv1, biases['bc1']))
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU + Pool
    conv2 = tf.nn.conv2d(pool1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(tf.nn.bias_add(conv2, biases['bc2']))
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # FC + ReLU + Dropout
    fc3 = tf.reshape(pool2, [-1, weights['wf3'].get_shape().as_list()[0]])
    fc3 = tf.add(tf.matmul(fc3, weights['wf3']), biases['bf3'])
    fc3 = tf.nn.relu(fc3)
    fc3 = tf.nn.dropout(fc3, keep_dropout)

    # Output FC
    out = tf.add(tf.matmul(fc3, weights['wo']), biases['bo'])

    return out

# Parameters
learning_rate = 0.001
training_iters = 20000
batch_size = 100
step_display = 50
step_save = 5000
path_save = './convnet'

# Network Parameters
h = 28 # MNIST data input (img shape: 28*28)
w = 28
c = 1
dropout = 0.5 # Dropout, probability to keep units

# Construct dataloader
opt_data_train = {
    'data_from' : 'emnist',
    'data_root' : 'EMNIST_data',
    'fine_size': h,
    'type'     : 'train',
    }

#opt_data_train = {
#    'data_from' : 'mnist',
#    'data_root' : 'MNIST_data',
#    'fine_size': h,
#    'type'     : 'train',
#    }

# Construct dataloader
loader = DataLoader(**opt_data_train)

# tf Graph input
x = tf.placeholder(tf.float32, [None, h, w, c])
y = tf.placeholder(tf.int64, None)
keep_dropout = tf.placeholder(tf.float32)

# Construct model
logits = cnn(x, keep_dropout)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Evaluate model
correct_pred = tf.equal(tf.argmax(logits, 1), y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# define initialization
init = tf.global_variables_initializer()

# define saver
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    # Initialization
    sess.run(init)

    step = 1
    while step < training_iters:
        # Load a batch of data
        images_batch, labels_batch = loader.next_batch(batch_size)

        # Run optimization op (backprop)
        sess.run(train_optimizer, feed_dict={x: images_batch, y: labels_batch, keep_dropout: dropout})

        if step % step_display == 0:
            # Calculate batch loss and accuracy while training
            l, acc = sess.run([loss, accuracy], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1.})
            print('Iter ' + str(step) + ', Minibatch Loss = ' + \
                  '{:.6f}'.format(l) + ", Training Accuracy = " + \
                  '{:.4f}'.format(acc))

        step += 1

        # Save model
        if step % step_save == 0:
            saver.save(sess, path_save, global_step=step)
            print('Model saved at Iter %d !' %(step))

    print('Optimization Finished!')

    # Calculate accuracy for 500 mnist test images
    images_test, labels_test = loader.load_test()
    accuracy_val = sess.run(accuracy, feed_dict={x: images_test[:500], y: labels_test[:500], keep_dropout: 1.})
    print('Testing Accuracy:', accuracy_val)
