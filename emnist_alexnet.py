import numpy as np
import tensorflow as tf
from EmnistDataLoader import *
from tensorflow.contrib.layers.python.layers import batch_norm

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def init_bias(shape):
    return tf.Variable(tf.zeros(shape))

# Dataset Parameters
batch_size = 64
fine_size = 24

# Parameters
learning_rate = 0.001
dropout = 0.5 # Dropout, probability to keep units
training_iters = 50000
batch_size = 32
step_display = 50
step_save = 10000
path_save = './alexnet'
start_from = 'alexnet-50000'
only_test = True

def batch_norm_layer(x, train_phase, scope_bn):
    return batch_norm(x, decay=0.9, center=True, scale=True,
    updates_collections=None,
    is_training=train_phase,
    reuse=None,
    trainable=True,
    scope=scope_bn)

def alexnet(x, keep_dropout, train_phase):
    weights = {
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 96], stddev=np.sqrt(2./(11*11*3)))),
        'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256], stddev=np.sqrt(2./(5*5*96)))),
        'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384], stddev=np.sqrt(2./(3*3*256)))),
        'wc4': tf.Variable(tf.random_normal([3, 3, 384, 256], stddev=np.sqrt(2./(3*3*384)))),
        'wc5': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2./(3*3*256)))),

        'wf6': tf.Variable(tf.random_normal([7*7*256, 4096], stddev=np.sqrt(2./(7*7*256)))),
        'wf7': tf.Variable(tf.random_normal([4096, 4096], stddev=np.sqrt(2./4096))),
        'wo': tf.Variable(tf.random_normal([4096, 47], stddev=np.sqrt(2./4096)))
    }

    biases = {
        'bo': tf.Variable(tf.ones(47))
    }

    # Conv + ReLU + Pool, 224->55->27
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    conv1 = batch_norm_layer(conv1, train_phase, 'bn1')
    conv1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU  + Pool, 27-> 13
    conv2 = tf.nn.conv2d(pool1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = batch_norm_layer(conv2, train_phase, 'bn2')
    conv2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU, 13-> 13
    conv3 = tf.nn.conv2d(pool2, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = batch_norm_layer(conv3, train_phase, 'bn3')
    conv3 = tf.nn.relu(conv3)

    # Conv + ReLU, 13-> 13
    conv4 = tf.nn.conv2d(conv3, weights['wc4'], strides=[1, 1, 1, 1], padding='SAME')
    conv4 = batch_norm_layer(conv4, train_phase, 'bn4')
    conv4 = tf.nn.relu(conv4)

    # Conv + ReLU + Pool, 13->6
    conv5 = tf.nn.conv2d(conv4, weights['wc5'], strides=[1, 1, 1, 1], padding='SAME')
    conv5 = batch_norm_layer(conv5, train_phase, 'bn5')
    conv5 = tf.nn.relu(conv5)
    #pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # FC + ReLU + Dropout
    fc6 = tf.reshape(conv5, [-1, weights['wf6'].get_shape().as_list()[0]])
    fc6 = tf.matmul(fc6, weights['wf6'])
    fc6 = batch_norm_layer(fc6, train_phase, 'bn6')
    fc6 = tf.nn.relu(fc6)
    fc6 = tf.nn.dropout(fc6, keep_dropout)

    # FC + ReLU + Dropout
    fc7 = tf.matmul(fc6, weights['wf7'])
    fc7 = batch_norm_layer(fc7, train_phase, 'bn7')
    fc7 = tf.nn.relu(fc7)
    fc7 = tf.nn.dropout(fc7, keep_dropout)

    # Output FC
    out = tf.add(tf.matmul(fc7, weights['wo']), biases['bo'])

    return out


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
    'randomize': True
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
train_phase = tf.placeholder(tf.bool)

# Construct model
logits = alexnet(x, keep_dropout, train_phase)

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
    if len(start_from)>1:
        saver.restore(sess, start_from)
    else:
        sess.run(init)

    if only_test == False:

        step = 1
        while step < training_iters:
            # Load a batch of data
            images_batch, labels_batch = loader.next_batch(batch_size)

            # Run optimization op (backprop)
            sess.run(train_optimizer, feed_dict={x: images_batch, y: labels_batch, keep_dropout: dropout, train_phase: True})

            if step % step_display == 0:
                # Calculate batch loss and accuracy while training
                l, acc = sess.run([loss, accuracy], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False})
                print('Iter ' + str(step) + ', Minibatch Loss = ' + \
                      '{:.6f}'.format(l) + ", Training Accuracy = " + \
                      '{:.4f}'.format(acc))

            step += 1

            # Save model
            if step % step_save == 0:
                saver.save(sess, path_save, global_step=step)
                print('Model saved at Iter %d !' %(step))

        print('Optimization Finished!')

    # Evaluate on the whole validation set
    print('Evaluation on the whole validation set...')
    num_batch = loader.test_size() // batch_size
    accuracy_val_total = 0.
    for i in range(num_batch):
        images_test, labels_test = loader.load_test_batch(batch_size)
        accuracy_val = sess.run(accuracy, feed_dict={x: images_test, y: labels_test, keep_dropout: 1., train_phase: False})
        accuracy_val_total += accuracy_val
        print("Validation Accuracy = " + \
              "{:.4f}".format(accuracy_val))

    accuracy_val_total /= num_batch
    print('Evaluation Finished! Testing Accuracy:', accuracy_val_total)
