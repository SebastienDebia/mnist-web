import tensorflow as tf
from tensorflow.python.framework import tensor_shape

import math
import os

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# constants
BATCH_SIZE = 200
STEPS = 20000
MODEL_NAME = "mnist_cnn_model"
CKPT_NAME = MODEL_NAME+"/"+MODEL_NAME+".ckpt"

# creation of the directory to save the model
if not os.path.exists(MODEL_NAME):
    os.makedirs(MODEL_NAME)


# placeholders
# input
x = tf.placeholder(tf.float32, [None, 784])
print(x.get_shape())

# reshape to apply convolutions etc
x_reshaped = tf.reshape(x, [-1,28,28,1])
print(x_reshaped.get_shape())

# labels
y_ = tf.placeholder(tf.float32, [None, 10])

# placeholder_with_default to choose wether distortions should be done or not
# either provide img_in with distorded images when training, or directly x for inference
img_in = tf.placeholder_with_default(x_reshaped, [None, 28, 28, 1])


def create_distord_ops(net, rotation_angle_rad = math.pi/6, shift = 7, outcrop = 0.20, incrop = 0.20, batch_size = 100):
    """
    creates the distortion ops

    ops to rotate, shift, crop and resize the image randomly
    gives much much (much!) better results when the numbers are off center, too small etc

    arguments:
    net -- input tensor of shape [-1,28,28,1]
    rotation_angle_rad -- max angle to rotate left or right (i.e. pi/4 means it can rotate from -pi/4 to +pi/4)
    shift -- max number of pixels the image can be shifted in any direction
    outcrop -- max percentage the crop can go outside the image (thus reducing the actual content's size)
    incrop -- max percentage the crop can go inside the image (thus augmenting the actual content's size, and potentially cropping it)
    """
    # distord ops
    net_dist = net
    num_images_ = tf.shape(net_dist)[0]
    # random rotate
    rotatation_radian_val = tf.random_uniform([num_images_], minval=-rotation_angle_rad, maxval=rotation_angle_rad)
    net_dist = tf.contrib.image.rotate(net_dist, rotatation_radian_val)
    # random shift
    base_row = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], shape=[1, 8], dtype=tf.float32)
    base_ = tf.tile(base_row, [num_images_, 1])
    mask_row = tf.constant([0, 0, 1, 0, 0, 1, 0, 0], shape=[1, 8], dtype=tf.float32)
    mask_ = tf.tile(mask_row, [num_images_, 1])
    random_shift_ = tf.random_uniform([num_images_, 8], minval=-shift, maxval=shift, dtype=tf.float32)
    transforms_ = base_ + random_shift_ * mask_
    net_dist = tf.contrib.image.transform(images=net_dist, transforms=transforms_)
    # crop and resize
    rnd_1 = tf.random_uniform([num_images_, 2], minval=-outcrop, maxval=incrop, dtype=tf.float32) # y1, x1
    rnd_2 = tf.random_uniform([num_images_, 2], minval=1-incrop, maxval=1+outcrop, dtype=tf.float32) # y2, x2
    boxes_ = tf.concat([rnd_1, rnd_2], 1) # y1, x1, y2, x2
    net_dist = tf.image.crop_and_resize( net_dist,
        boxes_,
        box_ind=[i for i in xrange(batch_size)], # taille du batch en dur!
        crop_size=[28,28]
    )

    return net_dist

def create_model(net):
    # model
    net = tf.layers.conv2d(net, 8, 5, padding='same', activation=tf.nn.relu6)
    print(net.get_shape())
    net = tf.layers.conv2d(net, 16, 5, (2,2), padding='same', activation=tf.nn.relu6)
    print(net.get_shape())
    back = net
    net = tf.layers.separable_conv2d(net, 16, 3, padding='same', activation=tf.nn.relu6)
    print(net.get_shape())
    net = tf.layers.conv2d(net, 16, 1, padding='same', activation=tf.nn.relu6)
    print(net.get_shape())
    net = tf.layers.separable_conv2d(net, 16, 3, padding='same', activation=tf.nn.relu6)
    print(net.get_shape())
    net = tf.layers.conv2d(net, 16, 1, padding='same', activation=tf.nn.relu6)
    print(net.get_shape())
    net = tf.layers.separable_conv2d(net, 16, 3, padding='same', activation=tf.nn.relu6)
    print(net.get_shape())
    net = tf.layers.conv2d(net, 16, 1, padding='same', activation=tf.nn.relu6)
    print(net.get_shape())
    net = net + back
    net = tf.layers.conv2d(net, 64, 1, (2,2), activation=tf.nn.relu6)
    print(net.get_shape())
    back = net
    net = tf.layers.separable_conv2d(net, 64, 3, padding='same', activation=tf.nn.relu6)
    print(net.get_shape())
    net = tf.layers.conv2d(net, 64, 1, padding='same', activation=tf.nn.relu6)
    print(net.get_shape())
    net = tf.layers.separable_conv2d(net, 64, 3, padding='same', activation=tf.nn.relu6)
    print(net.get_shape())
    net = tf.layers.conv2d(net, 64, 1, padding='same', activation=tf.nn.relu6)
    print(net.get_shape())
    net = tf.layers.separable_conv2d(net, 64, 3, padding='same', activation=tf.nn.relu6)
    print(net.get_shape())
    net = tf.layers.conv2d(net, 64, 1, padding='same', activation=tf.nn.relu6)
    print(net.get_shape())
    net = net + back
    net = tf.layers.conv2d(net, 128, 1, activation=tf.nn.relu6)
    print(net.get_shape())
    net = tf.layers.separable_conv2d(net, 128, 5, padding='same', activation=tf.nn.relu6)
    print(net.get_shape())
    net = tf.layers.conv2d(net, 256, 1, activation=tf.nn.relu6)
    print(net.get_shape())
    net = tf.layers.separable_conv2d(net, 256, 7, (7,7), padding='same', activation=tf.nn.relu6)
    print(net.get_shape())
    net = tf.layers.flatten(net)
    print(net.get_shape())
    net = tf.layers.dense(net,84)
    print(net.get_shape())

    # final layer
    net = tf.layers.dense(net,10)
    print(net.get_shape())

    return net, tf.nn.softmax(net)

def create_training_ops(y, y_, with_entropy = True, entropy_scaling = 0.02):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    cr_batch = tf.reduce_mean(cross_entropy)

    #entropy = tf.zeros([tf.shape(cross_entropy)[0]])
    y_soft = tf.nn.softmax(y)
    entropy = -(10 * tf.reduce_sum(y_soft * tf.log(y_soft+0.01), 1) ) * entropy_scaling
    ent_batch = tf.reduce_mean(entropy)

    loss = cross_entropy
    if with_entropy:
        loss = loss - entropy
    loss_batch = tf.reduce_mean(loss)

    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return train_step, accuracy, loss_batch, cr_batch, ent_batch


# creation of the model
y, y_soft = create_model( img_in )

# init
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# try to load the parameters, or train the model
last_model_file = tf.train.latest_checkpoint(MODEL_NAME)
if last_model_file != None:
    print("loading...")

    saver.restore( sess, last_model_file )
else:
    print("training...")

    # preparing ops for training and distortion of training images
    net_dist = create_distord_ops( x_reshaped, batch_size = BATCH_SIZE )
    train_step, accuracy, loss_batch, cr_batch, ent_batch = create_training_ops(y, y_)

    tf.global_variables_initializer().run()

    # training loop
    for i in range(STEPS):
        # loading a batch
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        # applying distortions to the images in the batch
        batch_xs_dist = sess.run(net_dist, feed_dict={x: batch_xs})
        # running the train step
        sess.run(train_step, feed_dict={img_in: batch_xs_dist, y_: batch_ys})
        # printing some stats
        if i % 100 == 0:
            acc, lossP, crossP, entP = sess.run([accuracy, loss_batch, cr_batch, ent_batch], feed_dict={x: mnist.test.images[:2000], y_: mnist.test.labels[:2000]})
            print("[{:<7}] acc: {:0<6.2%} loss: {:0<5.3} cross: {:0<5.3} entropy: {:0<5.3}".format(i, acc, lossP, crossP, entP))
        # saving every so often
        if i % 1000 == 0:
            save_path = saver.save(sess, CKPT_NAME )
    save_path = saver.save(sess, CKPT_NAME)

def predict(img):
    return sess.run(y_soft, feed_dict={x: [img]})
