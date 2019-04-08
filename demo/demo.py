from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

import tensorflow as tf

a = tf.constant(32)
b = tf.constant(10)
c = tf.add(a, b)
sess = tf.Session()
print(sess.run(c))
sess.close()

