# This file is a part of StarNet code.
# https://github.com/nekitmm/starnet
# 
# StarNet is a neural network that can remove stars from images leaving only background.
# 
# Throughout the code all input and output images are 8 bits per channel tif images.
# This code in original form will not read any images other than these (like jpeg, etc), but you can change that if you like.
# 
# Copyright (c) 2018 Nikita Misiura
# http://www.astrobin.com/users/nekitmm/
# 
# This code is distributed on an "AS IS" BASIS WITHOUT WARRANTIES OF ANY KIND, express or implied.
# Please review LICENSE file before use.

import tensorflow as tf
import numpy as np
import model
import starnet_utils
from PIL import Image as img
from scipy.misc import toimage
import os

PANELS = 5                             # Number of panels in output pictures.
OUT_DIR = './test/'                    # Output directory for test showcases.
WINDOW_SIZE = 256                      # Size of the image fed to net. Do not change until you know what you are doing! Default is 256
                                       # and changing this will force you to train the net anew.

def test(input, numtests = 20):
    if not os.path.exists(OUT_DIR):
                os.makedirs(OUT_DIR)
    
    # placeholder for tensorflow
    X = tf.placeholder(tf.float32, shape = [None, WINDOW_SIZE, WINDOW_SIZE, 3], name = "X")
    Y = tf.placeholder(tf.float32, shape = [None, WINDOW_SIZE, WINDOW_SIZE, 3], name = "Y")
    
    # create model
    train, avers, outputs = model.model(X, Y)
    
    #initialize variables
    init = tf.global_variables_initializer()
    
    # create saver instance to load model
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # initialize all variables and start training
        sess.run(init)
        
        # restore state of the model
        print("Restoring current state of the model...")
        saver.restore(sess, "./model.ckpt")
        print("Done!")
        
        print("Opening test image...")
        test = np.array(img.open(input), dtype = np.float32)
        print("Done!")
        
        head = np.array(img.open("./test_head.tif"), dtype = np.float32)
        head /= 255
        
        for i in range(numtests):
            print("Test %d..." % i)
            all = np.zeros((1, WINDOW_SIZE * 2, 3))
            all = np.concatenate((head, all), axis = 0)
            for d in range(PANELS):
                X_test = starnet_utils.get_test_samples(test, 1)
                output = sess.run(outputs, feed_dict = {X:X_test})
                im = (np.concatenate((X_test[0], output[0]), axis = 1) + 1 ) / 2
                all = np.concatenate((all, im), axis = 0)
                all = np.concatenate((all, np.zeros((1, WINDOW_SIZE * 2, 3))), axis = 0)
            toimage(all * 255, cmin = 0, cmax = 255).save(OUT_DIR + '/test_' + str(i) + '.tif')
            print("Done!")