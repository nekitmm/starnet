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
from PIL import Image as img
import starnet_utils
from scipy.misc import toimage
import sys
import os
import time

PANELS = 5                             # Number of panels in output pictures showcasing image transformations done by net.
MAX_TRAIN_IMGS = 10                    # Max number of training images loaded in each epoch. Increasing this value will increase memory
                                       # consumption, but will make outputs like losses and accuracy more smooth. 10 is default, but the
                                       # optimal value will depend on your machine and training image sizes.
L1_MULT = 100                          # L1 loss multiplier for output (just because it is much smaller than the most). Default is 100.
ACC_MULT = 100                         # Accuracy multiplier (to convert to percentage).
LOGS_DIR = './logs/'                   # Folder to store text logs.
IMGS_DIR = './logs/'                   # Folder to store showcase images.
WINDOW_SIZE = 256                      # Size of the image fed to net. Do not change until you know what you are doing! Default is 256
                                       # and changing this will force you to train the net anew.

def train(epochs = 1, batch = 1, steps = 1000, output_freq = 50, verbose = False, gen_plots = True,
          images = True, log_freq = 50, resume = True, learning_rates = [0.0002, 0.0002]):
    
    if gen_plots:
        import plot
    
    # get a list of training images
    train_list = starnet_utils.list_train_images("./train/")
    
    # open head image for showcases
    head = np.array(img.open("./train_head.tif"), dtype = np.float32)
    head /= 255
    
    # placeholders for tensorflow
    X = tf.placeholder(tf.float32, shape = [None, WINDOW_SIZE, WINDOW_SIZE, 3], name = "X")
    Y = tf.placeholder(tf.float32, shape = [None, WINDOW_SIZE, WINDOW_SIZE, 3], name = "Y")
    
    #initialize variables
    train, avers, outputs = model.model(X, Y, lr = learning_rates)
    init = tf.global_variables_initializer()
    
    # create saver instance to save and load model parameters
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        # initialize all variables and start training
        sess.run(init)
        if(resume):
            # restore old state of the model
            print("Restoring previous state of the model...", end = ' ', flush = True)
            saver.restore(sess, "./model.ckpt")
            
            # open log files to append
            l1 = open(LOGS_DIR + '/L1_loss.txt', 'a')
            total = open(LOGS_DIR + '/total_loss.txt', 'a')
            p = open(LOGS_DIR + '/perceptual_losses.txt', 'a')
            acc = open(LOGS_DIR + '/accuracy.txt', 'a')
            adv = open(LOGS_DIR + '/adversarial_losses.txt', 'a')
            
            # load global step
            abs_step = int(np.loadtxt('./step', dtype = np.int))
            
            print("Done!", flush = True)
        else:
            # create LOGS_DIR directory if does not exist
            if not os.path.exists(LOGS_DIR):
                os.makedirs(LOGS_DIR)
                
            # create new log files
            l1 = open(LOGS_DIR + '/L1_loss.txt', 'w')
            total = open(LOGS_DIR + '/total_loss.txt', 'w')
            p = open(LOGS_DIR + '/perceptual_losses.txt', 'w')
            acc = open(LOGS_DIR + '/accuracy.txt', 'w')
            adv = open(LOGS_DIR + '/adversarial_losses.txt', 'w')
            
            # write headers into log files
            l1.write('Epoch	L1_loss (x%s)\n' % (L1_MULT))
            total.write('Epoch	Total_loss\n')
            p.write('Epoch	P1	P2	P3	P4	P5	P6	P7	P8\n')
            acc.write('Epoch	Accuracy %\n')
            adv.write('Epoch	GAN	Discriminative\n')
            
            # initialize global step as zero
            abs_step = 0
            
            # create IMGS_DIR directory if does not exist
            if not os.path.exists(IMGS_DIR):
                os.makedirs(IMGS_DIR)
        
        # here goes
        for e in range(epochs):
            start = time.time()
            # open few images from training set
            # we do not open all images at ones because it will take too much memory
            original, starless = starnet_utils.open_train_images("./train/", train_list, MAX_TRAIN_IMGS)
            
            # loop through training set
            for i in range(steps):
                abs_step += 1
                
                # get training examples and run one step of training
                # these are two lines that do the job, the rest of the code is just to output different stuff and save model
                (X_input, Y_input) = starnet_utils.get_train_samples_with_augmentation(original, starless, batch)
                sess.run(train, feed_dict = {X:X_input, Y:Y_input})
                
                # update absolute epoch
                abs_epoch = abs_step / steps
                
                # output to console if necessary
                if i % output_freq == 0:
                    losses = avers
                    if(verbose):
                        print("Epoch %d: step %d; discrim_loss: %.4f; gen_loss_GAN: %.4f; gen_loss_L1: %.4f; acc: %.2f" % (abs_epoch, i, losses[0].eval(), losses[1].eval(), L1_MULT * losses[2].eval(), ACC_MULT * losses[3].eval()))
                        print("                   p1_loss: %.4f; p2_loss: %.4f; p3_loss: %.4f; p4_loss: %.4f" % (losses[4].eval(), losses[5].eval(), losses[6].eval(), losses[7].eval()))
                        print("                   p5_loss: %.4f; p6_loss: %.4f; p7_loss: %.4f; p8_loss: %.4f" % (losses[8].eval(), losses[9].eval(), losses[10].eval(), losses[11].eval()))
                    else:
                        print("Epoch %d: step %d; L1 loss: %.4f; Total loss: %.4f; acc: %.2f" % (abs_epoch, i, L1_MULT * losses[2].eval(), losses[12].eval(), ACC_MULT * losses[3].eval()))
                    sys.stdout.flush()
                
                # output to files if necessary
                if i % log_freq == 0:
                    l1.write('%.4f	%.5f\n' % (abs_epoch, float(L1_MULT * losses[2].eval())))
                    total.write('%.4f	%.5f\n' % (abs_epoch, float(losses[12].eval())))
                    p.write('%.4f	%.5f	%.5f	%.5f	%.5f	%.5f	%.5f	%.5f	%.5f\n' % (abs_epoch, float(losses[4].eval()), float(losses[5].eval()), float(losses[6].eval()), float(losses[7].eval()), float(losses[8].eval()), float(losses[9].eval()), float(losses[10].eval()), float(losses[11].eval())))
                    acc.write('%.4f	%.5f\n' % (abs_epoch, float(ACC_MULT * losses[3].eval())))
                    adv.write('%.4f	%.5f	%.5f\n' % (abs_epoch, float(losses[1].eval()), float(losses[2].eval())))
            stop = time.time()
            t = float(stop - start)
            # final console output from the epoch
            print("Epoch %d took %.1f s; L1 loss: %.4f; Total loss: %.4f; acc: %.2f" % (abs_epoch - 1, t, L1_MULT * losses[2].eval(), losses[12].eval(), ACC_MULT * losses[3].eval()))
            sys.stdout.flush()
            
            # save weights of the model
            saver.save(sess, "./model.ckpt")
            
            # save few examples to take a look
            if images:
                # line of zeros
                all = np.zeros((1, WINDOW_SIZE * 3, 3))
                all = np.concatenate((head, all), axis = 0)
                for d in range(PANELS):
                    (X_test, Y_test) = starnet_utils.get_train_samples_with_augmentation(original, starless, 1)
                    output = sess.run(outputs, feed_dict = {X:X_test, Y:Y_test})
                    im = (np.concatenate((X_test[0], output[0], Y_test[0]), axis = 1) + 1 ) / 2
                    # add images
                    all = np.concatenate((all, im), axis = 0)
                    # add line of zeros
                    all = np.concatenate((all, np.zeros((1, WINDOW_SIZE * 3, 3))), axis = 0)
                toimage(all * 255, cmin = 0, cmax = 255).save(IMGS_DIR + '/epoch_' + str(int(abs_epoch - 1)) + '.tif')
            
            # save global step
            s = open('./step', 'w')
            s.write(str(abs_step))
            s.close()
            
            if gen_plots:
                plot.plot()
            
            # flush all file buffer to make sure everything is written
            # in case the script is aborted
            p.flush()
            l1.flush()
            acc.flush()
            adv.flush()
            total.flush()
            
    # close files
    p.close()
    l1.close()
    acc.close()
    adv.close()
    total.close()