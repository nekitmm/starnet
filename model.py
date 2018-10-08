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

def generator(input):
    layers = []
    
    filters = [64, 128, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512, 256, 128, 64]
    
    # layer 0
    with tf.variable_scope("g_conv0"):
        convolved = tf.layers.conv2d(input, filters[0], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.contrib.layers.xavier_initializer())
        layers.append(convolved)
        
    # layer 1
    with tf.variable_scope("g_conv1"):
        rectified = tf.nn.leaky_relu(layers[-1], 0.2)
        convolved = tf.layers.conv2d(rectified, filters[1], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.contrib.layers.xavier_initializer())
        normalized = tf.layers.batch_normalization(convolved, axis = 3, epsilon = 1e-5, momentum = 0.1, training = True, gamma_initializer = tf.random_normal_initializer(1.0, 0.02))
        layers.append(normalized)
        
    # layer 2
    with tf.variable_scope("g_conv2"):
        rectified = tf.nn.leaky_relu(layers[-1], 0.2)
        convolved = tf.layers.conv2d(rectified, filters[2], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.contrib.layers.xavier_initializer())
        normalized = tf.layers.batch_normalization(convolved, axis = 3, epsilon = 1e-5, momentum = 0.1, training = True, gamma_initializer = tf.random_normal_initializer(1.0, 0.02))
        layers.append(normalized)
        
    # layer 3
    with tf.variable_scope("g_conv3"):
        rectified = tf.nn.leaky_relu(layers[-1], 0.2)
        convolved = tf.layers.conv2d(rectified, filters[3], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.contrib.layers.xavier_initializer())
        normalized = tf.layers.batch_normalization(convolved, axis = 3, epsilon = 1e-5, momentum = 0.1, training = True, gamma_initializer = tf.random_normal_initializer(1.0, 0.02))
        layers.append(normalized)
        
    # layer 4
    with tf.variable_scope("g_conv4"):
        rectified = tf.nn.leaky_relu(layers[-1], 0.2)
        convolved = tf.layers.conv2d(rectified, filters[4], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.contrib.layers.xavier_initializer())
        normalized = tf.layers.batch_normalization(convolved, axis = 3, epsilon = 1e-5, momentum = 0.1, training = True, gamma_initializer = tf.random_normal_initializer(1.0, 0.02))
        layers.append(normalized)
        
    # layer 5
    with tf.variable_scope("g_conv5"):
        rectified = tf.nn.leaky_relu(layers[-1], 0.2)
        convolved = tf.layers.conv2d(rectified, filters[5], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.contrib.layers.xavier_initializer())
        normalized = tf.layers.batch_normalization(convolved, axis = 3, epsilon = 1e-5, momentum = 0.1, training = True, gamma_initializer = tf.random_normal_initializer(1.0, 0.02))
        layers.append(normalized)
    
    # layer 6
    with tf.variable_scope("g_conv6"):
        rectified = tf.nn.leaky_relu(layers[-1], 0.2)
        convolved = tf.layers.conv2d(rectified, filters[6], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.contrib.layers.xavier_initializer())
        normalized = tf.layers.batch_normalization(convolved, axis = 3, epsilon = 1e-5, momentum = 0.1, training = True, gamma_initializer = tf.random_normal_initializer(1.0, 0.02))
        layers.append(normalized)
    
    # layer 7
    with tf.variable_scope("g_conv7"):
        rectified = tf.nn.leaky_relu(layers[-1], 0.2)
        convolved = tf.layers.conv2d(rectified, filters[7], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.contrib.layers.xavier_initializer())
        normalized = tf.layers.batch_normalization(convolved, axis = 3, epsilon = 1e-5, momentum = 0.1, training = True, gamma_initializer = tf.random_normal_initializer(1.0, 0.02))
        layers.append(normalized)
    
    # layer 8
    with tf.variable_scope("g_deconv0"):
        rectified = tf.nn.relu(layers[-1])
        deconvolved = tf.layers.conv2d_transpose(rectified, filters[8], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.contrib.layers.xavier_initializer())
        normalized = tf.layers.batch_normalization(deconvolved, axis = 3, epsilon = 1e-5, momentum = 0.1, training = True, gamma_initializer = tf.random_normal_initializer(1.0, 0.02))
        layers.append(normalized)
        
    # layer 9
    with tf.variable_scope("g_deconv1"):
        concatenated = tf.concat([layers[-1], layers[6]], axis = 3)
        rectified = tf.nn.relu(concatenated)
        deconvolved = tf.layers.conv2d_transpose(rectified, filters[9], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.contrib.layers.xavier_initializer())
        normalized = tf.layers.batch_normalization(deconvolved, axis = 3, epsilon = 1e-5, momentum = 0.1, training = True, gamma_initializer = tf.random_normal_initializer(1.0, 0.02))
        layers.append(normalized)
    
    # layer 10
    with tf.variable_scope("g_deconv2"):
        concatenated = tf.concat([layers[-1], layers[5]], axis = 3)
        rectified = tf.nn.relu(concatenated)
        deconvolved = tf.layers.conv2d_transpose(rectified, filters[10], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.contrib.layers.xavier_initializer())
        normalized = tf.layers.batch_normalization(deconvolved, axis = 3, epsilon = 1e-5, momentum = 0.1, training = True, gamma_initializer = tf.random_normal_initializer(1.0, 0.02))
        layers.append(normalized)
        
    # layer 11
    with tf.variable_scope("g_deconv3"):
        concatenated = tf.concat([layers[-1], layers[4]], axis = 3)
        rectified = tf.nn.relu(concatenated)
        deconvolved = tf.layers.conv2d_transpose(rectified, filters[11], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.contrib.layers.xavier_initializer())
        normalized = tf.layers.batch_normalization(deconvolved, axis = 3, epsilon = 1e-5, momentum = 0.1, training = True, gamma_initializer = tf.random_normal_initializer(1.0, 0.02))
        layers.append(normalized)
        
    # layer 12
    with tf.variable_scope("g_deconv4"):
        concatenated = tf.concat([layers[-1], layers[3]], axis = 3)
        rectified = tf.nn.relu(concatenated)
        deconvolved = tf.layers.conv2d_transpose(rectified, filters[12], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.contrib.layers.xavier_initializer())
        normalized = tf.layers.batch_normalization(deconvolved, axis = 3, epsilon = 1e-5, momentum = 0.1, training = True, gamma_initializer = tf.random_normal_initializer(1.0, 0.02))
        layers.append(normalized)
        
    # layer 13
    with tf.variable_scope("g_deconv5"):
        concatenated = tf.concat([layers[-1], layers[2]], axis = 3)
        rectified = tf.nn.relu(concatenated)
        deconvolved = tf.layers.conv2d_transpose(rectified, filters[13], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.contrib.layers.xavier_initializer())
        normalized = tf.layers.batch_normalization(deconvolved, axis = 3, epsilon = 1e-5, momentum = 0.1, training = True, gamma_initializer = tf.random_normal_initializer(1.0, 0.02))
        layers.append(normalized)
        
    # layer 14
    with tf.variable_scope("g_deconv6"):
        concatenated = tf.concat([layers[-1], layers[1]], axis = 3)
        rectified = tf.nn.relu(concatenated)
        deconvolved = tf.layers.conv2d_transpose(rectified, filters[14], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.contrib.layers.xavier_initializer())
        normalized = tf.layers.batch_normalization(deconvolved, axis = 3, epsilon = 1e-5, momentum = 0.1, training = True, gamma_initializer = tf.random_normal_initializer(1.0, 0.02))
        layers.append(normalized)
        
    # layer 15
    with tf.variable_scope("g_deconv7"):
        concatenated = tf.concat([layers[-1], layers[0]], axis = 3)
        rectified = tf.nn.relu(concatenated)
        deconvolved = tf.layers.conv2d_transpose(rectified, 3, kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.contrib.layers.xavier_initializer())
        rectified = tf.nn.relu(deconvolved)
        output = tf.subtract(input, rectified)
        layers.append(output)
    
    return layers[-1]

def discriminator(inputs):
    layers = []
    
    filters = [32, 64, 64, 128, 128, 256, 256, 256, 8]
    
    # layer 1
    with tf.variable_scope("d_conv0"):
        convolved = tf.layers.conv2d(inputs, filters[0], kernel_size = 3, strides = (1, 1), padding="same", kernel_initializer = tf.contrib.layers.xavier_initializer())
        rectified = tf.nn.leaky_relu(convolved, 0.2)
        #layers[0] - perceptual loss #1
        layers.append(rectified)
        
    # layer 2
    with tf.variable_scope("d_conv1"):
        convolved = tf.layers.conv2d(layers[-1], filters[1], kernel_size = 3, strides = (2, 2), padding="valid", kernel_initializer = tf.contrib.layers.xavier_initializer())
        normalized = tf.layers.batch_normalization(convolved, axis = 3, epsilon = 1e-5, momentum = 0.1, training = True, gamma_initializer = tf.random_normal_initializer(1.0, 0.02))
        rectified = tf.nn.leaky_relu(normalized, 0.2)
        #layers[1] - perceptual loss #2
        layers.append(rectified)
        
    # layer 3
    with tf.variable_scope("d_conv2"):
        convolved = tf.layers.conv2d(layers[-1], filters[2], kernel_size = 3, strides = (1, 1), padding="same", kernel_initializer = tf.contrib.layers.xavier_initializer())
        normalized = tf.layers.batch_normalization(convolved, axis = 3, epsilon = 1e-5, momentum = 0.1, training = True, gamma_initializer = tf.random_normal_initializer(1.0, 0.02))
        rectified = tf.nn.leaky_relu(normalized, 0.2)
        #layers[2] - perceptual loss #3
        layers.append(rectified)
        
    # layer 4
    with tf.variable_scope("d_conv3"):
        convolved = tf.layers.conv2d(layers[-1], filters[3], kernel_size = 3, strides = (2, 2), padding="valid", kernel_initializer = tf.contrib.layers.xavier_initializer())
        normalized = tf.layers.batch_normalization(convolved, axis = 3, epsilon = 1e-5, momentum = 0.1, training = True, gamma_initializer = tf.random_normal_initializer(1.0, 0.02))
        rectified = tf.nn.leaky_relu(normalized, 0.2)
        #layers[3] - perceptual loss #4
        layers.append(rectified)
        
    # layer 5
    with tf.variable_scope("d_conv4"):
        convolved = tf.layers.conv2d(layers[-1], filters[4], kernel_size = 3, strides = (1, 1), padding="same", kernel_initializer = tf.contrib.layers.xavier_initializer())
        normalized = tf.layers.batch_normalization(convolved, axis = 3, epsilon = 1e-5, momentum = 0.1, training = True, gamma_initializer = tf.random_normal_initializer(1.0, 0.02))
        rectified = tf.nn.leaky_relu(normalized, 0.2)
        #layers[4] - perceptual loss #5
        layers.append(rectified)
        
    # layer 6
    with tf.variable_scope("d_conv5"):
        convolved = tf.layers.conv2d(layers[-1], filters[5], kernel_size = 3, strides = (2, 2), padding="valid", kernel_initializer = tf.contrib.layers.xavier_initializer())
        normalized = tf.layers.batch_normalization(convolved, axis = 3, epsilon = 1e-5, momentum = 0.1, training = True, gamma_initializer = tf.random_normal_initializer(1.0, 0.02))
        rectified = tf.nn.leaky_relu(normalized, 0.2)
        #layers[5] - perceptual loss #6
        layers.append(rectified)
        
    # layer 7
    with tf.variable_scope("d_conv6"):
        convolved = tf.layers.conv2d(layers[-1], filters[6], kernel_size = 3, strides = (1, 1), padding="same", kernel_initializer = tf.contrib.layers.xavier_initializer())
        normalized = tf.layers.batch_normalization(convolved, axis = 3, epsilon = 1e-5, momentum = 0.1, training = True, gamma_initializer = tf.random_normal_initializer(1.0, 0.02))
        rectified = tf.nn.leaky_relu(normalized, 0.2)
        #layers[6] - perceptual loss #7
        layers.append(rectified)
        
    # layer 8
    with tf.variable_scope("d_conv7"):
        convolved = tf.layers.conv2d(layers[-1], filters[7], kernel_size = 3, strides = (2, 2), padding="valid", kernel_initializer = tf.contrib.layers.xavier_initializer())
        normalized = tf.layers.batch_normalization(convolved, axis = 3, epsilon = 1e-5, momentum = 0.1, training = True, gamma_initializer = tf.random_normal_initializer(1.0, 0.02))
        rectified = tf.nn.leaky_relu(normalized, 0.2)
        #layers[7] - perceptual loss #8
        layers.append(rectified)
        
    # layer 9
    with tf.variable_scope("d_conv8"):
        convolved = tf.layers.conv2d(layers[-1], filters[8], kernel_size = 3, strides = (2, 2), padding="valid", kernel_initializer = tf.contrib.layers.xavier_initializer())
        normalized = tf.layers.batch_normalization(convolved, axis = 3, epsilon = 1e-5, momentum = 0.1, training = True, gamma_initializer = tf.random_normal_initializer(1.0, 0.02))
        rectified = tf.nn.leaky_relu(normalized, 0.2)
        #layers[8]
        layers.append(rectified)
        
    # layer 10
    with tf.variable_scope("d_dense"):
        dense = tf.layers.dense(layers[-1], 1)
        sigmoid = tf.nn.sigmoid(dense)
        #layers[9]
        layers.append(sigmoid)
        
    return layers[0], layers[1], layers[2], layers[3], layers[4], layers[5], layers[6], layers[7], layers[-1]
    
    
    
def model(inputs, targets, lr = [0.0002, 0.0002]):
    
    with tf.variable_scope("generator"):
        outputs = generator(inputs)
    
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            p1_real, p2_real, p3_real, p4_real, p5_real, p6_real, p7_real, p8_real, predict_real = discriminator(targets)
            

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse = True):
            p1_fake, p2_fake, p3_fake, p4_fake, p5_fake, p6_fake, p7_fake, p8_fake, predict_fake = discriminator(outputs)
            
    with tf.name_scope("discriminator_loss"):
        # maximizes probability of being real picture. Oprimal for discriminator is predict_real = 1, predict_fake = 0. 
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + 1E-8) + tf.log(1 - predict_fake + 1E-8)))
        
    with tf.name_scope("generator_loss"):
        # the best for generator is predict_fake = 1
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + 1E-8))
        gen_p1_loss = tf.reduce_mean(tf.abs(p1_fake - p1_real))
        gen_p2_loss = tf.reduce_mean(tf.abs(p2_fake - p2_real))
        gen_p3_loss = tf.reduce_mean(tf.abs(p3_fake - p3_real))
        gen_p4_loss = tf.reduce_mean(tf.abs(p4_fake - p4_real))
        gen_p5_loss = tf.reduce_mean(tf.abs(p5_fake - p5_real))
        gen_p6_loss = tf.reduce_mean(tf.abs(p6_fake - p6_real))
        gen_p7_loss = tf.reduce_mean(tf.abs(p7_fake - p7_real))
        gen_p8_loss = tf.reduce_mean(tf.abs(p8_fake - p8_real))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * 0.01 + gen_p1_loss * 0.1 + gen_p2_loss * 1 + gen_p3_loss * 1 + gen_p4_loss * 1 + gen_p5_loss * 1 + gen_p6_loss * 1 + gen_p7_loss * 1 + gen_p8_loss * 10 + gen_loss_L1 * 10
        
        gen_acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(targets * 100, tf.int8), tf.cast(outputs * 100, tf.int8)), tf.float32))
        
        losses = [discrim_loss, gen_loss_GAN, gen_loss_L1, gen_acc, gen_p1_loss, gen_p2_loss, gen_p3_loss, gen_p4_loss, gen_p5_loss, gen_p6_loss, gen_p7_loss, gen_p8_loss, gen_loss]
        
    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(lr[1])
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list = discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(lr[0])
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list = gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)
    
    ema = tf.train.ExponentialMovingAverage(decay = 0.999, zero_debias = True)
    update_losses = ema.apply(losses)
    
    avers = [ema.average(l) for l in losses]
    
    train = tf.group(update_losses, gen_train)
    
    return train, avers, outputs