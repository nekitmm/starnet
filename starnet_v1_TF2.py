from PIL import Image as img

from os import listdir
from os.path import isfile, join
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
import copy
import pickle

from matplotlib import pyplot as plt

from IPython import display

class StarNet():
    def __init__(self, mode:str, window_size:int = 512, stride:int = 256, lr:float = 1e-4, train_folder:str = './train/', batch_size:int = 1):
        assert mode in ['RGB', 'Greyscale'], "Mode should be either RGB or Greyscale"
        self.mode = mode
        if self.mode == 'RGB': self.input_channels = 3
        else: self.input_channels = 1
        self.window_size = window_size
        self.stride = stride
        self.train_folder = train_folder
        self.batch_size = batch_size
        self.history = {}
        self._ema = 0.9999
        self.weights = []
        self.lr = lr
        
        self.original = []
        self.starless = []   
        
    def __str__(self):
        return "Starnet instance"
        
    def load_training_dataset(self):
        self.weights = []
        original_files = [f for f in listdir(self.train_folder + "/original/") if isfile(join(self.train_folder + "/original/", f))\
                          and f.endswith(".tif")]
        starless_files = [f for f in listdir(self.train_folder + "/starless/") if isfile(join(self.train_folder + "/starless/", f))\
                          and f.endswith(".tif")]
        
        assert len(original_files) == len(starless_files), 'Numbers of files in `starless` and `original` subfolders should be equal'
        
        assert len(original_files) > 0 and len(starless_files) > 0, 'No training data found in {}'.format(self.train_folder)
        
        for i in range(len(original_files)):
            assert(original_files[i] == starless_files[i]), 'Corresponding names of original and starless files should be equal'
        
        print("Total training images found: {}".format(len(original_files)))
        
        self.original = []
        self.starless = []
        
        for i in original_files:
            self.original.append(np.array(img.open(self.train_folder + "/original/" + i), dtype = np.float32))
            self.starless.append(np.array(img.open(self.train_folder + "/starless/" + i), dtype = np.float32))
            
        
        total_pixels = 0
        
        for i in range(len(original_files)):
            assert self.original[i].shape == self.starless[i].shape, 'Image sizes are not equal: {}/original/{} and {}/starless/{}'\
                                                                      .format(self.train_folder, original_files[i],\
                                                                      self.train_folder, starless_files[i])
            
            total_pixels += self.original[i].shape[0] * self.original[i].shape[1]
            self.weights.append(self.original[i].shape[0] * self.original[i].shape[1])
        
        print("Total size of training images: %.2f MP" % (total_pixels / 1e6))
        
        self.iters_per_epoch = total_pixels // (self.window_size * self.window_size)
        
        self.weights = [i / np.sum(self.weights) for i in self.weights]
        
        print("One epoch is set to %d iterations" % self.iters_per_epoch)
        print("Training dataset has been successfully loaded!")
        
    def load_model(self, weights = None, history = None):
        self.G = self._generator(m = 64)
        self.D = self._discriminator()
        
        self.gen_optimizer = tf.optimizers.Adam(self.lr)
        self.dis_optimizer = tf.optimizers.Adam(self.lr / 4)
        
        self.D.build(input_shape = (None, self.window_size, self.window_size, self.input_channels))
        self.G.build(input_shape = (None, self.window_size, self.window_size, self.input_channels))
        
        #if weights: self.G.load_weights(weights + '_' + self.mode + '.h5')
        if weights:
            self.G.load_weights(weights + '_G_' + self.mode + '.h5')
            self.D.load_weights(weights + '_D_' + self.mode + '.h5')
        if history:
            with open(history + '_' + self.mode + '.pkl', "rb") as h:
                self.history = pickle.load(h)
                
    def initialize_model(self):
        self.load_model()
    
    def _ramp(self, x):
        return tf.clip_by_value(x, 0, 1)

    def _augmentator(self, o, s):
        # flip horizontally
        if np.random.rand() < 0.50:
            o = np.flip(o, axis = 1)
            s = np.flip(s, axis = 1)
        
        # flip vertically
        if np.random.rand() < 0.50:
            o = np.flip(o, axis = 0)
            s = np.flip(s, axis = 0)
        
        # rotate 90, 180 or 270
        if np.random.rand() < 0.50:
            k = int(np.random.rand() * 3 + 1)
            o = np.rot90(o, k, axes = (1, 0))
            s = np.rot90(s, k, axes = (1, 0))
        
        if self.mode == 'RGB':
            # tweak colors
            if np.random.rand() < 0.70:
                ch = int(np.random.rand() * 3)
                m = np.min((o, s))
                offset = np.random.rand() * 0.25 - np.random.rand() * m
                o[:, :, ch] = o[:, :, ch] + offset * (1.0 - o[:, :, ch])
                s[:, :, ch] = s[:, :, ch] + offset * (1.0 - s[:, :, ch])
            
            # flip channels
            if np.random.rand() < 0.70:
                seq = np.arange(3)
                np.random.shuffle(seq)
                Xtmp = np.copy(o)
                Ytmp = np.copy(s)
                for i in range(3):
                    o[:, :, i] = Xtmp[:, :, seq[i]]
                    s[:, :, i] = Ytmp[:, :, seq[i]]
        else:
            # tweak brightness
            if np.random.rand() < 0.70:
                m = np.min((o, s))
                offset = np.random.rand() * 0.25 - np.random.rand() * m
                o[:, :] = o[:, :] + offset * (1.0 - o[:, :])
                s[:, :] = s[:, :] + offset * (1.0 - s[:, :])
        
        o = np.clip(o, 0.0, 1.0)
        s = np.clip(s, 0.0, 1.0)
        
        if self.mode == 'RGB': return o, s
        else:
            c = np.random.randint(3)
            return o[:, :, c, None], s[:, :, c, None]
            
    def _get_sample(self, r, h, w, type:str):
        assert type in ['original', 'starless']
        if type == 'original':
            return self.original[r][h:h+self.window_size, w:w+self.window_size] / 255
        else:
            return self.starless[r][h:h+self.window_size, w:w+self.window_size] / 255
        
    def generate_input(self, iterations = 1, augmentation = False):
        for _ in range(iterations):
            o = np.zeros((self.batch_size, self.window_size, self.window_size, self.input_channels), dtype = np.float32)
            s = np.zeros((self.batch_size, self.window_size, self.window_size, self.input_channels), dtype = np.float32)
            for i in range(self.batch_size):
                if augmentation:
                    r = int(np.random.choice(range(len(self.original)), 1, p = self.weights))
                    h = np.random.randint(self.original[r].shape[0] - self.window_size)
                    w = np.random.randint(self.original[r].shape[1] - self.window_size)
                    o[i], s[i] = self._augmentator(self._get_sample(r, h, w, type = 'original'),\
                                                   self._get_sample(r, h, w, type = 'starless'))
                else:
                    o[i] = self._get_sample(type = 'original')
                    s[i] = self._get_sample(type = 'starless')
        return o, s
        
        
    def train(self, epochs, augmentation = True, plot_progress = False, plot_interval = 50, save_backups = True, warm_up = False):
        assert self.original != [], 'Training dataset was not loaded, use load_training_dataset() first'
        
        for e in range(epochs):
            for i in range(self.iters_per_epoch):
                
                x, y = self.generate_input(augmentation = augmentation)
                
                x = x * 2 - 1
                y = y * 2 - 1
                
                if warm_up: y = x
                
                if i % plot_interval == 0 and plot_progress:
                    plt.close()
                    fig, ax = plt.subplots(1, 3, sharex = True, figsize=(16.5, 16.5))
                    if self.mode == 'RGB':
                        ax[0].imshow((x[0] + 1) / 2)
                        ax[0].set_title('Original')
                        ax[1].imshow((self.G(x)[0] + 1) / 2)
                        ax[1].set_title('Starless')
                        ax[2].imshow((y[0] + 1) / 2)
                        ax[2].set_title('Target')
                    else:
                        ax[0].imshow((x[0, :, :, 0] + 1) / 2, cmap='gray', vmin = 0, vmax = 1)
                        ax[0].set_title('Original')
                        ax[1].imshow((self.G(x)[0, :, :, 0] + 1) / 2, cmap='gray', vmin = 0, vmax = 1)
                        ax[1].set_title('Starless')
                        ax[2].imshow((y[0, :, :, 0] + 1) / 2, cmap='gray', vmin = 0, vmax = 1)
                        ax[2].set_title('Target')
                    
                    display.clear_output(wait = True)
                    display.display(plt.gcf())
                
                if i > 0:
                    print("\rEpoch: %d. Iteration %d / %d Loss %f    " % (e, i, self.iters_per_epoch, self.history['total'][-1]), end = '')
                else:
                    print("\rEpoch: %d. Iteration %d / %d " % (e, i, self.iters_per_epoch), end = '')
                
                with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
                    gen_output = self.G(x)
                    
                    p1_real, p2_real, p3_real, p4_real, p5_real, p6_real, p7_real, p8_real, predict_real = self.D(y)
                    p1_fake, p2_fake, p3_fake, p4_fake, p5_fake, p6_fake, p7_fake, p8_fake, predict_fake = self.D(gen_output)
                    
                    d = {}
                    
                    dis_loss = tf.reduce_mean(-(tf.math.log(predict_real + 1E-8) + tf.math.log(1 - predict_fake + 1E-8)))
                    d['dis_loss'] = dis_loss
                    
                    gen_loss_GAN = tf.reduce_mean(-tf.math.log(predict_fake + 1E-8))
                    d['gen_loss_GAN'] = gen_loss_GAN
                    
                    gen_p1 = tf.reduce_mean(tf.abs(p1_fake - p1_real))
                    d['gen_p1'] = gen_p1
                    
                    gen_p2 = tf.reduce_mean(tf.abs(p2_fake - p2_real))
                    d['gen_p2'] = gen_p2
                    
                    gen_p3 = tf.reduce_mean(tf.abs(p3_fake - p3_real))
                    d['gen_p3'] = gen_p3
                    
                    gen_p4 = tf.reduce_mean(tf.abs(p4_fake - p4_real))
                    d['gen_p4'] = gen_p4
                    
                    gen_p5 = tf.reduce_mean(tf.abs(p5_fake - p5_real))
                    d['gen_p5'] = gen_p5
                    
                    gen_p6 = tf.reduce_mean(tf.abs(p6_fake - p6_real))
                    d['gen_p6'] = gen_p6
                    
                    gen_p7 = tf.reduce_mean(tf.abs(p7_fake - p7_real))
                    d['gen_p7'] = gen_p7
                    
                    gen_p8 = tf.reduce_mean(tf.abs(p8_fake - p8_real))
                    d['gen_p8'] = gen_p8
                    
                    gen_L1 = tf.reduce_mean(tf.abs(y - gen_output))
                    d['gen_L1'] = gen_L1 * 100
                    
                    gen_loss = gen_loss_GAN * 0.1 + gen_p1 * 0.1 + gen_p2 * 10 + gen_p3 * 10 + gen_p4 * 10 + gen_p5 * 10 + gen_p6 * 10 + gen_p7 * 10 + gen_p8 * 10 + gen_L1 * 100
                    d['total'] = gen_loss
                    
                    for k in d:
                        if k in self.history.keys():
                            self.history[k].append(d[k] * (1 - self._ema) + self.history[k][-1] * self._ema)
                        else:
                            self.history[k] = [d[k]]
                    
                    gen_grads = gen_tape.gradient(gen_loss, self.G.trainable_variables)
                    self.gen_optimizer.apply_gradients(zip(gen_grads, self.G.trainable_variables))
                    
                    dis_grads = dis_tape.gradient(dis_loss, self.D.trainable_variables)
                    self.dis_optimizer.apply_gradients(zip(dis_grads, self.D.trainable_variables))
                    
            if save_backups:
                if e % 2 == 0:
                    self.G.save_weights("./starnet_backup_G_even.h5")
                    self.D.save_weights("./starnet_backup_D_even.h5")
                else:
                    self.G.save_weights("./starnet_backup_G_odd.h5")
                    self.D.save_weights("./starnet_backup_D_odd.h5")
            
            if plot_progress: plt.close()
    
    
    def plot_history(self, last = None):
        assert self.history != {}, 'Empty training history, nothing to plot'
        fig, ax = plt.subplots(4, 3, sharex = True, figsize=(16, 14))
        
        keys = list(self.history.keys())
        
        keys = [k for k in keys if k != '']
        
        for i in range(4):
            for j in range(3):
                if last: ax[i][j].plot(self.history[keys[j+3*i]][-last:])
                else: ax[i][j].plot(self.history[keys[j+3*i]])
                ax[i][j].set_title(keys[j+3*i])
                
    def save_model(self, weights_filename, history_filename = None):
        self.G.save_weights(weights_filename + '_G_' + self.mode + '.h5')
        self.D.save_weights(weights_filename + '_D_' + self.mode + '.h5')
        if history_filename:
            with open(history_filename + '_' + self.mode + '.pkl', 'wb') as f:
                pickle.dump(self.history, f)
      
    def transform(self, name):
        image = np.array(img.open(name), dtype = np.float32)
        
        if self.mode == 'Greyscale' and len(image.shape) == 3:
            raise ValueError('You loaded Greyscale model, but the image is RGB!')
            
        if self.mode == 'Greyscale':
            image = image[:, :, None]
        
        if self.mode == 'RGB' and len(image.shape) == 2:
            raise ValueError('You loaded RGB model, but the image is Greyscale!')
        
        offset = int((self.window_size - self.stride) / 2)
        
        h, w, _ = image.shape
        
        ith = int(h / self.stride) + 1
        itw = int(w / self.stride) + 1
        
        dh = ith * self.stride - h
        dw = itw * self.stride - w
        
        image = np.concatenate((image, image[(h - dh) :, :, :]), axis = 0)
        image = np.concatenate((image, image[:, (w - dw) :, :]), axis = 1)
        
        h, w, _ = image.shape
        image = np.concatenate((image, image[(h - offset) :, :, :]), axis = 0)
        image = np.concatenate((image[: offset, :, :], image), axis = 0)
        image = np.concatenate((image, image[:, (w - offset) :, :]), axis = 1)
        image = np.concatenate((image[:, : offset, :], image), axis = 1)
        
        image /= 255
        
        image = image * 2 - 1
        
        output = copy.deepcopy(image)
        
        h, w, _ = image.shape
        ith = int(h / self.stride)
        itw = int(w / self.stride)
        for i in range(ith):
            for j in range(itw):
                tile = np.expand_dims(image[self.stride*i:self.stride*i+self.window_size, self.stride*j:self.stride*j+self.window_size, :], axis = 0)
                tile = (self.G(tile)[0] + 1) / 2
                tile = tile[offset:offset+self.stride, offset:offset+self.stride, :]
                output[self.stride*i+offset:self.stride*(i+1)+offset, self.stride*j+offset:self.stride*(j+1)+offset, :] = tile
        
        output = np.clip(output, 0, 1)
        
        if self.mode == 'Greyscale':
            return output[offset:-(offset+dh), offset:-(offset+dw), 0] * 255
        else:
            return output[offset:-(offset+dh), offset:-(offset+dw), :] * 255
    
    def _generator(self, m):
        layers = []
    
        filters = [64, 128, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512, 256, 128, 64]
        
        input = L.Input(shape=(self.window_size, self.window_size, self.input_channels), name = "gen_input_image")
        
        # layer 0
        convolved = L.Conv2D(filters[0], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(input)
        layers.append(convolved)
            
        # layer 1
        rectified = L.LeakyReLU(alpha = 0.2)(layers[-1])
        convolved = L.Conv2D(filters[1], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
        normalized = L.BatchNormalization()(convolved, training = True)
        layers.append(normalized)
            
        # layer 2
        rectified = L.LeakyReLU(alpha = 0.2)(layers[-1])
        convolved = L.Conv2D(filters[2], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
        normalized = L.BatchNormalization()(convolved, training = True)
        layers.append(normalized)
            
        # layer 3
        rectified = L.LeakyReLU(alpha = 0.2)(layers[-1])
        convolved = L.Conv2D(filters[3], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
        normalized = L.BatchNormalization()(convolved, training = True)
        layers.append(normalized)
            
        # layer 4
        rectified = L.LeakyReLU(alpha = 0.2)(layers[-1])
        convolved = L.Conv2D(filters[4], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
        normalized = L.BatchNormalization()(convolved, training = True)
        layers.append(normalized)
            
        # layer 5
        rectified = L.LeakyReLU(alpha = 0.2)(layers[-1])
        convolved = L.Conv2D(filters[5], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
        normalized = L.BatchNormalization()(convolved, training = True)
        layers.append(normalized)
        
        # layer 6
        rectified = L.LeakyReLU(alpha = 0.2)(layers[-1])
        convolved = L.Conv2D(filters[6], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
        normalized = L.BatchNormalization()(convolved, training = True)
        layers.append(normalized)
        
        # layer 7
        rectified = L.LeakyReLU(alpha = 0.2)(layers[-1])
        convolved = L.Conv2D(filters[7], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
        normalized = L.BatchNormalization()(convolved, training = True)
        layers.append(normalized)
        
        # layer 8
        rectified = L.ReLU()(layers[-1])
        deconvolved = L.Conv2DTranspose(filters[8], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
        normalized = L.BatchNormalization()(deconvolved, training = True)
        layers.append(normalized)
            
        # layer 9
        concatenated = tf.concat([layers[-1], layers[6]], axis = 3)
        rectified = L.ReLU()(concatenated)
        deconvolved = L.Conv2DTranspose(filters[9], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
        normalized = L.BatchNormalization()(deconvolved, training = True)
        layers.append(normalized)
        
        # layer 10
        concatenated = tf.concat([layers[-1], layers[5]], axis = 3)
        rectified = L.ReLU()(concatenated)
        deconvolved = L.Conv2DTranspose(filters[10], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
        normalized = L.BatchNormalization()(deconvolved, training = True)
        layers.append(normalized)
            
        # layer 11
        concatenated = tf.concat([layers[-1], layers[4]], axis = 3)
        rectified = L.ReLU()(concatenated)
        deconvolved = L.Conv2DTranspose(filters[11], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
        normalized = L.BatchNormalization()(deconvolved, training = True)
        layers.append(normalized)
            
        # layer 12
        concatenated = tf.concat([layers[-1], layers[3]], axis = 3)
        rectified = L.ReLU()(concatenated)
        deconvolved = L.Conv2DTranspose(filters[12], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
        normalized = L.BatchNormalization()(deconvolved, training = True)
        layers.append(normalized)
            
        # layer 13
        concatenated = tf.concat([layers[-1], layers[2]], axis = 3)
        rectified = L.ReLU()(concatenated)
        deconvolved = L.Conv2DTranspose(filters[13], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
        normalized = L.BatchNormalization()(deconvolved, training = True)
        layers.append(normalized)
            
        # layer 14
        concatenated = tf.concat([layers[-1], layers[1]], axis = 3)
        rectified = L.ReLU()(concatenated)
        deconvolved = L.Conv2DTranspose(filters[14], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
        normalized = L.BatchNormalization()(deconvolved, training = True)
        layers.append(normalized)
            
        # layer 15
        concatenated = tf.concat([layers[-1], layers[0]], axis = 3)
        rectified = L.ReLU()(concatenated)
        deconvolved = L.Conv2DTranspose(self.input_channels, kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
        rectified = L.ReLU()(deconvolved)
        output = tf.math.subtract(input, rectified)
        
        return K.Model(inputs = input, outputs = output, name = "generator")
        
    def _discriminator(self):
        layers = []
        filters = [32, 64, 64, 128, 128, 256, 256, 256, 8]
        
        input = L.Input(shape=(self.window_size, self.window_size, self.input_channels), name = "dis_input_image")
        
        # layer 1
        convolved = L.Conv2D(filters[0], kernel_size = 3, strides = (1, 1), padding="same")(input)
        rectified = L.LeakyReLU(alpha = 0.2)(convolved)
        layers.append(rectified)
            
        # layer 2
        convolved = L.Conv2D(filters[1], kernel_size = 3, strides = (2, 2), padding="valid")(layers[-1])
        normalized = L.BatchNormalization()(convolved, training = True)
        rectified = L.LeakyReLU(alpha = 0.2)(normalized)
        layers.append(rectified)
            
        # layer 3
        convolved = L.Conv2D(filters[2], kernel_size = 3, strides = (1, 1), padding="same")(layers[-1])
        normalized = L.BatchNormalization()(convolved, training = True)
        rectified = L.LeakyReLU(alpha = 0.2)(normalized)
        layers.append(rectified)
            
        # layer 4
        convolved = L.Conv2D(filters[3], kernel_size = 3, strides = (2, 2), padding="valid")(layers[-1])
        normalized = L.BatchNormalization()(convolved, training = True)
        rectified = L.LeakyReLU(alpha = 0.2)(normalized)
        layers.append(rectified)
            
        # layer 5
        convolved = L.Conv2D(filters[4], kernel_size = 3, strides = (1, 1), padding="same")(layers[-1])
        normalized = L.BatchNormalization()(convolved, training = True)
        rectified = L.LeakyReLU(alpha = 0.2)(normalized)
        layers.append(rectified)
            
        # layer 6
        convolved = L.Conv2D(filters[5], kernel_size = 3, strides = (2, 2), padding="valid")(layers[-1])
        normalized = L.BatchNormalization()(convolved, training = True)
        rectified = L.LeakyReLU(alpha = 0.2)(normalized)
        layers.append(rectified)
            
        # layer 7
        convolved = L.Conv2D(filters[6], kernel_size = 3, strides = (1, 1), padding="same")(layers[-1])
        normalized = L.BatchNormalization()(convolved, training = True)
        rectified = L.LeakyReLU(alpha = 0.2)(normalized)
        layers.append(rectified)
            
        # layer 8
        convolved = L.Conv2D(filters[7], kernel_size = 3, strides = (2, 2), padding="valid")(layers[-1])
        normalized = L.BatchNormalization()(convolved, training = True)
        rectified = L.LeakyReLU(alpha = 0.2)(normalized)
        layers.append(rectified)
            
        # layer 9
        convolved = L.Conv2D(filters[8], kernel_size = 3, strides = (2, 2), padding="valid")(layers[-1])
        normalized = L.BatchNormalization()(convolved, training = True)
        rectified = L.LeakyReLU(alpha = 0.2)(normalized)
        layers.append(rectified)
            
        # layer 10
        dense = L.Dense(1)(layers[-1])
        sigmoid = tf.nn.sigmoid(dense)
        layers.append(sigmoid)
        
        output = [layers[0], layers[1], layers[2], layers[3], layers[4], layers[5], layers[6], layers[7], layers[-1]]
            
        return K.Model(inputs = input, outputs = output, name = "discriminator")
