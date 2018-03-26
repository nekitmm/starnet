**StarNet** is a neural network that can remove stars from images in one simple step leaving only background.

More technically it is a convolutional residual net with a typical encoder-decoder architecture, with L1, Adversarial and Perceptual losses.

**Small example:**

<div align="center">
  <img src="https://github.com/nekitmm/starnet/blob/master/example.jpg"><br><br>
</div>

**Intro**

Star removal using classical methods is a very tricky and painful multi-step procedure, which is hard to master
and hard to get nice results from, especially in case of images busy with stars.

This will remove most of stars from input image in one step, leaving only really huge ones, and leaving (well, hopefully)
intact all other small bright things whose shape is significantly different from that of a typical star, like small spiral
galaxies, fine details in nebulosity, HH objects, etc.

It is intended to be used by astrophotographers. Primary use is for background nebulosity enhancement in rich star fields,
but it can also help in creation of nice starless image.

**Literature**

This code is partially based on pix2pix code and ideas from pix2pix paper.

pix2pix code: https://github.com/phillipi/pix2pix

pix2pix paper: <a href="https://arxiv.org/pdf/1611.07004v1.pdf">Image-to-Image Translation with Conditional Adversarial Networks</a>

Udea of using Perceptual Adversarial losses is from this paper as well as some other ideas:

<a href="https://arxiv.org/abs/1706.09138">Perceptual Adversarial Networks for Image-to-Image Transformation</a>

Other papers I took ideas from of found useful during development:

<a href="https://arxiv.org/abs/1701.05957">Image De-raining Using a Conditional Generative Adversarial Network</a>

<a href="https://arxiv.org/abs/1606.08921">Image Restoration Using Convolutional Auto-encoders with Symmetric Skip Connections</a>

<a href="http://www.columbia.edu/~jwp2128/Papers/FuHuangetal2017.pdf">Removing rain from single images via a deep detail network</a>

**Usage**

Its primary purpose is to partially replace initial steps of star removal in tutorials, like one by Gerald Wechselberger,
aiming to enhance nebulosity without pushing stars up. The tutorial itself was available under
<a href="https://dl.dropboxusercontent.com/u/57910417/Howto_enhance_nebuala_without_pushing_stars.wmv">this</a> link, but not any more,
for some reason. Haven't found any newer links to it. Anyway, you got the idea.

**Suggested Work Flow**

The transformation by this neural net can be part of PixInsight/Photoshop processing work flow. Something like this:

1. Start from **stretched** LRGB image. Save as 8 bits/channel tif file.
2. Feed to StarNet.
3. Open output in Photoshop, correct some of the worst artefacts that will most likely appear in the image. If there are huge stars
in the image left, you will have to take care of them in some other way.
4. Perhaps use some noise reduction (we don't want to push noise up).
5. Use resulting image to enhance nebulosity using some method (Screen-Mask-Invert for example) or enjoy the result.
6. Profit!

**Some tips and tricks**

1. Do not use heavily processed images for input. If star shapes are unusual (if star reduction techniques were used, too much sharpening,
deconvolution, image was heavily resized, etc) the performance might be much worse than it could be. I am almost sure. Or maybe it will be
better in your case, but that's unlikely.

2. This newral net was trained using data from refractor telescope (FSQ106 + QSI 683 wsg-8), so will work best for data from similar imaging
systems. That's a bit of a bad news for many users of reflector telescopes. If you have long spikes in your images, then the net will not take
care of these spikes very well, I tried and didn't like results too much. What you can do, however, is you can train the net a bit more
using your own data. Just prepare one or two starless images from your data and run training for 20 epochs or so. This should significantly
improve results and make the net much better for **your** data.

3. The point above is valid for all images, for which you are not getting good results. You can prepare a starless version even of a **small
part** of that image (but not smaller than 256x256 pixels) and run training for 20 epochs or so on this image. This should improve quality of
transformation of the whole image. This will take a lot of time, of course, but in the end you not only getting a starless image, but also
train a net so it will perform better next time on a similar image.

4. Also it might help, for example, to make image brighter (darker) if it is unusually dark (bright), or things like that.

**Weights for the network**

This repository contains only a code base needed to run the net. Pre-trained weights are available for now through my dropbox account
because they weight too much (lol) - about 700 Mb. You need to download them to begin using StarNet:

<div align="center">
<a href="https://www.dropbox.com/s/6zrlhrd03hlo810/starnet_weights.zip?dl=0">LINK</a>
</div>

**Some technical info**
 
Throughout the code all input and output images are 8 bits per channel tif images.
This code in original form will not read any images other than these (like jpeg, etc), but you can change that if you like.

**Prerequisites**

Python and Tensorflow, preferably Tensorflow-GPU if you have an NVidia GPU. In this case you will also need CUDA and CuDNN libraries.

I tested it in Python 3.6.3 (Anaconda) + TensorFlow-GPU 1.4.0

Environment: Win 10 + Cygwin

GPU was NVidia GeForce 840M 2Gb, compute capability 5.0, CUDA version 9.1



 
      Modes of use:
      
      python.exe -u starnet.py transform <input_image> - The most probable use. This command will transform 
                                                         input image (namely will remove stars) and will 
                                                         create a mask showing changes regions. Output images
                                                         names will be <input_image>_starless.tif and
                                                         <input_image>_mask.tif
      
      python.exe -u starnet.py train                   - Use if you want to train the model some more using
                                                         your training data. This will also output logs and
                                                         showcases of training transformations in 
                                                         './logs' sub-folder.
                                                         
      python.exe -u starnet.py plot                    - Will plot graphs from log files inside './logs' 
                                                         sub-folder.
      
      python.exe -u starnet.py train new               - Use only if you want to train a completely new model,
                                                         erasing all older weights and other output, such as
                                                         logs. Use only if you know what you are doing!
 
      python.exe -u starnet.py test <input_image>      - This will create some test transformations of patches
                                                         of the input. Similar to transform, but instead of
                                                         transforming an entire image, will create showcases
                                                         of transformations. Fast option to take a look at 
                                                         possible result.
                                                         By default output will be in './test' sub-folder.







<div align="center">
  <img src="https://github.com/nekitmm/starnet/blob/master/show.jpg"><br><br>
</div>

**Licenses**

Code is available under MIT License, please review LICENSE file inside repo. Its very permissive, but no liability
or warranty of any kind.

Weights are available under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">
Attribution-NonCommercial-ShareAlike 4.0 International Creative Commons</a> license.

In short:
You are free to use and redistribute them in any medium or format, but only **under the same** license terms.
You can transform, and build your projects upon them.
You can **NOT** use them for commercial purposes.
You must give appropriate credit for usage of these weights.

The weights are distributed on an "AS IS" BASIS WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED.
