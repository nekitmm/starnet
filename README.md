<h1>Update 2</h1>
 Updated documentation for using Tensorflow-directml on windows for broad support on any modern gpu with sufficient memory.

<hr>
<h1>Update</h1>

Pushed a new implementation os starnet in TF2.x. The whole implementation is in one file *starnet_v1_TF2.py*.

I also created a few Jupyter notebooks for ease of use:

1. starnet_v1_TF2_transform.ipynb - loads and transforms an image.
2. starnet_v1_TF2.ipynb - more detailed example that loads a model and shows how to train it (really simple as well I think).

Weights for the new model can be found <a href="https://www.dropbox.com/s/lcgn5gvnxpo27s5/starnet_weights2.zip?dl=0">here</a>.

<hr>

**StarNet** is a neural network that can remove stars from images in one simple step leaving only background.

More technically it is a convolutional residual net with encoder-decoder architecture and with L1, Adversarial and Perceptual losses.

**Small example:**

<div align="center">
  <img src="https://github.com/nekitmm/starnet/blob/master/for_git/1.jpg"><br><br>
</div>

<center><h1>Intro</h1></center>

Star removal using classical methods is a very tricky and painful multi-step procedure, which is hard to master
and hard to get nice results from, especially in case of images busy with stars.

This neural net will remove most of stars from input image in one step, leaving only really huge ones, and leaving (well, hopefully)
intact all other small bright things whose shape is significantly different from that of a typical star, like small spiral
galaxies, fine details in nebulosity, HH objects, etc.

It is intended to be used by astrophotographers. Primary use is for background nebulosity enhancement in rich star fields,
but it can also help in creation of nice starless image.

<center><h1>Literature</h1></center>

This code is partially based on pix2pix code and ideas from pix2pix paper.

pix2pix code: https://github.com/phillipi/pix2pix

pix2pix paper: <a href="https://arxiv.org/pdf/1611.07004v1.pdf">Image-to-Image Translation with Conditional Adversarial Networks</a>

Udea of using Perceptual Adversarial losses is from this paper as well as some other ideas:

<a href="https://arxiv.org/abs/1706.09138">Perceptual Adversarial Networks for Image-to-Image Transformation</a>

Other papers I took ideas from or found useful during development:

<a href="https://arxiv.org/abs/1701.05957">Image De-raining Using a Conditional Generative Adversarial Network</a>

<a href="https://arxiv.org/abs/1606.08921">Image Restoration Using Convolutional Auto-encoders with Symmetric Skip Connections</a>

<a href="http://www.columbia.edu/~jwp2128/Papers/FuHuangetal2017.pdf">Removing rain from single images via a deep detail network</a>

<center><h1>Usage</h1></center>

Its primary purpose is to partially replace initial steps of star removal in tutorials, like one by Gerald Wechselberger,
aiming to enhance nebulosity without pushing stars up. The tutorial itself was available under
<a href="https://dl.dropboxusercontent.com/u/57910417/Howto_enhance_nebuala_without_pushing_stars.wmv">this</a> link, but not any more,
for some reason. Haven't found any newer links to it. Anyway, you got the idea.

<center><h1>Suggested Work Flow</h1></center>

The transformation by this neural net can be part of PixInsight/Photoshop processing work flow. Something like this:

1. Start from **stretched** LRGB image. Save as 8 bits/channel tif file.
2. Feed to StarNet.
3. Open output in Photoshop, correct some of the worst artifacts that will most likely appear in the image. If there are huge stars
in the image left, you will have to take care of them in some other way.
4. Perhaps use some noise reduction (we don't want to push noise up).
5. Use resulting image to enhance nebulosity using some method (Screen-Mask-Invert for example) or enjoy the result.
6. ?
7. Profit!

<center><h1>Weights for the network</h1></center>

This repository contains only a code base needed to run the net, but does not contain all the weights (which are uploaded into LFS.
Pre-trained weights are also available for now through my dropbox account
because they weight too much (lol) - about 700 Mb. You need to download them and unpack into root folder of starnet (the one with all
python scripts, not into some sub-folder) to begin using StarNet:

<div align="center">
<a href="https://www.dropbox.com/s/atcs42ox4n99w96/starnet_weights.zip?dl=0">LINK</a>
</div>

This will rewrite some files in this repo!

The weights are also uploaded into LFS, but depending how you clone the repo they might or might not download.

<center><h1>Some tips and tricks</h1></center>

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

5. Sometimes the net will leave small stars in the output, if you feed it very busy image. In this case it is helpful to feed output to the 
net again.

<center><h1>Training Dataset</h1></center>

This is one part I'd like to keep for myself for now, but you can create your own dataset creating starless versions of your images.
One extremely important note: the only difference between two images (original and starless) should be stars, which are replaced
by background in the starless version. The rest of the image should be perfectly intact. If you will throw in a starless image which
is super nice looking, but in a process of creation of this image you altered much more than just stars, this will only degrade
network performance. Also be aware that quality of star removal by the net will never be better than that in your training set, so
if you want a top-notch quality starless images, be ready to provide training images of even higher quality.

I left one training image to show organization of folders my code expects. Inside a folder named 'train' there are two sub-folders named
'original' and starless', both should contain equal number of images with identical name pairs.

<center><h1>Some technical info</h1></center>
 
Throughout the code all input and output images I use are 8 bits per channel **tif** images.
This code should read some other image formats (like jpeg, 16bit tiff, etc), but I did not check all of them.

<center><h1>Prerequisites and installation Guide</h1></center>

for all environments, using conda is strongly encouraged, installation instructions assume a conda install of either Anaconda python or miniconda:
  - https://docs.conda.io/en/latest/miniconda.html

## Windows (New!)

On windows we can now run starnet on GPU on any modern graphics card! (yes AMD and Intel included)

### Prerequisites

Windows 10 Version 1709, 64-bit (Build 16299 or higher) or Windows 11 Version 21H2, 64-bit (Build 22000 or higher)

### Installation

Once anaconda is installed, you can open an "anaconda powershell prompt" to proceed.

We use the environment config file provided to configure and install all the dependencies:

#### With GPU support (Windows):
```
conda env create -f environment-windows.yml
```
#### With CUDA support (linux or windows):
```
conda env create -f environment-lnx-cuda.yml
```
#### CPU only(Mac, Linux, Windows):
```
conda env create -f environment-cpu.yml
```
### Post installation
Initialize the environment with:
```
conda activate starnet
```
And you're ready to go!


Originally tested on:
- Win 10 + Cygwin
- NVidia GeForce 840M 2Gb, compute capability 5.0, CUDA version 9.1

Windows general GPU support tested on:
- Win 10 12H1
- AMD RX 6800-XT 16GB

<center><h1>Usage</h1></center>
 
      
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


<center><h1>Couple more examples</h1></center>

More examples can be found <a href="https://www.astrobin.com/339099/0/">here</a>.

<div align="center">
  <img src="https://github.com/nekitmm/starnet/blob/master/for_git/2.jpg"><br><br>
</div>

Original:

<div align="center">
  <img src="https://github.com/nekitmm/starnet/blob/master/for_git/3.jpg"><br><br>
</div>

Starless:

<div align="center">
  <img src="https://github.com/nekitmm/starnet/blob/master/for_git/4.jpg"><br><br>
</div>


<center><h1>FAQ</h1></center>

**What is all this 'python-sudo-mumbo-jumbo'**?

This whole thing works as command line program, which means that there is no graphical interface: you have to
run it in a console using some text commands (like ones you see above) and it outputs text (and writes image files
of course!).

2. Where exactly do I put weights of the network?

All the files you download should be in one folder: all the files with extension .py (starnet.py, train.py, transform.py, etc.) should
be in the same folder with weights for the network (model.ckpt.data-00000-of-00001, model.ckpt.index, model.ckpt.meta, etc.)

<center><h1>Some Troubleshooting</h1></center>

1. <b>Error: 'No package named tensorflow'.</b> Should be pretty self-explanatory: your python can not find tensorflow. That means you did not
run pip to install it (<b>pip install tensorflow</b>) or something went wrong during this step if you did.

2. <b>Error: 'ImportError: libcublas.so.9.0: cannot open shared object file: No such file or directory.'</b> You are trying to use GPU version of
tensorflow and you don't have CUDA properly installed.

3. <b>Error: 'ValueError: The passed save_path is not a valid checkpoint: ./model.ckpt.'</b> You did not copy network weights into proper location. See above.

Let me know if you have any other issues with the code. (Preferably through *Astrobin*)

<center><h1>Licenses</h1></center>

Code is available under MIT License, please review LICENSE.md file inside repo. Its very permissive, but no liability
or warranty of any kind.

Weights are available under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">
Attribution-NonCommercial-ShareAlike 4.0 International Creative Commons</a> license.

In short:
You are free to use and redistribute them in any medium or format, but only **under the same** license terms.
You can transform, and build your projects upon them.
You can **NOT** use them for commercial purposes.
You must give appropriate credit for usage of these weights.

The weights are distributed on an "AS IS" BASIS WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED.
