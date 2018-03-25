**StarNet** is a neural network that can remove stars from images in one simple step leaving only background.

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

**Usage**

Its primary purpose is to partially replace initial steps of star removal in tutorials, like a great tutorial by Gerald Wechselberger,
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

4. Also it might help to make image brighter (darker) if it is unusually dark (bright).
 
Throughout the code all input and output images are 8 bits per channel tif images.
This code in original form will not read any images other than these (like jpeg, etc), but you can change that if you like.
 
 
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