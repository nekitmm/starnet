**StarNet** is a neural network that can remove stars from images leaving only background.

It will remove most of stars from input image, except for really huge ones, leaving (well, hopefully) intact all other small bright
things whose shape is significantly different from that of a typical star, like small spiral galaxies, fine details in nebulosity,
HH objects, etc.

It is intended to be used by astrophotographers. Primary use is for background nebulosity enhancement in rich star fields,
but it can also help in creation of nice starless image.

Its primary purpose is to partially replace initial steps of star removal in tutorials, like that of Gerald Wechselberger,
aiming to enhance nebulosity without pushing stars too much. The tutorial itself was available under
<a href="https://dl.dropboxusercontent.com/u/57910417/Howto_enhance_nebuala_without_pushing_stars.wmv">this</a> link, but not anymore
for some reason. Anyway, you got the idea.
 
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