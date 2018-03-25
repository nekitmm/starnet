**StarNet** is a neural network that can remove stars from images leaving only background.
 
 Throughout the code all input and output images are 8 bits per channel tif images.
 This code in original form will not read any images other than these (like jpeg, etc), but you can change that if you like.
 
 
      Modes of use:
      
      python.exe -u starnet.py transform <input_image> - The most probable use. This command will transform input image (namely will remove stars)
                                                         and will create a mask showing changes regions. Output images names will be 
                                                         <input_image>_starless.tif and <input_image>_mask.tif
      
      python.exe -u starnet.py train                   - Use if you want to train the model some more using your training data.
                                                         This will also output logs and showcases of training transformations in 
                                                         './logs' sub-folder.
                                                         
      python.exe -u starnet.py plot                    - Will plot graphs from log files. Into './logs' sub-folder.
      
      python.exe -u starnet.py train new               - use only if you want to train a completely new model, erasing all older weights and other
                                                         output, such as logs. Use only if you know what you are doing!
 
      python.exe -u starnet.py test <input_image>      - this will create some test transformations of patches of the input
                                                         Similar to transform, but instead of transforming an entire image,
                                                         will create showcases of transformation. Fast option to take a look at possible result.
                                                         By default output will be in './test' sub-folder.







<div align="center">
  <img src="https://github.com/nekitmm/starnet/blob/master/epoch_5.tif"><br><br>
</div>