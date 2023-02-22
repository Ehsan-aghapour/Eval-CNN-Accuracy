Accuracy.py:

inference caffe model for imagenet images. 
The input is 50000 images from validataion image and output is csv file including 5 top prediction with wordnet id (NID). 
Then you should move the csv file to evaluation dir and use Eval.py to calculate accuracy based on ground truth.

you could directly load imagenet images preprocess and make inference, or 
load resiezed images (with image preprocess just resize them to 227*227 using caffe.io.resize() function 
and then preprocess(subtract img_mean and transpose and reorder channels) and make inference)

you could determine number of images for evalution with n
and batch size for each inference with batch_size



rkk_inference:
Use it for convert to rknn model or inference rknn models.

if you run it with caffe.pb and caffe.caffemodel then it convert and export the converted rknn model an exit.

by setting do_quantization=True in rknn.build function it will do quantization by dataset.
dataset is also generated for images from i to j in the same function.
the type of quantization should be determined in rknn.conf (default is u8)

if run it with model.rknn then it will run the model 
by setting PC=0 it will be run on rockpi 
if PC=1 it will be run on pc (simulation) and is very slow


image_preprocess:
could be used to resize images and save in resized dir, or even more preprocess them by img_mean subtraction and transposing then save in .npy