from cProfile import run
import numpy as np
import keras
from PIL import Image
import cv2
from keras.preprocessing import image
from keras.applications import imagenet_utils
import keras_load_img as l
from keras.applications.resnet50 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
import os
#from keras.applications import mobilenet



Input_size=224
nh=Input_size
nw=Input_size
batch_size=100
n=50000

#explicit_mean_reduction=False
#explicit_channel_reorder=False
NPY=False
n1=1000
n2=2000

models_dir='/home/ehsan/UvA/ARMCL/Khadas/ARMCL-Local/scripts/blobs_extractor/Working_tree/models/'
acc_dir="/home/ehsan/UvA/ARMCL/Khadas/ARMCL-Local/scripts/blobs_extractor/Working_tree/Accuracy/"
img_dir=acc_dir+'/Imagenet/'

networks=['mobilenet','resnet50']
network=networks[0]

#MobileNet:
if network==networks[0]:
	m='MobileNet/MobileNet.h5'

#ResNet50:
if network==networks[1]:
	m='Resnet50/ResNet50.h5'

model_dir=models_dir+m

# input: image number, outout: image name (iamgenet)
def image_name(im_n):
    im_n=str(im_n).zfill(8)
    im_n=im_n
    img_name='ILSVRC2012_val_'+im_n+'.JPEG'
    return img_dir+'/ILSVRC2012_img_val/'+img_name

#https://stackoverflow.com/questions/70180899/how-is-the-mobilenet-preprocess-input-in-tensorflow
#https://github.com/keras-team/keras/blob/2c48a3b38b6b6139be2da501982fd2f61d7d48fe/keras/applications/imagenet_utils.py#L168
#https://faroit.com/keras-docs/1.2.2/applications/
#https://www.kaggle.com/code/ilhamk/resnet50-example-preprocessing-check/notebook
#https://docs.w3cub.com/tensorflow~python/tf/keras/applications/resnet50/preprocess_input

mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
#mean = np.array([ 104.01, 116.67, 122.68 ], dtype=np.float32)
def prepare_image(file):
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    #img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    if m=='Resnet50/ResNet50.h5':
        return keras.applications.resnet50.preprocess_input(img_array)
        #return keras.applications.resnet50.preprocess_input(img_array,data_format=None,mode='caffe')
        #return imagenet_utils.preprocess_input(img_array, data_format=None, mode='caffe')
    if m=='MobileNet/MobileNet.h5':
        return keras.applications.mobilenet.preprocess_input(img_array)

def prepare_image_2(file):
    img = l.load_img(file, target_size=(224, 224))
    img=np.array(img,dtype="float32")
    #img=np.expand_dims(img,axis=0)
    #ToDo
    if m=='Resnet50/ResNet50.h5':
        img = img[..., ::-1]
        img=img-mean
        return img
        #img[..., 0] -= mean[0]
        #img[..., 1] -= mean[1]
        #img[..., 2] -= mean[2]
        #return img
    if m=='MobileNet/MobileNet.h5':
        img /= 127.5
        img -= 1.
        return img










	
def resize_image_i(im_n):
	#im_n=input("enter number of image: ")
	im_n=str(im_n).zfill(8)
	im_n1='ILSVRC2012_val_'+im_n+'.JPEG'
	
	print(im_n)
	im1=img_dir+'/ILSVRC2012_img_val/'+im_n1
	if NPY:
		im_n2='ILSVRC2012_val_'+im_n+'.npy'
		p=img_dir+'/keras_'+network+'_preprocessed_'+str(Input_size)+'/'
		os.makedirs(p, exist_ok=True)
		im2=p+im_n2
	else:
		im_n2='ILSVRC2012_val_'+im_n+'.PNG'
		p=img_dir+'/ILSVRC2012_img_val_keras_resized_PNG_'+str(Input_size)+'/'
		os.makedirs(p, exist_ok=True)
		im2=p+im_n2



	img = l.load_img(im1, target_size=(224, 224))
    #img=np.expand_dims(img,axis=0)
	if NPY:
		img = np.array(img,dtype="float32")
		if m=='Resnet50/ResNet50.h5':
			img = img[..., ::-1]
			img=img-mean
			with open(im2,'wb') as f:
				np.save(f,img)
			#img[..., 0] -= mean[0]
			#img[..., 1] -= mean[1]
			#img[..., 2] -= mean[2]
			#return img
		if m=='MobileNet/MobileNet.h5':
			img /= 127.5
			img -= 1.
			with open(im2,'wb') as f:
				np.save(f,img)

	else:
		img.save(im2)
		
		



for i in range(n1,n2+1):
	resize_image_i(i)

