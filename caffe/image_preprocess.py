from __future__ import print_function
import argparse
import numpy as np
import caffe

from PIL import Image
import os
import sys
import cv2
from scipy.ndimage import zoom

from skimage.transform import resize

Input_size=224
#explicit_mean_reduction=False
#explicit_channel_reorder=False
NPY=False
n1=1000
n2=2000

d='/home/ehsan/UvA/ARMCL/Khadas/ARMCL-Local/scripts/blobs_extractor/Working_tree/models'

_dir="/home/ehsan/UvA/ARMCL/ARMCL-Local/ARMCL-Local/Large/bvlc_alexnet"
acc_dir="/home/ehsan/UvA/ARMCL/Khadas/ARMCL-Local/scripts/blobs_extractor/Working_tree/Accuracy/"
img_dir=acc_dir+'/Imagenet/'

model_dir=d+'/AlexNet/bvlc_alexnet/new/'
model=model_dir+'/bvlc_alexnet.caffemodel'
proto=model_dir+'/deploy.prototxt'

def parse_args():
    parser = argparse.ArgumentParser(
        description='evaluate pretrained mobilenet models')
    parser.add_argument('--proto', dest='proto',
                        help="path to deploy prototxt.", type=str)
    parser.add_argument('--model', dest='model',
                        help='path to pretrained weights', type=str)
    parser.add_argument('--image', dest='image',
                        help='path to color image', type=str)

    args = parser.parse_args()
    return args, parser


global args, parser
#args, parser = parse_args()


def resize_image(im, new_dims, interp_order=1):
    """
    Resize an image array with interpolation.
    Parameters
    ----------
    im : (H x W x K) ndarray
    new_dims : (height, width) tuple of new dimensions.
    interp_order : interpolation order, default is linear.
    Returns
    -------
    im : resized ndarray with shape (new_dims[0], new_dims[1], K)
    """
    if im.shape[-1] == 1 or im.shape[-1] == 3:
        im_min, im_max = im.min(), im.max()
        #print(f'min and max {im_min},{im_max}')
        if im_max > im_min:
            # skimage is fast but only understands {1,3} channel images
            # in [0, 1].
            im_std = (im - im_min) / (im_max - im_min)
            #print(im_std)
            resized_std = resize(im_std, new_dims, order=interp_order, mode='constant')
            #print(im_std)
            resized_im = resized_std * (im_max - im_min) + im_min
        else:
            # the image is a constant -- avoid divide by 0
            ret = np.empty((new_dims[0], new_dims[1], im.shape[-1]),
                           dtype=np.float32)
            ret.fill(im_min)
            return ret
    else:
        # ndimage interpolates anything but more slowly.
        scale = tuple(np.array(new_dims, dtype=float) / np.array(im.shape[:2]))
        resized_im = zoom(im, scale + (1,), order=interp_order)
    return resized_im.astype(np.float32)


#m=np.load('/usr/lib/python3/dist-packages/caffe/imagenet/ilsvrc_2012_mean.npy')
#img_mean=m.mean(1).mean(1)
img_mean = np.array([ 104.01, 116.67, 122.68 ], dtype=np.float32)


def read_image():
	im_n=input("enter number of image: ")
	im_n=str(im_n).zfill(8)
	im_n='ILSVRC2012_val_'+im_n+'.JPEG'
	print(im_n)
	im=img_dir+'/ILSVRC2012_img_val/'+im_n
	img=Image.open(im)
	img=np.asarray(img)
	return img

	
def resize_image_i(im_n):
	#im_n=input("enter number of image: ")
	im_n=str(im_n).zfill(8)
	im_n1='ILSVRC2012_val_'+im_n+'.JPEG'
	
	print(im_n)
	im1=img_dir+'/ILSVRC2012_img_val/'+im_n1
	if NPY:
		im_n2='ILSVRC2012_val_'+im_n+'.npy'
		p=img_dir+'/preprocessed_'+str(Input_size)+'/'
		os.makedirs(p, exist_ok=True)
		im2=+im_n2
	else:
		im_n2='ILSVRC2012_val_'+im_n+'.PNG'
		p=img_dir+'/ILSVRC2012_img_val_resized_PNG_'+str(Input_size)+'/'
		os.makedirs(p, exist_ok=True)
		im2=p+im_n2
	
	img=cv2.imread(im1)
	#img=img[:,:,[2,1,0]] # convert to rgb because cv2 read image in bgr format
	#img=caffe.io.resize_image(img,[Input_size,Input_size],1)
	img=resize_image(img,[Input_size,Input_size],1)	
	if NPY:
		img=img-img_mean
		with open(im2,'wb') as f:
			np.save(f,img)		
		
	else:
		img=img[:,:,[2,1,0]] #-->RGB
		cv2.imwrite(im2,img)
		
		
		
	

	
	
def read_image_i_crop_center(im_n):
	im_n=str(im_n).zfill(8)
	im_n='ILSVRC2012_val_'+im_n+'.JPEG'
	print(im_n)
	im=img_dir+'/ILSVRC2012_img_val/'+im_n
	nh, nw = 224, 224
	img = caffe.io.load_image(im)
	h, w, _ = img.shape
	if h < w:
		off = (w - h) // 2
		img = img[:, off:off + h]
	else:
		off = (h - w) // 2
		img = img[off:off + h, :]
		
	img = caffe.io.resize_image(img, [nh, nw])
	return img
	
	







nh=Input_size
nw=Input_size


# net.blobs['layername'].data is output of layer layername
# net.params['layername'][0].data is weight of layer layername
# net.params['layername'][1].data is biases of layer layername

batch_size=1000
#if preprocess:
#	net.blobs['data'].reshape(batch_size, 3, nh, nw)

f=open('alex.csv','w')



last_i=1
l='labels.txt'
label_names = np.loadtxt(l, str, delimiter='\t')

for i in range(n1,n2+1):
	resize_image_i(i)

'''
if preprocess:
	for indx in range(n1+batch_size+1,n2+batch_size+1,batch_size):
		print(f"start of batch with index {last_i} to {indx}")
		for j in range(last_i,indx):
			local_indx=((j-1)%batch_size)
			net.blobs['data'].data[local_indx] = transformer.preprocess('data', read_image_i(j))
			im_n=str(j).zfill(8)
			im_n='ILSVRC2012_val_'+im_n+'.npy'
			print(im_n)
			im='/home/ehsan/UvA/ARMCL/Khadas/ARMCL-Local/scripts/blobs_extractor/Working_tree/Accuracy/Imagenet/preprocessed/'+im_n
			#print(type(net.blobs['data'].data[local_indx]))
			#print(net.blobs['data'].data[local_indx])
			with open(im,'wb') as f:
				np.save(f,net.blobs['data'].data[local_indx])
	
	
else:		
	for i in range(n1,n2+1):
		resize_image_i(i)	

'''
'''
if preprocess:
	

	#img_mean = np.array([104.00698793, 116.66876762, 122.67891434], dtype=np.float32)


	caffe.set_mode_cpu()

	#net = caffe.Net(args.proto, args.model, caffe.TEST)
	net = caffe.Net(proto, model, caffe.TEST)



	#this transformer use by read_image_1 and others and next transformer to be used with _read_image_1
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	print(net.blobs['data'].data.shape)
	#transformer = caffe.io.Transformer({'data': [10,3,375,500]})
	#transformer.set_transpose('data', (2, 0, 1))  # row to col

	transformer.set_channel_swap('data', (2, 1, 0))  # RGB to BGR
	#transformer.set_raw_scale('data', 255)  # [0,1] to [0,255] # As my image is ppm it is on range of 0 to 255 not 0 to 1 so this trans is not required
	transformer.set_mean('data', img_mean)
	#transformer.set_input_scale('data', 0.017) # scale in ARMCL is 1
	transformer.set_input_scale('data', 1)



	_transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	_transformer.set_transpose('data', (2, 0, 1))  # row to col
	_transformer.set_channel_swap('data', (2, 1, 0))  # RGB to BGR
	# this scale is applied before mean substraction
	_transformer.set_raw_scale('data', 255)  # [0,1] to [0,255]
	_transformer.set_mean('data', img_mean)
	# this scale is applied after mean substraction:
	#_transformer.set_input_scale('data', 0.017)

'''
