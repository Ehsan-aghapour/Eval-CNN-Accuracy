#Alexnet:
#time python rknn_inference.py /home/ehsan/UvA/ARMCL/Khadas/ARMCL-Local/scripts/blobs_extractor/Working_tree/models/AlexNet/bvlc_alexnet/new/deploy.prototxt /home/ehsan/UvA/ARMCL/Khadas/ARMCL-Local/scripts/blobs_extractor/Working_tree/models/AlexNet/bvlc_alexnet/new/bvlc_alexnet.caffemodel

#Googlenet:
#time python rknn_inference.py /home/ehsan/UvA/ARMCL/Khadas/ARMCL-Local/scripts/blobs_extractor/Working_tree/models/GoogleNet/bvlc_googlenet/new/deploy.prototxt /home/ehsan/UvA/ARMCL/Khadas/ARMCL-Local/scripts/blobs_extractor/Working_tree/models/GoogleNet/bvlc_googlenet/new/bvlc_googlenet.caffemodel

#Squeezenet:
#time python rknn_inference.py /home/ehsan/UvA/ARMCL/Khadas/ARMCL-Local/scripts/blobs_extractor/Working_tree/models/SqueezeNet-master/SqueezeNet_v1.0/new/deploy.prototxt /home/ehsan/UvA/ARMCL/Khadas/ARMCL-Local/scripts/blobs_extractor/Working_tree/models/SqueezeNet-master/SqueezeNet_v1.0/new/squeezenet_v1.0.caffemodel

import numpy as np
import cv2
from rknn.api import RKNN
import sys
from PIL import Image

from scipy.ndimage import zoom
from skimage.transform import resize
#conda install scikit-image
#import caffe


# explicit_mean_reduction means that you want to do mean reduction explicitly
# explicit_channel_reorder means that you want to do channel reordering of caffe models explicitly
# otherwise in npu rknn.config these operations should be done 
# for F16 set explicit_mean_reduction and explicit_channel_reorder to false and set quant to false (it is for U8)
# (it is also ok to set both to true which explicitly do mean reducion and channel reordering)
# for U8 set both to False and do not forget to set quant to True
# (it is possible to set them to false both the quantization dataset should be preprocessed in terms of 
# mean reduction and channel reordering and so save them as proper size in npy format --> NPY=true)
explicit_mean_reduction=False
explicit_channel_reorder=False

# this is for quantization dataset if are .npy (resized and mean reduced and channels are compatible with model)
# otherwise it is png images that resized (if npy=flase explicit variables should be set to false)
NPY=False

quant=False
precompile=False
PC=False

#Alexnet and Squeezenet:
Input_size=227

#Others:
#Input_size=224

acc_dir="/home/ehsan/UvA/ARMCL/Khadas/ARMCL-Local/scripts/blobs_extractor/Working_tree/Accuracy/"
img_dir=acc_dir+'/Imagenet/'

name=sys.argv[1]
rknn_name=name.split('/')[-1].split('.')[0]+'.rknn'
csv_name=name.split('/')[-1].split('.')[0]+'.csv'
rknn_name_precompiled=name.split('/')[-1].split('.')[0]+'_precompiled.rknn'
model_type=name.split('.')[-1]

def show_outputs(outputs):
	output = outputs[0][0]
	output_sorted = sorted(output, reverse=True)
	top5_str = 'mobilenet_v1\n-----TOP 5-----\n'
	for i in range(5):
		value = output_sorted[i]
		index = np.where(output == value)
		for j in range(len(index)):
			if (i + j) >= 5:
				break
			if value > 0:
				topi = '{}: {}\n'.format(index[j], value)
			else:
				topi = '-1: 0.0\n'
			top5_str += topi
	print(top5_str)

def show_perfs(perfs):
	perfs = 'perfs: {}\n'.format(outputs)
	print(perfs)
	

# resize image function in caffe.io lib:
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
        if im_max > im_min:
            # skimage is fast but only understands {1,3} channel images
            # in [0, 1].
            im_std = (im - im_min) / (im_max - im_min)
            resized_std = resize(im_std, new_dims, order=interp_order, mode='constant')
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



img_mean = np.array([ 104.01, 116.67, 122.68 ], dtype=np.float32)
#img_mean=img_mean[[2,1,0]] # convert it to RGB	
def read_image_i(im_n):
	#im_n=input("enter number of image: ")
	im_n=str(im_n).zfill(8)
	im_n='ILSVRC2012_val_'+im_n+'.JPEG'
	print(im_n)
	im=img_dir+'/ILSVRC2012_img_val/'+im_n
	img=cv2.imread(im) #BGR
	if len(img.shape)==2:
		img = np.stack((img,)*3, axis=-1)
		print("gray image\n\n\n")
	
	if explicit_channel_reorder==False:	
		img=img[:,:,[2,1,0]] # RGB, convert to rgb because cv2 read image in bgr format
	
	img=resize_image(img,[Input_size,Input_size],1)	
	if explicit_mean_reduction:
		img=img-img_mean
		
	return img
	

def read_resized_image_1(im_n):
	im_n=str(im_n).zfill(8)
	im_n='ILSVRC2012_val_'+im_n+'.PNG'
	print(im_n)
	p=img_dir+'/ILSVRC2012_img_val_resized_PNG_'+str(Input_size)+'/'
	im=p+im_n
	#im='/home/ehsan/UvA/ARMCL/Khadas/ARMCL-Local/scripts/blobs_extractor/Working_tree/Accuracy/Imagenet/ILSVRC2012_img_val_resized/'+im_n
	img=cv2.imread(im)
	if explicit_channel_reorder==False:
		img=img[:,:,[2,1,0]] # because cv2 read image in bgr format
	if explicit_mean_reduction:
		img=img-img_mean
			
	return img
	
def read_npy(im_n):
	im_n=str(im_n).zfill(8)
	im_n='ILSVRC2012_val_'+im_n+'.npy'
	print(im_n)
	prefix=img_dir+'/preprocessed_'+str(Input_size)+'/'
	im=prefix+im_n
	img=np.load(im)
	return img
	
	
	
def dataset(i,j):
	prefix=img_dir+'/ILSVRC2012_img_val_resized_PNG_'+str(Input_size)+'/'
	if NPY:
		prefix=img_dir+'/preprocessed_'+str(Input_size)+'/'
	#_resized
	file_name=f'./dataset_{i}_{j}'
	f=open(file_name,'w')
	for im_n in range(i,j+1):
		im_n=str(im_n).zfill(8)
		im_n='ILSVRC2012_val_'+im_n
		if NPY:
			im_n=im_n+'.npy'
		else:		
			im_n=im_n+'.PNG'
		print(im_n)
		im=prefix+im_n
		f.write(im+'\n')
		
	return file_name
if __name__ == '__main__':

	# Create RKNN object
	rknn = RKNN()
	
	
	# init runtime environment
	print('--> Init runtime environment')
	
	if model_type!='rknn':
		# pre-process config
		print('--> config model')
		#rknn.config(channel_mean_value='103.94 116.78 123.68 58.82', reorder_channel='0 1 2')
		#rknn.config(channel_mean_value='103.94 116.78 123.68 1', reorder_channel='2 1 0')
		if explicit_channel_reorder:
			r_ch='0 1 2'
		else:
			r_ch='2 1 0'
		
		if explicit_mean_reduction:
			ch_m='0 0 0 1'
		else:
			ch_m='104.01 116.67 122.68 1'
		
		rknn.config(channel_mean_value=ch_m, reorder_channel=r_ch)
			
		

		print('done')

		# Load tensorflow model
		print('--> Loading model')
		#ret = rknn.load_tflite(model='./mobilenet_v1.tflite')
		print('--> Loading model')
	#rknn.load_tensorflow(tf_pb='model.pb',
	#                     inputs=['test_in'],
	#                     outputs=['test_out/BiasAdd'],
	#                     input_size_list=[[INPUT_SIZE]])
	#rknn.load_onnx(name)
		if model_type=='pb':
			if len(sys.argv)==5:
				inputs=sys.argv[2]
				outputs=sys.argv[3]
				INPUT_SIZE=sys.argv[4]
			rknn.load_tensorflow(tf_pb=name,
				inputs=[inputs],
				outputs=[outputs],
				input_size_list=[INPUT_SIZE])
				

		if model_type=='onnx':
			rknn.load_onnx(name)

		if model_type=='prototxt':
			#p=Path(name)
			#proto_name=p.with_suffix('.prototxt')
			print(f'name:{name},blobs:{sys.argv[2]}')
			ret = rknn.load_caffe(model=name,
				proto='caffe',
				blobs=sys.argv[2])

		print('done')

		# Build model
		print('--> Building model')
		if quant:
			ret = rknn.build(do_quantization=True, dataset=dataset(1000,2000))
		else:
			ret = rknn.build(do_quantization=False)
		#ret = rknn.build(do_quantization=False)
		#ret = rknn.build(do_quantization=False,rknn_batch_size=10)
		if ret != 0:
			print('Build mobilenet_v1 failed!')
			exit(ret)
		print('done')

		# Export rknn model
		print('--> Export RKNN model')
		ret = rknn.export_rknn(rknn_name)
		if ret != 0:
			print(f'Export {rknn_name} failed!')
			exit(ret)
		print('done')
		exit(0)
		####ret = rknn.init_runtime(target='RK3399Pro',rknn2precompile=True)
		ret = rknn.init_runtime()
		if ret != 0:
			print('Init runtime environment failed')
			exit(ret)
		'''ret = rknn.export_rknn_precompile_model(rknn_name_precompiled)
		if ret != 0:
			print('export prcompile failed')
			exit(ret)'''
	else:
		#rknn.load_rknn('./mobilenet_v1_sample_test_precompiled.rknn')
		rknn.load_rknn(sys.argv[1])
		if 'precompiled' in sys.argv[1]:
			PC=0
		if PC:
			ret = rknn.init_runtime()
		else:
			ret = rknn.init_runtime(target='RK3399Pro')
		if ret != 0:
			print('Init runtime environment failed')
			exit(ret)
		print('done')

	# Inference
	print('--> Running model')

	
	n=50000
	l='labels.txt'
	label_names = np.loadtxt(l, str, delimiter='\t')
	f=open(csv_name,'w')
	for i in range(1,n+1):
		#out = rknn.inference(inputs=[read_resized_image_1(i)])
		out = rknn.inference(inputs=[read_image_i(i)])
		
		#print(type(out))-->list
		#print(out.shape)
		prob=np.array(out)
		print(prob.shape)
		#prob = out['prob']
		prob = np.squeeze(prob)
		idx = np.argsort(-prob)

		l='labels.txt'
		label_names = np.loadtxt(l, str, delimiter='\t')
		for i in range(5):
			label = idx[i]
			#print('%d   %.2f - %s' % (idx[i],prob[label], label_names[label]))
			#print(label_names[label].split(' ')[0])
			
			f.write(label_names[label].split(' ')[0])
			if i==4:
				f.write('\n')
			else:
				f.write(',')
	
	
	
	#show_outputs(outputs)
	print('done')
	'''print(f'output shape:{np.array(outputs).shape}')
	p=0
	if p:
		for i,output in enumerate(outputs[0][0]):
			if (i+1)%10:
				print(f'{i:<4}:{output:^10.4f}',end='\t')
			else:
				print(f'{i:<4}:{output:^10.4f}')

	perf=0
	if perf:
		# perf
		print('--> Begin evaluate model performance')
		perf_results = rknn.eval_perf(inputs=[img])
		print('done')
	'''
	rknn.release()
	
	
	
	
	'''
	batch_size=10
	f=open('alex_rknn.csv','w')
	n=400
	last_i=1
	#l='imagenet_labels.txt'
	l='labels.txt'
	label_names = np.loadtxt(l, str, delimiter='\t')
	for indx in range(batch_size+1,n+batch_size+1,batch_size):
		print(f"start of batch with index {last_i} to {indx}")
		images=[]
		for j in range(last_i,indx):
			local_indx=((j-1)%batch_size)
			images.append(read_image_1(j))
		#input("first batch image read\n")
		out = rknn.inference(inputs=[np.array(images)])
		#print(type(out))
		#print(out.shape)
		prob=np.array(out)
		print(prob.shape)
		#prob = out['prob']
		prob = np.squeeze(prob)
		idx = np.argsort(-prob)
		#input("inference for first batch finished\n")
		for j in range(last_i,indx):
			local_indx=((j-1)%batch_size)
			for i in range(5):
				label = idx[local_indx][i]
				print('%d   %.2f - %s' % (idx[local_indx][i],prob[local_indx][label], label_names[label]))
				#print(label_names[label].split(' ')[0])
			
				f.write(label_names[label].split(' ')[0])
				if i==4:
					f.write('\n')
				else:
					f.write(',')
		#input("first batch was written\n")
		last_i=indx
	'''

