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
#from keras.applications import mobilenet
import threading

Input_size=224
nh=Input_size
nw=Input_size
batch_size=100
n=50000
res={}

models_dir='/home/ehsan/Partial_Q/models/'
acc_dir="/home/ehsan/Partial_Q/"
img_dir=acc_dir+'/Imagenet/'
#MobileNet:
m='MobileNet/MobileNet.h5'
model_dir=models_dir+m


# ResNet50:
# m='Resnet50/ResNet50.h5'

# input: image number, outout: image name (iamgenet)
def image_name(im_n):
    im_n=str(im_n).zfill(8)
    im_n=im_n
    img_name='ILSVRC2012_val_'+im_n+'.JPEG'
    return img_dir+'/ILSVRC2012_img_val/'+img_name

# https://stackoverflow.com/questions/70180899/how-is-the-mobilenet-preprocess-input-in-tensorflow
# https://github.com/keras-team/keras/blob/2c48a3b38b6b6139be2da501982fd2f61d7d48fe/keras/applications/imagenet_utils.py#L168
# https://faroit.com/keras-docs/1.2.2/applications/
# https://www.kaggle.com/code/ilhamk/resnet50-example-preprocessing-check/notebook
# https://docs.w3cub.com/tensorflow~python/tf/keras/applications/resnet50/preprocess_input

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


def run_inference(image_list,model=None,_model_dir=model_dir):
    # make predictions on test image using mobilenet
    imgs=[prepare_image(image_name(i)) for i in image_list]
    imgs=np.array(imgs)
    if model==None:
        model=keras.models.load_model(_model_dir)
    #img=prepare_image(image_name(image_number))
    prediction = model.predict(imgs)
    
    # obtain the top-5 predictions
    #results = imagenet_utils.decode_predictions(prediction)
    #print(results)
    return prediction


def Evaluate():
    model=keras.models.load_model(model_dir)
    f=open(model_dir.split('/')[-1]+'.csv','w')
    l='labels.txt'
    label_names = np.loadtxt(l, str, delimiter='\t')
    last_i=1
    for indx in range(batch_size+1,n+batch_size+1,batch_size):
        print(f"Inference batch with index {last_i} to {indx-1}")
        prob=run_inference(list(range(last_i,indx)),model)
        prob = np.squeeze(prob)
        idx = np.argsort(-prob)
        #input("inference for first batch finished\n")
        for j in range(last_i,indx):
            local_indx=((j-1)%batch_size)
            for i in range(5):
                label = idx[local_indx][i]
                #print('%d   %.2f - %s' % (idx[local_indx][i],prob[local_indx][label], label_names[label]))
                #print(label_names[label].split(' ')[0])
            
                f.write(label_names[label].split(' ')[0])
                if i==4:
                    f.write('\n')
                else:
                    f.write(',')
        #input("first batch was written\n")
        last_i=indx



