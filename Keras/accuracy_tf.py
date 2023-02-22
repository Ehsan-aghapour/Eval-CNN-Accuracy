import tensorflow as tf
import numpy as np

import numpy as np
import keras
from PIL import Image
import cv2
from keras.preprocessing import image
from keras.applications import imagenet_utils
import keras_load_img as l

Input_size=224
nh=Input_size
nw=Input_size
batch_size=100
n=50000


models_dir='/home/ehsan/UvA/ARMCL/Khadas/ARMCL-Local/scripts/blobs_extractor/Working_tree/models/'
acc_dir="/home/ehsan/UvA/ARMCL/Khadas/ARMCL-Local/scripts/blobs_extractor/Working_tree/Accuracy/"
img_dir=acc_dir+'/Imagenet/'

#MobileNet:
#m='MobileNet/MobileNet.pb'

#ResNet50:
m='Resnet50/ResNet50.pb'

model_dir=models_dir+m

# input: image number, outout: image name (iamgenet)
def image_name(im_n):
    im_n=str(im_n).zfill(8)
    im_n=im_n
    img_name='ILSVRC2012_val_'+im_n+'.JPEG'
    return img_dir+'/ILSVRC2012_img_val/'+img_name

#https://stackoverflow.com/questions/70180899/how-is-the-mobilenet-preprocess-input-in-tensorflow
#https://github.com/keras-team/keras/blob/2c48a3b38b6b6139be2da501982fd2f61d7d48fe/keras/applications/imagenet_utils.py#L168

'''
def prepare_image(file):
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    #img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array)
    '''
#https://faroit.com/keras-docs/1.2.2/applications/
#https://www.kaggle.com/code/ilhamk/resnet50-example-preprocessing-check/notebook
#https://docs.w3cub.com/tensorflow~python/tf/keras/applications/resnet50/preprocess_input

mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
def prepare_image(file):
  img = image.load_img(file, target_size=(224, 224))
  img_array = image.img_to_array(img)
  #img_array_expanded_dims = np.expand_dims(img_array, axis=0)
  if m=='Resnet50/ResNet50.pb':
    return keras.applications.resnet50.preprocess_input(img_array, data_format=None, mode='caffe')
    #return imagenet_utils.preprocess_input(img_array, data_format=None, mode='caffe')
  if m=='MobileNet/MobileNet.pb':
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

def prepare_images(image_list):
    imgs=[prepare_image(image_name(i)) for i in image_list]
    imgs=np.array(imgs)


def run_inference(image_number,_model_dir=model_dir):
    
    img=p_i(image_name(image_number))
    img=np.expand_dims(img,axis=0)
    

    with tf.gfile.GFile(_model_dir, "rb") as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
 
    with tf.Graph().as_default() as graph:
      tf.import_graph_def(graph_def,
                          input_map=None,
                          return_elements=None,
                          name=""
      )

    #all_nodes = [n for n in tf.get_default_graph().as_graph_def().node]
    #all_nodes = [n for n in graph.as_graph_def().node] 
    #all_ops = graph.get_operations()
    all_tensors = [tensor for op in graph.get_operations() for tensor in op.values()]
    all_placeholders = [placeholder for op in graph.get_operations() if op.type=='Placeholder' for placeholder in op.values()]
    y_pred=all_tensors[-1]
    x=all_placeholders[0]
    
    sess= tf.Session(graph=graph)
    feed_dict_testing = {x: img}
    prediction=sess.run(y_pred, feed_dict=feed_dict_testing)
    print(prediction)

    # obtain the top-5 predictions
    results = imagenet_utils.decode_predictions(prediction)
    print(results)




def Evaluate():
  with tf.gfile.GFile(model_dir, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def,input_map=None, return_elements=None, name="")

    #all_nodes = [n for n in tf.get_default_graph().as_graph_def().node]
    #all_nodes = [n for n in graph.as_graph_def().node] 
    #all_ops = graph.get_operations()
    all_tensors = [tensor for op in graph.get_operations() for tensor in op.values()]
    all_placeholders = [placeholder for op in graph.get_operations() if op.type=='Placeholder' for placeholder in op.values()]
    y_pred=all_tensors[-1]
    x=all_placeholders[0]
    sess= tf.Session(graph=graph)
    
    f=open(model_dir.split('/')[-1]+'.csv','w')
    l='labels.txt'
    label_names = np.loadtxt(l, str, delimiter='\t')
    last_i=1
  for indx in range(batch_size+1,n+batch_size+1,batch_size):
    print(f"Inference batch with index {last_i} to {indx-1}")
    image_list=list(range(last_i,indx))
    imgs=[prepare_image(image_name(i)) for i in image_list]
    imgs=np.array(imgs)
    
    feed_dict_testing = {x: imgs}
    prob=sess.run(y_pred, feed_dict=feed_dict_testing)
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

Evaluate()  


'''
# obtain the top-5 predictions
results = imagenet_utils.decode_predictions(prediction)
print(results)
'''
