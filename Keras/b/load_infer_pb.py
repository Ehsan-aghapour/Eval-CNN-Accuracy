import tensorflow as tf
import os
import numpy as np
import os,glob,cv2
import sys,argparse

'''
import tensorflow as tf
def load_pb(path_to_pb):
    with tf.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph
'''

# First, pass the path of the image
#dir_path = os.path.dirname(os.path.realpath(file))
dir_path='/home/ehsan/UvA/DRLPM/Khadas/Large/_aml_npu_sdk_6.4.3/acuity-toolkit/conversion_scripts/'
##image_path=sys.argv[1]
image_path='data/space_shuttle_224.jpg'
filename = dir_path +'/' +image_path
image_size=224
num_channels=3
images = []
 
##  Reading the image using OpenCV
image = cv2.imread(filename)
#image=tf.io.read_file('data/space_shuttle_224.jpg')
#print(image)
#print(image.shape)


##   Resizing the image to our desired size and preprocessing will be done exactly as done during training
##image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
images.append(image)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.subtract (images, 127.5)
images = np.divide (images, 128)
##images = np.multiply(images, 1.0/255.0)
##   The input to the network is of shape [None image_size image_size num_channels]. 
## Hence we reshape.
 
x_batch = images.reshape(1, image_size,image_size,num_channels)
 
frozen_graph="./model/mobilenet_v1.pb"
with tf.gfile.GFile(frozen_graph, "rb") as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
 
with tf.Graph().as_default() as graph:
      tf.import_graph_def(graph_def,
                          input_map=None,
                          return_elements=None,
                          name=""
      )
## NOW the complete graph with values has been restored

#all_nodes = [n for n in tf.get_default_graph().as_graph_def().node]
all_nodes = [n for n in graph.as_graph_def().node]
all_ops = graph.get_operations()
all_tensors = [tensor for op in graph.get_operations() for tensor in op.values()]
all_placeholders = [placeholder for op in graph.get_operations() if op.type=='Placeholder' for placeholder in op.values()]
y_pred=all_tensors[-1]
y_pred=graph.get_tensor_by_name(all_nodes[-1].name+':0')
y_pred = graph.get_tensor_by_name("MobilenetV1/Logits/SpatialSqueeze:0")
## Let's feed the images to the input placeholders
x= graph.get_tensor_by_name("input:0")
##y_test_images = np.zeros((1, 2))
sess= tf.Session(graph=graph)
### Creating the feed_dict that is required to be fed to calculate y_pred 
feed_dict_testing = {x: x_batch}
result=sess.run(y_pred, feed_dict=feed_dict_testing)
print(result)

