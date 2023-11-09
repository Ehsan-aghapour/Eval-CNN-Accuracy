# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Q_Partial
#     language: python
#     name: q_partial
# ---

# !jupytext --set-formats ipynb,py accuracy_multithread.ipynb --sync
# !pip install numpy
# !pip install tensorflow keras
# !pip install Pillow

#make sure that it is the right environment
import sys
print(sys.executable)

from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import tensorflow as tf



# +
import concurrent.futures
import numpy as np
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
import os
from tensorflow.keras.models import load_model
import time




# Constants
model_name='MobileNet/MobileNet.h5'
models_dir='/home/ehsan/Partial_Q/models/'
Proj_DIR="/home/ehsan/Partial_Q/"
IMAGE_DIR = Proj_DIR+'/Imagenet/ILSVRC2012_img_val'
LABEL_FILE = Proj_DIR+'/Evaluation/Ground_labels/ground_labels.txt'
LABEL_MAP=Proj_DIR+'/Keras/labels.txt'
BATCH_SIZE = 100  # Adjust as needed
N=50000
MAX_WORKERS=64
NUM_CLASSES = 1000  # Number of ImageNet classes

_model_input_shape= (224, 224)



def _load_model(model_path,model_input_shape=_model_input_shape):
    #global model,model_format
    # support of tflite model
    if model_path.endswith('.tflite'):
        from tensorflow.lite.python import interpreter as interpreter_wrapper
        model = interpreter_wrapper.Interpreter(model_path=model_path)
        model.allocate_tensors()
        model_format = 'TFLITE'

        #Ehsan input shape correctness
        input_details = model.get_input_details()
        input_shape = input_details[0]['shape']
        input_shape[1] = model_input_shape[0]
        input_shape[2] = model_input_shape[1]
        input_shape[0]=BATCH_SIZE
        model.resize_tensor_input(0, input_shape)
        #print(f'shape of input is: {input_shape}')
        #model.allocate_tensors()
        


    # support of MNN model
    elif model_path.endswith('.mnn'):
        model = MNN.Interpreter(model_path)
        model_format = 'MNN'

    # support of TF 1.x frozen pb model
    elif model_path.endswith('.pb'):
        model = load_graph(model_path)
        model_format = 'PB'

    # support of ONNX model
    elif model_path.endswith('.onnx'):
        model = onnxruntime.InferenceSession(model_path)
        model_format = 'ONNX'

    # normal keras h5 model
    elif model_path.endswith('.h5'):
        #custom_object_dict = get_custom_objects()

        model = load_model(model_path, compile=False)#, custom_objects=custom_object_dict)
        model_format = 'H5'
        K.set_learning_phase(0)
    else:
        raise ValueError('invalid model file')

    return model,model_format

#_load_model("Quantization/cases/(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0).tflite",[224,224])
# +
def predict_tflite(image,model):
    model.allocate_tensors()
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    # check the type of the input tensor
    #if input_details[0]['dtype'] == np.float32:
        #floating_model = True

    
    #height = input_details[0]['shape'][1]
    #width = input_details[0]['shape'][2]
    #model_input_shape = (height, width)
    #model_input_shape = (608,608)
    #print(f'{input_details}\nimage shape:{np.array(image).shape}, model input shape {model_input_shape}')
    #input()
    #image_data = preprocess_image(image, model_input_shape)
    #origin image shape, in (height, width) format
    #image_shape = image.size[::-1]

    #interpreter.set_tensor(input_details[0]['index'], image_data)
    model.set_tensor(input_details[0]['index'], image)
    model.invoke()

    '''prediction = []
    for output_detail in output_details:
        output_data = model.get_tensor(output_detail['index'])
        prediction.append(output_data)
    return np.array(prediction[0]) '''  
    outp=(model.get_tensor(output_details[0]['index'])).copy()
    return outp
        
    #return model.get_tensor(output_details[0]['index'])
    

'''img_paths=['/home/ehsan/Partial_Q/Imagenet/ILSVRC2012_img_val/ILSVRC2012_val_00021428.JPEG',
           '/home/ehsan/Partial_Q/Imagenet/ILSVRC2012_img_val/ILSVRC2012_val_00021428.JPEG',
          '/home/ehsan/Partial_Q/Imagenet/ILSVRC2012_img_val/ILSVRC2012_val_00021429.JPEG']
batch_imgs = np.vstack([preprocess_image(img_path) for img_path in img_paths])
mobile_predict_tflite(batch_imgs)'''
# +
'''# Load model
def _load_model(MODEL_NAME):
    #model = load_model(models_dir+model_name)
    global model
    model = load_model(MODEL_NAME)
'''


# Step 1: Create a mapping from class names to indices based on labels.txt
label_index_map = {}
with open(LABEL_MAP, 'r') as f:
    for index, line in enumerate(f):
        class_name = line.strip().split(' ')[0]
        label_index_map[class_name] = index

# Step 2: Read ground_labels.txt and convert the labels to indices
ground_truth_indices = []
with open(LABEL_FILE, 'r') as f:
    for line in f:
        class_name = line.strip().split(' ')[0]
        if class_name in label_index_map:
            ground_truth_indices.append(label_index_map[class_name])
        else:
            print(f"Label {class_name} not found in label index map.")
            ground_truth_indices.append(None)  # Handle missing labels if necessary

# Now, ground_truth_indices contains the true indices for each image
# Preprocessing function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x



def run_predict(batch_images, model, model_format):
    if model_format=='H5':
        batch_preds = model.predict(batch_images)
    elif model_format=='TFLITE':
        batch_preds = predict_tflite(batch_images, model)
    return batch_preds


# Evaluation function to be used in each process
def evaluate_batch(batch_images, batch_labels, _m, m_form):
    #global batch_preds
    batch_preds = run_predict(batch_images, _m, m_form)
    #print(batch_preds)
    top1_correct = np.sum(np.argmax(batch_preds, axis=1) == batch_labels)
    top5_correct = np.sum([label in pred for label, pred in zip(batch_labels, np.argsort(batch_preds, axis=1)[:, -5:])])
    return top1_correct, top5_correct


# Thread worker function
def thread_worker(image_paths, labels, Model_Name):
    #print(f'running for images {len(image_paths)}')
    _m,m_form=_load_model(Model_Name)
    batch_images = np.vstack([preprocess_image(img_path) for img_path in image_paths])
    top1 , top5 = evaluate_batch(batch_images, labels, _m, m_form)
    del _m
    return [top1, top5]



def main(Model_Name=models_dir+model_name):
    #_load_model(Model_Name)
    # Gather image paths and labels
    image_paths = [os.path.join(IMAGE_DIR, fname) for fname in sorted(os.listdir(IMAGE_DIR))][:N]

    #labels = to_categorical(true_labels, NUM_CLASSES)
    labels=ground_truth_indices

    # Split into batches
    batches = [(image_paths[i:i + BATCH_SIZE], labels[i:i + BATCH_SIZE]) for i in range(0, len(image_paths), BATCH_SIZE)]


    time1=time.time()
    # Perform multi-threaded evaluation
    top1_correct = top5_correct = total_images = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all the tasks and get back Future objects
        futures = [executor.submit(thread_worker, batch[0], batch[1],Model_Name) for batch in batches]

        # Iterate over the completed futures as they complete
        for future in concurrent.futures.as_completed(futures):
            top1, top5 = future.result()  # Unpack the result from the future
            top1_correct += top1
            top5_correct += top5
            total_images += BATCH_SIZE  # Make sure to update the total_images if not all images are used

    time2=time.time()
    t=time2-time1
    print(f"Total time of Evaluation: {t}")
    # Calculate overall accuracies
    overall_top1_accuracy = top1_correct / total_images  # This should be total_images, not N, in case N is not a multiple of BATCH_SIZE
    overall_top5_accuracy = top5_correct / total_images
    print(f"Overall top-1 accuracy: {overall_top1_accuracy * 100:.4f}%")
    print(f"Overall top-5 accuracy: {overall_top5_accuracy * 100:.4f}%")
    return overall_top1_accuracy,overall_top5_accuracy
     

#main()
#main('/home/ehsan/Partial_Q/Keras/Quantization/cases/(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0).tflite')
