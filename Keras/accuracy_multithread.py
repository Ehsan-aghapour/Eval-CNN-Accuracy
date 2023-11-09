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

# !pip install numpy
# !pip install tensorflow keras
# !pip install Pillow

#make sure that it is the right environment
import sys
print(sys.executable)

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
BATCH_SIZE = 1000  # Adjust as needed
N=50000
NUM_CLASSES = 1000  # Number of ImageNet classes


# Load model
def _load_model(MODEL_NAME):
    #model = load_model(models_dir+model_name)
    global model
    model = load_model(MODEL_NAME)



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



def run_predict(batch_images):
    batch_preds = model.predict(batch_images)
    return batch_preds

# Evaluation function to be used in each process
def evaluate_batch(batch_images, batch_labels):
    batch_preds = run_predict(batch_images)
    top1_correct = np.sum(np.argmax(batch_preds, axis=1) == batch_labels)
    top5_correct = np.sum([label in pred for label, pred in zip(batch_labels, np.argsort(batch_preds, axis=1)[:, -5:])])
    return top1_correct, top5_correct


# Thread worker function
def thread_worker(image_paths, labels):
    #print(f'running for images {len(image_paths)}')
    batch_images = np.vstack([preprocess_image(img_path) for img_path in image_paths])
    top1 , top5 = evaluate_batch(batch_images, labels)
    return [top1, top5]



def main(Model_Name=models_dir+model_name):
    _load_model(Model_Name)
    # Gather image paths and labels
    image_paths = [os.path.join(IMAGE_DIR, fname) for fname in sorted(os.listdir(IMAGE_DIR))][:N]

    #labels = to_categorical(true_labels, NUM_CLASSES)
    labels=ground_truth_indices

    # Split into batches
    batches = [(image_paths[i:i + BATCH_SIZE], labels[i:i + BATCH_SIZE]) for i in range(0, len(image_paths), BATCH_SIZE)]


    time1=time.time()
    # Perform multi-threaded evaluation
    top1_correct = top5_correct = total_images = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
        # Submit all the tasks and get back Future objects
        futures = [executor.submit(thread_worker, batch[0], batch[1]) for batch in batches]

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
     

#main()

# -

