import scipy.io
import os
import sys

mat_name='meta.mat'
ground_truth_name='ILSVRC2012_validation_ground_truth.txt'
mat = scipy.io.loadmat(mat_name)

# observe keys by mat.keys()
key='synsets'

synsets=mat[key]
print(synsets.dtype)
for syn in synsets:
    print(f'ILSVRC2012_ID:{syn[0][0][0]}  WordNetID(WNID):{syn[0][1]}  words:{syn[0][2]}  gloss:{syn[0][3]}')


labels_file=open(ground_truth_name)
label_lines=labels_file.readlines()
labels=[int(l) for l in label_lines]

l_file=open('ground_labels.txt','w')

labels_data=[]
for label in labels:
    l_file.write(f'{synsets[label-1][0][1][0]} {synsets[label-1][0][2][0]}\n')
    t=[]
    t.append(label)
    t.append(synsets[label-1][0][1][0])
    t.append(synsets[label-1][0][2][0])
    labels_data.append(t)

