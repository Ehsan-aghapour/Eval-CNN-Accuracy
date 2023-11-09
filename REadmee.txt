
In Folder Imagenet there are image net test images which are 50000 labeled images for 1000 classes. 

In past I evaluate the accuracy of fully precision and fully quantized of 5 CNNs, for this purpose in corresponding dir(keras or caffe ..) I run the accuracy.py of it, and it generates the outputs for 50000 images (in original it should be index 0 to 999 for each image but it uses the labesls.txt to convert it to a format of n1938394 (something like this), then this .csv file of 50000 predicts need to moved in ../Evaluation dir that has a simple script that calculate top1 and top5 accurace (actually that predicts has 5 predicts for each image not one); I think I used NPU for quantized evaluation of the CNNs;


For partially quantization I created a env_Partial_Q in cPU server(with 128 cores) and write the multithread version, now the accuracy_mulithread.py itself run and evaluate a model (that could be .h5 or paritally quantized of tensoflowlite); and in Quantization dir I put the script that paritally quantize and use this accuracy_multithread for measuring the accuracy. In Quantiztion/Q.py if you set the test_partial_q (above main function) it just quantize the model for you desired main layers that ask you enter in input; then it prints the command that you can use to copy it from server to loal; then you can see if it is ok in for example Netron.net;

it has run_0 which explore fully precision and fully quantized
run_2 which selects two of n layers (and maybe also 1 of n)
run_3 which selects 3 form n layers
and run_montecarlo which randomly quantize layers (in comments of its generator you can see the other mentioned cases (select 1, 2 and 3 of n)

each run has a generator that generates the suspected layers list based on STATSRESULT file that is analyaze report of quantizing layers.

The environment is simply installed with python 3.10.9 and keras 2.14 and tensorflow 2.14 (however I just install last versions of tensorflwo and keras and its works! please becarefulluy to not installl the tensorflow night (tf-night) because has version inconsistency with other packages.
