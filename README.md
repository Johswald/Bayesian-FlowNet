# Bayesian FlowNetS in Tensorflow

Tensorflow implementation of optical flow predicting [FlowNetS](https://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15/flownet.pdf) 
by [Alexey Dosovitskiy](http://lmb.informatik.uni-freiburg.de/people/dosovits/) et al.

The network can be equiped with dropout layers to produce confidence images through MC dropout after training, 
as introduced [here](https://arxiv.org/abs/1506.02158). 
The positions of dropout layers are very similiar to other encoder-decoder architectures such as [Bayesian SegNet](https://arxiv.org/pdf/1511.02680) or [Deep Depth From Focus](https://arxiv.org/abs/1704.01085).

The confidence images are then used to improve results (with limited success) through post processing through the [Fast Bilateral Solver](https://arxiv.org/pdf/1511.03296.pdf).


## Training

The architecture is trained on the [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html) dataset, please feel free to provide a Tensorflow reader of the used .ppm images. To enable fast reading here, the images were first transformed to .jpg.

To get similiar results as reported below, just start training by

    python train.py --datadir /path/to/FlyingChairs/ 
  
where in the folder FlyingChairs/ you have simply have the ~27k numbered -img1.jpg, -img2.jpg, -.flo 
training files (note .jpg). To incorporate dropout layers, simply 

    python train.py --datadir /path/to/FlyingChairs/ --dropout True

Check standart hyperparameters in train.py, note that the results are sensitive to the "amount" of data augmentation you use. 
Training loss looks somthing like: 

<img src="https://github.com/Johswald/Bayesian-FlowNet/blob/master/images/l1.png" width="200">

## Data Augmentation

Heavy data augmentation is used to improve generalization / performance.  
Check flownet.py for 

- chromatic augmentation 
- geometric augmentation (rotation + translation)

Please note that when we flip, crop, rotate and scale, we must be carefull and change 
the flow directions (u,v) according to the change of pixels (x, y). 

## Loss

L1 loss is calculated multiple times while decoding, we must "downsample" the original flow which is done 
through a weighted average in the original caffe version. Here, simple bilinear interpolation is used which could have negative effects on performance. 

## Evaluation 

There are evaluation scripts for FlyingChairs, Sintel (clean / final) and Kitti datasets provided, e.g.

    python eval_var_flownet_s.py --dropout True/False

They either evaluate 
scaling the weights to fixed weights magnitudes after dropout training, parameters: 

    --dropout True / --is_training False

or

by loading one test example and creating a minibatch (of size = FLAGS.batchsize) of the same image  
and average results of the minibatch, parameters: 

    --dropout True / --is_training True

Note that is_training is falsly named due to simplicity.
Through variances of the minibatches results on the same image but inference on "different" models, 
confidence images can be created. 
Some images, groundtruth, results, confidence images and error images:



