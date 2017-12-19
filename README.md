# Bayesian FlowNetS in Tensorflow

Tensorflow implementation of optical flow predicting [FlowNetS](https://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15/flownet.pdf) 
by [Alexey Dosovitskiy](http://lmb.informatik.uni-freiburg.de/people/dosovits/) et al.

The network can be equipped with dropout layers to produce confidence images through MC dropout after training, 
as introduced [here](https://arxiv.org/abs/1506.02158). 
The positions of dropout layers are very similar to other encoder-decoder architectures such as [Bayesian SegNet](https://arxiv.org/pdf/1511.02680) or [Deep Depth From Focus](https://arxiv.org/abs/1704.01085).

The confidence images are then used to improve results (with limited success) through post-processing through the [Fast Bilateral Solver](https://arxiv.org/pdf/1511.03296.pdf).


## Training

The architecture is trained on the [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html) dataset, please feel free to provide a Tensorflow reader of the used .ppm images. To enable fast reading here, the images were first transformed to .jpg.

To get similar results as reported below, just start training by

    python train.py --datadir /path/to/FlyingChairs/ 
  
and in the folder FlyingChairs/ you have simply have the ~27k numbered -img1.jpg, -img2.jpg, -.flo 
training files (note .jpg). To incorporate dropout layers, simply 

    python train.py --datadir /path/to/FlyingChairs/ --dropout True

Check standard hyperparameters in train.py, note that the results are sensitive to the "amount" of data augmentation you use. 
Training loss looks somthing like: 

<img src="https://github.com/Johswald/Bayesian-FlowNet/blob/master/images/l1.png" width="400">

## Data Augmentation

Heavy data augmentation is used to improve generalization/ performance.  
Check flownet.py for 

- chromatic augmentation 
- geometric augmentation (rotation + translation)

Please note that when we flip, crop, rotate and scale, we must be careful and change the flow directions (u,v) according to the change of pixels (x, y). 

## Loss

L1 loss is calculated multiple times while decoding, we must "downsample" the original flow which is done through a weighted average in the original caffe version. Here, simple bilinear interpolation is used which could have negative effects on performance. 

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
Note that is_training is falsely named due to simplicity. Through variances of the minibatches results on the same image but inference on "different" models, confidence images can be created. Evaluation throughout training on FlyingCharis test set (pink) ad well as Sintel Clean (orange), Sintel Final (gray) and Kitti (blue) training sets.

<img src="https://github.com/Johswald/Bayesian-FlowNet/blob/master/images/chairs_epe.png" width="300"><img src="https://github.com/Johswald/Bayesian-FlowNet/blob/master/images/Kitti_Sintel_EPE.png" width="300">
## Evaluation 
Two training images as well as groundtruth, flow estimation, confidence and error images:

<img src="https://github.com/Johswald/Bayesian-FlowNet/blob/master/images/img_0214.png" width="400"><img src="https://github.com/Johswald/Bayesian-FlowNet/blob/master/images/img_0384.png" width="400">
Groundtruth images:

<img src="https://github.com/Johswald/Bayesian-FlowNet/blob/master/images/gt_0214.png" width="400"><img src="https://github.com/Johswald/Bayesian-FlowNet/blob/master/images/gt_0384.png" width="400">

Predicted flow images:
<img src="https://github.com/Johswald/Bayesian-FlowNet/blob/master/images/n_flow_214_better.png" width="400"><img src="https://github.com/Johswald/Bayesian-FlowNet/blob/master/images/n_flow_384_better.png" width="400">

Confidence images:

<img src="https://github.com/Johswald/Bayesian-FlowNet/blob/master/images/confidence_0214.png" width="400"><img src="https://github.com/Johswald/Bayesian-FlowNet/blob/master/images/confidence_0384.png" width="400">

Error images (note similiarities/differences to confidence images):

<img src="https://github.com/Johswald/Bayesian-FlowNet/blob/master/images/error_213.png" width="400"><img src="https://github.com/Johswald/Bayesian-FlowNet/blob/master/images/error_383.png" width="400">
