# CV Paper List
It's a list for state-of-art computer version papers that I have read.<br>
Collected by Xu Ma.  <xuma@my.unt.edu>



## Classification
+ [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
	<br><sub>almost the start of DNN, also the start of group channels.</sub>
+ [VGG](https://arxiv.org/pdf/1409.1556.pdf "VGG16")
	<br><sub>Block design: avoid overfitting to a specific dataset, improve generalization.</sub>
+ [Highway-Net](https://arxiv.org/pdf/1507.06228.pdf)
	<br><sub>a prelude of ResNet, which can be presented as a*shortcut+b*residual</sub>
+ [ResNet](https://arxiv.org/pdf/1512.03385.pdf)
	<br><sub> Shortcut,skip connection, mitigate the degradation problem.</sub>
+ [Identity ResNet](https://arxiv.org/pdf/1603.05027.pdf)
	<br><sub>Shows the importance of a clean shortcut.</sub>
+ [ResNet Behaves like ensemble](https://arxiv.org/pdf/1605.06431.pdf)
	<br><sub>Unraveled view of residual networks shows that it performs like ensemble ways.</sub>
+ [ResNext](http://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf)
	<br><sub>Create another dimension, named cardinality. In my opinion, it is also a study of channel-wise methods.</sub>
+ [Network in Network](https://arxiv.org/pdf/1312.4400.pdf)
	<br><sub>-</sub>
+ [DenseNet](https://arxiv.org/pdf/1608.06993.pdf)
	<br><sub>Densely connection, connect each layers. It alleviate the vanishing-gradient problem. So, lots of papers indicates that shortcut can alleviate the gradient problem.</sub>
+ [Deformable CNN](https://arxiv.org/pdf/1703.06211.pdf)
	<br><sub>Kernel(Filter) size is not required as fixed size. In this paper, it solves this problem.</sub>
+ [InceptionV4](https://arxiv.org/pdf/1602.07261.pdf)
	<br><sub>-</sub>
+ [MobileNet](https://arxiv.org/pdf/1704.04861.pdf)
	<br><sub>depthwise convolution + pointwise convolution(1X1). For further reduction: Width Multiplier + Resolution Multiplier</sub>
+ [Shufflenet](https://arxiv.org/pdf/1707.01083.pdf)
	<br><sub>-</sub>
+ [SE-Net](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf)
	<br><sub>An independent path, global pooling to 1*1*c, which can be considered as a weight of channels.</sub>

## Object Detection

+ [R-CNN](https://arxiv.org/pdf/1311.2524.pdf)
	<br><sub>-</sub>
+ [Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf)
	<br><sub>-</sub>
+ [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf)
	<br><sub>-</sub>
+ [YOLO V1](https://arxiv.org/pdf/1506.02640.pdf)
	<br><sub>-</sub>
+ [YOLO V2](https://arxiv.org/pdf/1612.08242.pdf)
	<br><sub>-</sub>
+ [YOLO V3](https://arxiv.org/pdf/1804.02767.pdf)
	<br><sub>-</sub>
+ [R-FCN](https://arxiv.org/pdf/1605.06409.pdf)
	<br><sub>-</sub>
+ [SSD](https://arxiv.org/pdf/1512.02325.pdf)
	<br><sub>-</sub>
+ [DSSD](https://arxiv.org/pdf/1701.06659.pdf)
	<br><sub>-</sub>
+ [FPN](https://arxiv.org/pdf/1612.03144.pdf)
	<br><sub>-</sub>
+ [Non-local Neural Networks](https://arxiv.org/pdf/1711.07971.pdf)
	<br><sub>-</sub>
+ [Object Detection Networks on Convolutional Feature Maps](https://arxiv.org/pdf/1504.06066.pdf)
	<br><sub>-</sub>
+ [Spatial transformer networks](https://arxiv.org/pdf/1506.02025.pdf)
	<br><sub>-</sub>
+ [Learning Deep Features for Discriminative Localization](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf)
	<br><sub>-</sub>
+ [RetinaNet(Focal Loss for Dense Object Detection)](https://arxiv.org/pdf/1708.02002.pdf)
	<br><sub>-</sub>
+ [dilated convolutions](https://arxiv.org/pdf/1511.07122.pdf)
	<br><sub>-</sub>
+ [SNIP](https://arxiv.org/pdf/1711.08189.pdf)
	<br><sub>-</sub>
+ [Inside-outside net](https://arxiv.org/pdf/1512.04143.pdf)
	<br><sub>-</sub>
+ [Generalized Intersection over Union](https://arxiv.org/pdf/1902.09630.pdf)
	<br><sub>-</sub>
+ [Cascade R-CNN](https://arxiv.org/pdf/1712.00726.pdf)
	<br><sub>-</sub>

## Segementation (Semantic/Instance)
+ [FCN](https://arxiv.org/pdf/1411.4038.pdf)
	<br><sub>-</sub>
+ [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf)
	<br><sub>-</sub>
+ [U-Net](https://arxiv.org/pdf/1505.04597.pdf)
	<br><sub>-</sub>




## Tricks
+ [SPP](https://arxiv.org/pdf/1406.4729.pdf)
	<br><sub>-</sub>
+ [Dropout](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
	<br><sub>-</sub>
+ [Bag of Freebies for Training Object Detection Neural Networks](https://arxiv.org/pdf/1902.04103.pdf)
	<br><sub>-</sub>
+ [BN](https://arxiv.org/pdf/1502.03167.pdf)
	<br><sub>-</sub>
+ [Visualizing and understanding convolutional neural networks](https://arxiv.org/pdf/1311.2901.pdf)
	<br><sub>-</sub>

