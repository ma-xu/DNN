# Details in DNN

+ __Global average pooling__<br>
```
presented in "Network in Network". Less parameters. Using GlobalAveragePooling replace FC.
Also, another potential advantage is that FC will destory spatial information,
may be not important for classification, but actually important for localization.
(refer to "Learning Deep Features for Discriminative Localization", CVPR2016, MIT)
```
<br>

+ __Why pooling, step size greater than 1, filters increase?__<br>
```
(Namely, feature maps size decrease, filters number increase)
The question can be understood from  the following perspective:
However, having too many filters for a single concept imposes extra burden on the next layer,
which needs to consider all combinations of variations from the previous layer 
[Piecewise linear multilayer perceptrons and dropout]. 
As in CNN, filters from higher layers map to larger regions in the original input.
It generates a higher level concept by combining the lower level concepts from the layer below.
presented in [Network in Network, page 2.]
```
<br>

+ __Why two 3X3 kernel can replace one 5X5 kernel?__<br>
```
The receptive field is the same.
```


+ __Less parameters usually means less overfitting.__ 

+ __Local receptive field limitation__
```
As we know, convolutional operations process one local field (kernel size) each time. This leads us could not take all whole image structure into consideration. Although the deeper layers always have bigger receptive field and has a better semantic representation， the shallower layers could not take such an advantage. Especially for object detection, instance segmentation (and some more related CV taskes), taking this global relationship into consideration is important due to the competition between resolution and semantics (explain in SSD, FPN, for small object). 
Hence, if the global (or weighted neighbour) fields could benefit for image classification or object detection?
(Inspired by the abstract of Non-local Neural Networks)

```
