# Details in DNN

+ Gobal average pooling<br>
presented in "Network in Network". Less parameters. Using GlobalAveragePooling replace FC.
Also, another potential advantage is that FC will destory spatial information, may be not important for classification, but actually important for localization.(refer to "Learning Deep Features for Discriminative Localization", CVPR2016, MIT)
<br>

+ Why pooling, step size greater than 1, filters increase?<br>
(Namely, feature maps size decrease, filters number increase)<br>
For the question, it can be understood from this view:<br>
However, having too many filters for a single concept imposes extra burden on the next layer, which needs to consider all combinations of variations from the previous layer [Piecewise linear multilayer perceptrons and dropout]. As in CNN, filters from higher layers map to larger regions in the original input. It generates a higher level concept by combining the lower level concepts from the layer below.<br>
presented in [Network in Network, page 2.]


