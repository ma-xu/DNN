## Results on CIFAR100
The table prvoides the models and results of various models on CIFAR100. 
Learning rate =0.1 and will be divided by 10 every 70 epochs. Total 300 epochs.
Using SGD optimizer, momentum=0.9, weight_decay=5e-4.
Loss is CrossEntropyLoss.
Batch-size=512.

Model | Parameters| Flops | CIFAR-100 | 
-------|:-------:|:--------:|:--------:|
[PreActResNet18](https://drive.google.com/open?id=1w2VGpFPDuS9NzcfcGfPUXoEdXwVftFep) |- |- |74.910%
[PreActResNet50]() |- |- |-
[PreActResNet101]() |- |- |-
[SEResNet18]() |- |- |-
[SEResNet50]() |- |- |-
[SEResNet101]() |- |- |-
[PSEResNet18]() |- |- |-
[PSEResNet50]() |- |- |-
[PSEResNet101]() |- |- |-
[CPSEResNet18](https://drive.google.com/open?id=12Hne8epBFV2YjakHP43PwYSYizdHlG0D) |- |- |75.250%
[CPSEResNet50]() |- |- |-
[CPSEResNet101]() |- |- |-
[SPPSEResNet18]() |- |- |-
[SPPSEResNet50]() |- |- |-
[SPPSEResNet101]() |- |- |-
[PSPPSEResNet18]() |- |- |-
[PSPPSEResNet50]() |- |- |-
[PSPPSEResNet101]() |- |- |-
[CPSPPSEResNet18]() |- |- |-
[CPSPPSEResNet50]() |- |- |-
[CPSPPSEResNet101]() |- |- |-
