
## Results on Downsampled (32pixel*32pixel) Imagenet
The table prvoides the models and results of various models on downsampled (32*32) Imagenet datasets. 
Learning rate =0.1 and will be divided by 10 every 30 epochs. Total 100 epochs.
Using SGD optimizer, momentum=0.9, weight_decay=5e-4.
Loss is CrossEntropyLoss.
Batch-size=512.

Model | Parameters| Flops | Downsampled ImageNet (Top1) | Downsampled ImageNet (Top5)
-------|:-------:|:--------:|:--------:|:--------:|
[PreActResNet18](https://drive.google.com/open?id=11pJX1ValkQLp1unMp1nml-2Azo8WSDpE) |- |- |53.632%|77.200%
[PreActResNet50]() |- |- |-|-
[PreActResNet101]() |- |- |-|-
[SEResNet18]() |- |- |-|-
[SEResNet50]() |- |- |-|-
[SEResNet101]() |- |- |-|-
[PSEResNet18](https://drive.google.com/open?id=1_QsG2t2i7HXmzKr7eJonrjeNU2NQlgeu) |- |- |53.754%|77.412%
[PSEResNet50]() |- |- |-|-
[PSEResNet101]() |- |- |-|-
[CPSEResNet18]() |- |- |-|-
[CPSEResNet50]() |- |- |-|-
[CPSEResNet101]() |- |- |-|-
[SPPSEResNet18]() |- |- |-|-
[SPPSEResNet50]() |- |-|-|-
[SPPSEResNet101]() |- |- |-|-
[PSPPSEResNet18]() |- |- |-|-
[PSPPSEResNet50]() |- |- |-|-
[PSPPSEResNet101]() |- |- |-|-
[CPSPPSEResNet18]() |- |- |-|-
[CPSPPSEResNet50]() |- |- |-|-
[CPSPPSEResNet101]() |- |- |-|-


For a better understanding, we reschedule the table as follows (the performance is Top1 accuracy):

Model | 18-Layer| 50-Layer | 101-Layer | 
-------|:-------:|:--------:|:--------:|
PreActResNet    |53.632%|-|-
SEResNet        |-|-|-
PSEResNet       |53.754%|-|-
CPSEResNet      |-|-|-
SPPSEResNet     |-|-|-
PSPPSEResNet    |-|-|-
CPSPPSEResNet   |-|-|-

