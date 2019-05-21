
## Results on Downsampled (32pixel*32pixel) Imagenet
The table prvoides the models and results of various models on downsampled (32*32) Imagenet datasets. 
Learning rate =0.1 and will be divided by 10 every 30 epochs. Total 100 epochs.
Using SGD optimizer, momentum=0.9, weight_decay=5e-4.
Loss is CrossEntropyLoss.
Batch-size=512.

Model | Parameters| Flops | Downsampled ImageNet (Top1) | Downsampled ImageNet (Top5)
-------|:-------:|:--------:|:--------:|:--------:|
[PreActResNet18](https://drive.google.com/open?id=11pJX1ValkQLp1unMp1nml-2Azo8WSDpE) |- |- |53.632%|77.200%
[PreActResNet50](https://drive.google.com/open?id=1koSAcWUiYgxKqPw4U6fMdOl6Kp44bNIb) |- |- |60.434%|82.476%
[PreActResNet101](https://drive.google.com/open?id=18Zx1UHfxGExXkRE9RxcdQKaiHm1SeoRH) |- |- |61.860%|83.522%
[SEResNet18](https://drive.google.com/open?id=1c5cULBL9NT17buv0oETmUaLvE1volIV4) |- |- |53.526%|77.424%
[SEResNet50](https://drive.google.com/open?id=1m2Cz7PfxRh5nMabAL4FB0WdL6r1vYIYZ) |- |- |59.414%|81.716%
[SEResNet101](https://drive.google.com/open?id=1zwBG8PZOhxkjcYdGDyPTZW5upXLgkmko) |- |- |60.928%|82.862%
[PSEResNet18](https://drive.google.com/open?id=1_QsG2t2i7HXmzKr7eJonrjeNU2NQlgeu) |- |- |53.754%|77.412%
[PSEResNet50](https://drive.google.com/open?id=1Jum6qH1bf8KWnqVwoURLyoI86s73dxMj) |- |- |59.316%|81.652%
[PSEResNet101](https://drive.google.com/open?id=1r6-VQp4Hurx6sa9p28NLLmhHqDTSUYac) |- |- |60.922%|82.774%
[CPSEResNet18](https://drive.google.com/open?id=1vl-MLRf7payUj7pCA_A-Q7BO35cLbOVh) |- |- |53.668%|77.610%
[CPSEResNet50](https://drive.google.com/open?id=1wEuueaBzJCoNCG92WI5ld3CfnFGGXAoB) |- |- |59.132%|81.648%
[CPSEResNet101](https://drive.google.com/open?id=1yUwPoh4vfkZVejCnIVvdd6SBTMnO6h0J) |- |- |60.826%|82.628%
[SPPSEResNet18](https://drive.google.com/open?id=1qRT3s8bNoMBtYe5EVQjYeqkj76XF8SSr) |- |- |54.502%|78.184%
[SPPSEResNet50](https://drive.google.com/open?id=1MBQZ1LUDYb7AHP7f7okTt6gHjquDoqMe) |- |-|61.304%|83.396%
[SPPSEResNet101](https://drive.google.com/open?id=1CQMdIL3eqNel7X8J7EXPZBdB0rIJ7qni) |- |- | 62.672%|84.122%
[PSPPSEResNet18](https://drive.google.com/open?id=19FQVn33fdptmPfbxViS7MZw1pMDksTI5) |- |- |54.236%|78.026
[PSPPSEResNet50](https://drive.google.com/open?id=1M4Kf2Y6dooX-T9JXWMTowIIdN6EeVEWe) |- |- |61.364%|83.514%
[PSPPSEResNet101](https://drive.google.com/open?id=14_JMSE0Wq8etwOTtqF72-L1S3R1G8XJZ) |- |- |62.808%|84.382%
[CPSPPSEResNet18](https://drive.google.com/open?id=1z1mgRmTWLb1NZ6ojWd_gvO_kwoWsro1-) |- |- |54.644%|78.430%
[CPSPPSEResNet50](https://drive.google.com/open?id=1beaQwpv5UwGPK8VO22eKUVVsWx5Dlcd6) |- |- |61.476%|83.502%
[CPSPPSEResNet101]() |- |- |-|-


For a better understanding, we reschedule the table as follows (the performance is Top1 accuracy):

Model | 18-Layer| 50-Layer | 101-Layer | 
-------|:-------:|:--------:|:--------:|
PreActResNet    |53.632%|60.434%|61.860%
SEResNet        |53.526%|59.414%|60.928%
PSEResNet       |53.754%|59.316%|60.922%
CPSEResNet      |53.668%|59.132%|60.826%
SPPSEResNet     |54.502%|61.304%|62.672%
PSPPSEResNet    |54.236%|61.364%|62.808%
CPSPPSEResNet   |54.644%|61.476%|-

