## Results on CIFAR100
The table prvoides the models and results of various models on CIFAR100. 
Learning rate =0.1 and will be divided by 10 every 70 epochs. Total 300 epochs.
Using SGD optimizer, momentum=0.9, weight_decay=5e-4.
Loss is CrossEntropyLoss.
Batch-size=512.

Model | Parameters| Flops | CIFAR-100 | 
-------|:-------:|:--------:|:--------:|
[PreActResNet18](https://drive.google.com/open?id=1w2VGpFPDuS9NzcfcGfPUXoEdXwVftFep) |- |- |74.91%
[PreActResNet50](https://drive.google.com/open?id=1Nz_JmzLxuzefGzekBRoCutDIeRgaKWMY) |- |- |77.39%
[PreActResNet101](https://drive.google.com/open?id=1gZoIQhJCzSMhN9b6OeoLL_lyxgU5vCVT) |- |- |77.74%
[SEResNet18](https://drive.google.com/open?id=17Ynt2pLrbew-n2Wu3P8coZ1vTUiV8h3I) |- |- |75.19%
[SEResNet50]() |- |- |-
[SEResNet101]() |- |- |-
[PSEResNet18](https://drive.google.com/open?id=1ZHYAyjiVsBtpCe7pDp3Ip204UYDpe_aR) |- |- |74.97%
[PSEResNet50]() |- |- |-
[PSEResNet101]() |- |- |-
[CPSEResNet18](https://drive.google.com/open?id=12Hne8epBFV2YjakHP43PwYSYizdHlG0D) |- |- |75.25%
[CPSEResNet50](https://drive.google.com/open?id=1axp5bjRTkmkxRd3CGRTP_WwBOcdh74GM) |- |- |77.43%
[CPSEResNet101]() |- |- |-
[SPPSEResNet18](https://drive.google.com/open?id=1EYcqDd70KHLKC2v_DaZ35qW1SLVzwaqN) |- |- |75.41%
[SPPSEResNet50]() |- |- |-
[SPPSEResNet101]() |- |- |-
[PSPPSEResNet18](https://drive.google.com/open?id=1h-d4b1qaGgzxu8_yPlwrVu-BIN9ZUbNo) |- |- |75.01%
[PSPPSEResNet50]() |- |- |-
[PSPPSEResNet101]() |- |- |-
[CPSPPSEResNet18](https://drive.google.com/open?id=1G1vPvLYFCTCq7nE4TQFTiwIthKFE9yso) |- |- |75.56%
[CPSPPSEResNet50]() |- |- |-
[CPSPPSEResNet101]() |- |- |failed
