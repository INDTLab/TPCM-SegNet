# TPCM-SegNet

### Prepare Dataset
Please download Synapse datasets from these links. And the text information is the class name of each image in a txt file. You may generate it yourself based on the image.

Synapse

        https://github.com/Beckschen/TransUNet

### Train
        python train.py --model TPCMSegNet --cuda_id 0

### Evaluate
        python test.py --model TPCMSegNet --cuda_id 0


