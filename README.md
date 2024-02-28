# DIRK(-REF)

This is the implementation of our AAAI'24 paper (Distilling Reliable Knowledge for Instance-dependent Partial Label Learning).

Requirements: 
Python 3.8.12, 
numpy 1.22.3, 
torch 1.12.1,
torchvision 0.13.1.

You need to:

1. Download FMNIST/KMNIST/CIFAR-10/CIFAR-100 datasets into './data/'.
2. Download model weights from [Google Driver](https://drive.google.com/drive/folders/1E3R7kO8VC6TKQCste_RDjkybqcNQyvUL?usp=drive_link) into './partial_models/weights'
3. For the method **DIRK** Run the following demos:

```sh
python -u main.py --dataset fmnist --arch resnet18 --epochs 500 --batch_size 64 --rate 1.0  --weight 0.0 --seed 3407
python -u main.py --dataset kmnist --arch resnet18 --epochs 500 --batch_size 64 --rate 0.9  --weight 0.0 --seed 3407
python -u main.py --dataset cifar10 --arch resnet34 --epochs 500 --batch_size 64 --rate 1.0  --weight 0.0 --seed 3407
python -u main.py --dataset cifar100 --arch resnet34 --epochs 500 --batch_size 64 --rate 0.1  --weight 0.0 --seed 3407
```

4. For the method **DIRK-REF** Run the following demos:

```sh
python -u main.py --dataset fmnist --arch resnet18 --epochs 500 --batch_size 64 --rate 1.0  --weight 1.0 --seed 3407
python -u main.py --dataset kmnist --arch resnet18 --epochs 500 --batch_size 64 --rate 0.9  --weight 1.0 --seed 3407
python -u main.py --dataset cifar10 --arch resnet34 --epochs 500 --batch_size 64 --rate 1.0  --weight 1.0 --seed 3407
python -u main.py --dataset cifar100 --arch resnet34 --epochs 500 --batch_size 64 --rate 0.1  --weight 1.0 --seed 3407
```



If you have any further questions, please feel free to send an e-mail to: dongdongwu@seu.edu.cn. Have fun!
