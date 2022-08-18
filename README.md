# On Robust Learning from Noisy Labels: A Permutation Layer Approach

This repository is the official implementation of [On Robust Learning from Noisy Labels: A Permutation Layer Approach](). 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Run Symmetric Noise CIFAR-10
```train
python main.py --dataset=cifar10 --early_stopping=-1 --epochs=120 --gamma=0.1 --init_max_prob=0.35 --logits_softmax_mode=log_perm_softmax --lr_scheduler=step_lr --milestones=80,100 --model_name=resnet34 --momentum=0.9 --networks_lr=0.02 --networks_optim=sgd --noise=0.2 --permutation_lr=1.5 --weight_decay=0.0005 --noise_mode=sym
```

## Run Asymmetric Noise CIFAR-10
```train
python main.py --dataset=cifar10 --early_stopping=-1 --epochs=120 --gamma=0.1 --init_max_prob=0.35 --logits_softmax_mode=log_perm_softmax --lr_scheduler=step_lr --milestones=80,100 --model_name=resnet34 --momentum=0.9 --networks_lr=0.02 --networks_optim=sgd --noise=0.2 --permutation_lr=1.5 --weight_decay=0.0005 --noise_mode=asym
```

## Run Symmetric Noise CIFAR-100
```train
python main.py --dataset=cifar100 --early_stopping=-1 --epochs=120 --gamma=0.1 --init_max_prob=0.225 --logits_softmax_mode=log_perm_softmax --lr_scheduler=step_lr --milestones=100 --model_name=resnet34 --momentum=0.9 --networks_lr=0.02 --networks_optim=sgd --noise=0.2 --permutation_lr=3 --weight_decay=0.001 --noise_mode=sym
```
## Run Asymmetric Noise CIFAR-100
```train
python main.py --dataset=cifar100 --early_stopping=-1 --epochs=120 --gamma=0.1 --init_max_prob=0.225 --logits_softmax_mode=log_perm_softmax --lr_scheduler=step_lr --milestones=100 --model_name=resnet34 --momentum=0.9 --networks_lr=0.02 --networks_optim=sgd --noise=0.2 --permutation_lr=3 --weight_decay=0.001 --noise_mode=asym
```
