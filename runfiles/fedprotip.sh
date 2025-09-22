# CIFAR100 
python main.py --config configs/cifar100/cifar100_fedprotip.json 

# DomainNet 
python main.py --config configs/domainnet/dom_fedprotip.json

# ImageNetR
# 5-split 
python main.py --config configs/imagenetr/imagenetr_fedprotip.json --n_tasks 5 --increment 40 

# 10-split
python main.py --config configs/imagenetr/imagenetr_fedprotip.json --n_tasks 10 --increment 20

# 20-split 
python main.py --config configs/imagenetr/imagenetr_fedprotip.json --n_tasks 20 --increment 10 