# FedProTIP: Gradient Projection-Based Federated Continual Learning Aided by Task Identity Prediction

### Requirements 
```
pip install -r requirements.txt
```

### 10-split CIFAR-100
```
python main.py --config configs/cifar100/fedprotip.json 
```

### 6-split DomainNet 
```
python main.py --config configs/domainnet/dom_fedprotip.json
```

### ImageNetR (5, 10, 20-split)
* 5-split 
```
python main.py --config configs/imagenetr/imagenet-r_fedprotip.json --n_tasks 5 --increment 40 
```

* 10-split
```
python main.py --config configs/imagenetr/imagenet-r_fedprotip.json --n_tasks 10 --increment 20
```

* 20-split 
```
python main.py --config configs/imagenetr/imagenet-r_fedprotip.json --n_tasks 20 --increment 10 
```

### Code references 
* [LANDER](https://github.com/tmtuan1307/LANDER/tree/main)
* [GPM](https://github.com/sahagobinda/GPM/tree/main) 