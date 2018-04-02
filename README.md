# Federated Learning

This is partly the reproduction of the paper of [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)   
Only experiments on MNIST (both IID and non-IID) is produced by far.

Note: The scripts will be slow without the implementation of parallel computing. 

## Run

The MLP and CNN models are produced by:
> python [main_nn.py](main_nn.py)

The testing accuracy of MLP: 92.14% (10 epochs training) with the learning rate of 0.01.
The testing accuracy of CNN: 98.37% (10 epochs training) with the learning rate of 0.01.

Federated learning with MLP and CNN is produced by:
> python [main_fed.py](main_fed.py)

See the arguments in [options.py](optifons.py). 

For example:
> python main_fed.py --dataset mnist --model cnn --epochs 50 --gpu 0 


## Results
### MNIST
Results are shown in Table 1 and Table 2, with the parameters C=0.1, B=10, E=5.

Table 1. results of 10 epochs training with the learning rate of 0.01

| Model     | Acc. of IID | Acc. of Non-IID|
| -----     | -----       | ----           |
| FedAVG-MLP|  85.66%     | 72.08%         |
| FedAVG-CNN|  95.00%     | 74.92%         |

Table 2. results of 50 epochs training with the learning rate of 0.01

| Model     | Acc. of IID | Acc. of Non-IID|
| -----     | -----       | ----           |
| FedAVG-MLP| 84.42%      | 88.17%         |
| FedAVG-CNN| 98.17%      | 89.92%         |


## Requirements
python 3.6

pytorch 0.3