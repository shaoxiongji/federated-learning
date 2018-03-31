# Federated Learning

This is partly the reproduction of the paper of [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)   
Only IID data of MNIST is produced by far.

Note: The scripts will be slow without the implementation of parallel computing. 

## Run

The MLP and CNN models are produced by:
> python [main_nn.py](main_nn.py)

The testing accuracy of MLP: 92.14% (10 epochs) with learning rate of 0.01.
The testing accuracy of CNN: 98.37% (10 epochs) with learning rate of 0.01.

Federated learning with MLP and CNN is produced by:
> python [main_fed.py](main_fed.py)

See the arguments in [options.py](optifons.py). 

For example:
> python main_fed.py --dataset mnist --model cnn --epochs 50 --gpu 0 


## Requirements
python 3.6

pytorch 0.3