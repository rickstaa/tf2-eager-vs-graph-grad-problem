# tf2 Eager vs Graph gradients problem

A Simple example repository to show the problems I have with computing gradients of a
Squashed gaussian actor [Haarnoja et al. 2019](https://arxiv.org/abs/1801.01290). I
encountered these problems when I tried to translate the tf1 code of the Lyapunov Actor
Critic Agent of [Han et al 2019](http://arxiv.org/abs/2004.14288) into tf2 eager code.

## Use instructions

### Conda environment

From the general python package sanity perspective, it is a good idea to use conda environments to make sure packages from different projects do not interfere with each other.

To create a conda env with python3, one runs

```bash
conda create -n lac_clean_tf2_eager python=3.8
```

To activate the env:

```bash
conda activate lac_clean_tf2_eager
```

### Installation Environment

```bash
pip install -r requirements.txt
```

Then you are free to run main.py to train agents. Hyperparameters for training LAC in Cartpole are ready to run by default. If you would like to test other environments and algorithms, please open variant.py and choose corresponding 'env_name' and 'algorithm_name'.

### Run scripts

#### Full LAC implementations

The full LAC implementations LAC-tf1, LAC-tf2-eager and LAC-tf2-GRAPH can be started with the following python command:

```bash
python LAC-tf1/train.py
```

#### Grad problem scripts

The grad problem scripts can be started with the following python command:

```python
python tf2_val_grad_eager.py
```
