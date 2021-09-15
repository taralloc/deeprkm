# Training Algorithms for Constrained Deep Restricted Kernel Machines

## Abstract

Code for optimization algorithms to train Constrained Deep Restricted Kernel Machines (C-DRKMs). The available algorithms are the quadratic penalty and the augmented Lagrangian method with Adam or L-BFGS optimization for the subproblems, the Projected Gradient method, and the Cayley ADAM algorithm.

## Code structure

- The main script is located in `train_cdrkm.py`: here all the available hyperparameters can be set and you can find the main loop of each training algorithm.
- The Projected Gradient and L-BFGS optimizers are implemented in `st_optimizers.py`.
- The optimization problem of C-DRKMs is defined in `cdrkm_model.py`.

## Usage
### Create conda environment
Navigate to the cloned repository. Create a conda environment named *rkm_env* with all the required packages with the following command

```
conda env create -f environment.yml
```

Make sure you have two input and output directories at `~/out` and `~/data`.

### Datasets

Any PyTorch dataset can be used by adding it to `utils.py`. Procedural datasets used in disentanglement experiments can be downloaded from [disentanglement_lib](https://github.com/google-research/disentanglement_lib). The datasets are downloaded the first time you use them and are stored in `~/data` for future use.

### Train

Activate the conda environment with the command `conda activate rkm_env` and run one of the following commands, for example:
```bash
python train_cdrkm.py --dataset mnist -N 100 --train_algo 0 --maxiterations 3000 --maxouteriterations 10
```
for training with Quadratic Penalty on a subset of 100 images from MNIST for maximum 3000 inner iterations and maximum 10 outer iterations, or

```bash
python train_cdrkm.py --dataset mnist -N 100 --train_algo 3 --maxiterations 1000
```

for training with Projected Gradient.

### Training Algorithms

The `train_algo` command line argument specifies the training algorithm. It is an integer with the following meaning: (0) for Quadratic Penalty, (5) for Augmented Lagrangian, (2) for Cayley ADAM, (3) for Projected Gradient, and (7) for Quadratic Penalty using the AM routine.

### Evaluation

The program creates a directory in `~/out/deeprkm/` containing the trained model and its hyperparameters. The `cdrkm_eval.py` file evaluates the quality of the found solution, displays information on running time, and compares different solutions of different training algorithms; [disentanglement_lib](https://github.com/google-research/disentanglement_lib) can be used to evaluate disentangling performance.

### Help

```
usage: train_cdrkm.py [-h] [-N ND] [-a {0,5,2,3,7}] [-s S [S ...]]
                      [-k {0,1,2,3} [{0,1,2,3} ...]]
                      [-kp KERNELPARAM [KERNELPARAM ...]] [-lwi] [-d DATASET]
                      [-mi MAXITERATIONS] [-mit MAXINNERTIME]
                      [-moi MAXOUTERITERATIONS] [-ok] [-gamma GAMMA]
                      [-epsilon EPSILON] [-rs SEED] [-lr LR]
                      [--tau_min TAU_MIN] [--p P] [--beta BETA]
                      [-ia {lbfgs,adam}]

optional arguments:
  -h, --help            show this help message and exit
  -N ND, --Nd ND        number of training samples
  -a {0,5,2,3,7}, --train_algo {0,5,2,3,7}
                        training algorithm: (0) penalty, (5) augmented
                        Lagrangian, (2) Cayley ADAM, (3) Projected Gradient,
                        (7) quadratic penalty based on AM
  -s S [S ...], --s S [S ...]
                        number of components in each layer
  -k {0,1,2,3} [{0,1,2,3} ...], --kernel {0,1,2,3} [{0,1,2,3} ...]
                        kernel of each layer: (0) RBF, (1) poly, (2) Laplace
                        or (3) sigmoid
  -kp KERNELPARAM [KERNELPARAM ...], --kernelparam KERNELPARAM [KERNELPARAM ...]
                        kernel parameter of each level: RBF or Laplace
                        bandwidth or poly degree
  -lwi, --layerwisein   layer-wise initialization
  -d DATASET, --dataset DATASET
                        name of the dataset
  -mi MAXITERATIONS, --maxiterations MAXITERATIONS
                        maximum number of training inner iterations
  -mit MAXINNERTIME, --maxinnertime MAXINNERTIME
                        maximum number of minutes for inner loop
  -moi MAXOUTERITERATIONS, --maxouteriterations MAXOUTERITERATIONS
                        maximum number of training outer iterations
  -ok, --optimizekernel
                        optimizes kernel parameters
  -gamma GAMMA, --gamma GAMMA
                        gamma for all levels
  -epsilon EPSILON, --epsilon EPSILON
                        epsilon for terminating condition of optimization
  -rs SEED, --seed SEED
                        random seed
  -lr LR, --lr LR       learning rate
  --tau_min TAU_MIN     tau min for algorithm 0 or epsilon min for algorithm 7
  --p P                 p for algorithm 0 or delta for algorithm 7
  --beta BETA           beta for algorithm 0 and 7
  -ia {lbfgs,adam}, --inneralgorithm {lbfgs,adam}
                        inner training algorithm
```
