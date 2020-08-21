# Perceptron-8051
A perceptron for the 8051. Useful for end-to-end implementation of any application. Simply requires retraining.

# Installation

It is important to first clone into the repository. We can achieve this in the following ways:

## Git Users

In a Git-enabled machine, type into git commandline, or if git is in PATH, in windows CMD:

```
git clone https://github.com/KillingJoke42/perceptron-8051
cd perceptron-8051
```

## Non-Git Users

For non-git users, we need to siphon the files manually. Do so by clicking the green `Code` button in the right corner of the repository.

# Usage

There are two aspects to the code: one is a python module for retraining the weights and the other are C source files for keil to flash to an 8051. <br>
The file tree is as follows:

```
root
  keil_files
    - neuron.c
    - weights.c
    - weights.h
  - get_weights.py
```

## 8051 Usage
`keil_files` possesses all the files that need to be added to the keil project. 
<ul>
  <li> neuron.c: file executes operation on the 8051 </li>
  <li> weights.c: all weights and biases are stored here </li>
  <li> weights.h: header file for neuron.c to link to weights </li>
</ul>

## get_weights Usage
python script `get_weights.py` are to retrain the weights.
Simply place the input/output mappings in `main()` as when calling `train()` as follows:
```
train(<input>, <output>)
```
Execute `get_weights.py` using a python interpreter to re-write `weights.c`
Once this is done, you may observe your application output on `neuron.c`. Any changes as per application must also be made to the same file.
