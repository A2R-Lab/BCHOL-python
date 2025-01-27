# BCHOL-python

Python proof of concept for [BCHOL](https://github.com/A2R-Lab/BCHOL): A Parallel linear system solver for optimal control. 

## Table of Contents  
1. [Introduction](#introduction)  
2. [Requirements](#requirements)  
3. [Installation](#installation)  
4. [Usage](#usage)  
6. [Citing](#citing)  

## Introduction

Solves for x in Ax = b, using the Recursive Schur Linear Quadratic Regulator explained in the paper [A Parallell Linear System Solver for Optimal Control](https://bjack205.github.io/papers/rslqr.pdf) by Brian E.Jackson. It requires A to be a positive semi-definite matrix to guarantee a good result.

This method is part of the Python implementation of trajectory optimization (trajopt) algorithms and model predictive control (MPC). Learn more about different TrajoptMPC [here](https://github.com/A2R-Lab/TrajoptMPCReference).

## Requirements
- [Python 3.10+](https://www.python.org/downloads/)
- Libraries:
    - Numpy
    - SciPy
## Instalation

- The following libraries (Numpy, Scipy) are included in the requirments.txt and can be downloaded with the following command
```shell
git clone https://github.com/A2R-Lab/BCHOL-python.git
pip3 install -r requirements.txt
```

## Usage


If you already have a defined LQR problem in a KKT form in a saved file **.json/.csv** you can look at ```solve_load.py``` for an example of how to load the problem. The program will return the solution in two forms:
1. Flattened dxul vector 
2. Brian Jackson's format as mentioned in the paper [lambda_i ; x_i; u_i]

```shell
python solve_load.py
```

If you just have an A matrix and a b vector look at  ```solve_build.py``` for an example of usage. Alternatively, you can use this package as part of our bigger solver[here](https://github.com/A2R-Lab/TrajoptMPCReference).



### Citing

Author: Yana Botvinnik
Contact: ybotvinn@barnard.edu

<!-- Finish the paper to be able to cite it!
To cite this work in your research, please use the following bibtex:
```

``` -->
