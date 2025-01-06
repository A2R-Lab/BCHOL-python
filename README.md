# BCHOL-python

Python proof of concept for [BCHOL](https://github.com/A2R-Lab/BCHOL). 

Solves for x in Ax = b, using the Recursive Schur Linear Quadratic Regulator explained in the paper [A Parallell Linear System Solver for Optimal Control](https://bjack205.github.io/papers/rslqr.pdf) by Brian E.Jackson. It requires A to be a positive semi-definite matrix to guarantee a good result.

## Requirements
- [Python 3.10+](https://www.python.org/downloads/)


- The following libraries (Numpy, Scipy) are included in the requirments.txt and can be downloaded with the following command
```shell
pip3 install -r requirements.txt
```

## Usage

<!-- Add actual code lines for example! -->

If you already have a defined LQR problem in a KKT form in a saved file **.json/.csv** you can look at ```solve_load.py``` for an example. The program will return the solution in two forms:
1. Flattened dxul vector 
2. Brian Jackson's format as mentioned in the paper [lambda_i ; x_i; u_i]

```shell
python solve_load.py
```

If you just have an A matrix and a b vector look at  ```solve_build.py``` for an example.

Both files will return an xyz solution vector in dxul.



### Citing

Author: Yana Botvinnik
Contact: ybotvinn@barnard.edu

<!-- Finish the paper to be able to cite it!
To cite this work in your research, please use the following bibtex:
```

``` -->
