# ODIN with second order derivatives
This code extends ODIN by incorporating second order derivatives.
ODIN is a Gaussian Process based gradient matching technique, that performs parameter inference for ODEs in the case where observations are avaible.

The original paper:
> [**ODIN: ODE-Informed Regression for Parameter and State Inference in Time-Continuous Dynamical Systems.**](https://arxiv.org/abs/1902.06278)
> *Philippe Wenk\*, Gabriele Abbati\*, Michael A Osborne, Bernhard Schölkopf, Andreas Krause and Stefan Bauer*. Proceedings of the 34th AAAI Conference on Artificial Intelligence (AAAI 20). New York, NY, USA. AAAI Press.
> \
> \*: equal contribution

## Code

The code provided is written in Python 3.6, and relies on the following libraries:
* [TensorFlow](https://www.tensorflow.org/)
* [numpy](http://www.numpy.org/)
* [scipy](https://www.scipy.org/)
* [sympy](https://www.sympy.org/)

