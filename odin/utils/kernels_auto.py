"""
Collection of kernel classes to be used in the Gaussian Process Regression. Compared to the standard implementation
of the kernels, here the derivatives are not hard coded, but automatically derived using sympy.

Felix Schur, ETH Zürich

based on code from

Gabriele Abbati, Machine Learning Research Group, University of Oxford
February 2019
"""

# Libraries
import tensorflow as tf
import numpy as np
from abc import ABC
import sympy as sp
import itertools


class GenericKernel(ABC):
    """
    Generic class for a Gaussian Process kernel.
    """

    def __init__(self, input_dim: int, use_single_gp: bool = False, m: int = 2, sim=sp.factor):
        """
        Constructor.
        :param input_dim: number of states.
        :param use_single_gp: boolean, indicates whether to use a single set of
        hyperparameters for all states (useful for extremely scarce data
        setting).
        :m: maximum order of derivatives to be computed
        :sim: type of simplification used for the symbolic derivatives
        """
        self.dimensionality = tf.constant(input_dim, dtype=tf.int32)
        self._initialize_variables(use_single_gp)
        self.m = m
        self.sim = sim
        return

    def _initialize_variables(self, use_single_gp: bool = False) -> None:
        """
        Initialize the hyperparameters of the kernel as TensorFlow variables.
        A logarithm-exponential transformation is used to ensure positivity
        during optimization.
        :param use_single_gp: boolean, indicates whether to use a single set of
        hyperparameters for all states (useful for extremely scarce data
        setting).
        """
        with tf.variable_scope('gaussian_process_kernel'):
            if use_single_gp:
                self.log_lengthscale = tf.Variable(np.log(1.0),
                                                   dtype=tf.float64,
                                                   trainable=True,
                                                   name='log_lengthscale')
                self.log_variance = tf.Variable(np.log(1.0),
                                                dtype=tf.float64,
                                                trainable=True,
                                                name='log_variance')
                self.lengthscales = \
                    tf.exp(self.log_lengthscale) \
                    * tf.ones([self.dimensionality, 1, 1], dtype=tf.float64)
                self.variances = \
                    tf.exp(self.log_variance) \
                    * tf.ones([self.dimensionality, 1, 1], dtype=tf.float64)
            else:
                self.log_lengthscales = tf.Variable(
                    np.log(1.0) * tf.ones([self.dimensionality, 1, 1],
                                          dtype=tf.float64),
                    dtype=tf.float64, trainable=True, name='log_lengthscales')
                self.log_variances = tf.Variable(
                    tf.ones([self.dimensionality, 1, 1],
                            dtype=tf.float64),
                    dtype=tf.float64, trainable=True, name='log_variances')
                self.variances = tf.exp(self.log_variances)
                self.lengthscales = tf.exp(self.log_lengthscales)
        return

    def _initilize_kernel_derivatives(self, kernel_fun, x, y):
        """
        Initializes the kernel derivatives.
        :kernel_fun: sympy function of the kernel k(x,y)
        :x: 'x' in sympy
        :y: 'y' in sympy
        """
        self.derivatives = [[None]*(self.m+1) for _ in range(self.m+1)]
        for i, j in itertools.product(range(self.m+1), range(self.m+1)):
            if i == 0 and j == 0:
                self.derivatives[i][j] = kernel_fun
            elif j == 0:
                self.derivatives[i][j] = self.sim(sp.diff(self.derivatives[i-1][j], x))
            else:
                self.derivatives[i][j] = self.sim(sp.diff(self.derivatives[i][j-1], y))
        return

    def _prepare_input(self, x, y):
        """
        Prepares the inputs to fit into the tensorflow derivative function.
        :x: 'x' the tensorflow tensor of k(x, y)
        :y: 'y' the tensorflow tensor of k(x, y)
        """
        shape = x.shape
        x = tf.reshape(x, [1, shape[0]])
        y = tf.reshape(y, [1, shape[0]])
        l = tf.reshape(self.lengthscales, [self.lengthscales.shape[0], 1])
        v = tf.reshape(self.variances, [self.variances.shape[0], 1])

        x_new = tf.repeat(x, shape[0], axis=1)
        x_new = tf.repeat(x_new, self.dimensionality, axis=0)

        y_new = tf.concat([y] * shape[0], axis=1)
        y_new = tf.repeat(y_new, self.dimensionality, axis=0)

        v_new = tf.repeat(v, shape[0] * shape[0], axis=1)
        l_new = tf.repeat(l, shape[0] * shape[0], axis=1)
        return x_new, y_new, l_new, v_new

    def compute_c_phi(self, xx: tf.Tensor,
                      yy: tf.Tensor) -> tf.Tensor:
        """
        To be implemented, compute the kernel covariance matrix between xx and
        yy for each state:
                    c_phi[n_s, i, j] = kernel(xx[i], yy[j])_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the tensor containing the covariance matrices.
        """
        dim = xx.shape[0]
        cov_matrix = self.derivatives[0][0](*self._prepare_input(xx, yy))
        cov_matrix = tf.reshape(cov_matrix, [-1, dim, dim])
        return cov_matrix

    def compute_diff_c_phi(self, xx: tf.Tensor,
                           yy: tf.Tensor) -> tf.Tensor:
        """
        To be implemented, compute the derivative of the kernel covariance
        matrix between xx and yy with respect to xx, for each state:
                    diff_c_phi[n_s, i, j] = d/dxx kernel(xx[i], yy[j])_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the tensor containing the covariance matrices.
        """
        dim = xx.shape[0]
        out = self.derivatives[1][0](*self._prepare_input(xx, yy))
        out = tf.reshape(out, [-1, dim, dim])
        return out

    def compute_c_phi_diff(self, xx: tf.Tensor,
                           yy: tf.Tensor) -> tf.Tensor:
        """
        To be implemented, compute the derivative of the kernel covariance
        matrix between xx and yy with respect to yy, for each state:
                    diff_c_phi[n_s, i, j] = d/dyy kernel(xx[i], yy[j])_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the tensor containing the covariance matrices.
        """
        return - self.compute_diff_c_phi(xx, yy)

    def compute_diff_c_phi_diff(self, xx: tf.Tensor,
                                yy: tf.Tensor) -> tf.Tensor:
        """
        To be implemented, compute the derivative of the kernel covariance
        matrix between xx and yy with respect to xx and yy, for each state:
            diff_c_phi[n_s, i, j] = d/dxx d/dyy kernel(xx[i], yy[j])_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the tensor containing the covariance matrices.
        """
        dim = xx.shape[0]
        out = self.derivatives[1][1](*self._prepare_input(xx, yy))
        out = tf.reshape(out, [-1, dim, dim])
        return out

    def compute_diff_diff_c_phi(self, xx: tf.Tensor,
                                yy: tf.Tensor) -> tf.Tensor:
        """
        To be implemented, compute the derivative of the kernel covariance
        matrix between xx and yy with respect to xx twice,
        for each state:
                    diff_c_phi[n_s, i, j] =
                    d^2/dxx^2 kernel(xx[i], yy[j])_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the tensor containing the covariance matrices.
        """
        dim = xx.shape[0]
        out = self.derivatives[2][0](*self._prepare_input(xx, yy))
        out = tf.reshape(out, [-1, dim, dim])
        return out

    def compute_c_phi_diff_diff(self, xx: tf.Tensor,
                                yy: tf.Tensor) -> tf.Tensor:
        """
        To be implemented, compute the derivative of the kernel covariance
        matrix between xx and yy with respect to y twice, for each state:
                    diff_c_phi[n_s, i, j] = d^2/dyy^2 kernel(xx[i], yy[j])_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the tensor containing the covariance matrices.
        """

        return self.compute_diff_diff_c_phi(xx, yy)

    def compute_diff_diff_c_phi_diff(self, xx: tf.Tensor,
                                     yy: tf.Tensor) -> tf.Tensor:
        """
        To be implemented, compute the derivative of the kernel covariance
        matrix between xx and yy with respect to twice xx and y, for each state:
                    diff_c_phi[n_s, i, j] = d^2/dxx^2 d/dyy kernel(xx[i], yy[j])_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the tensor containing the covariance matrices.
        """
        dim = xx.shape[0]
        out = self.derivatives[2][1](*self._prepare_input(xx, yy))
        out = tf.reshape(out, [-1, dim, dim])
        return out

    def compute_diff_c_phi_diff_diff(self, xx: tf.Tensor,
                                     yy: tf.Tensor) -> tf.Tensor:
        """
        To be implemented, compute the derivative of the kernel covariance
        matrix between xx and yy with respect to xx and twice to yy,
        for each state:
                    diff_c_phi[n_s, i, j] =
                    d/dxx d^2/dyy^2 kernel(xx[i], yy[j])_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the tensor containing the covariance matrices.
        """
        return - self.compute_diff_diff_c_phi_diff(xx, yy)

    def compute_diff_diff_c_phi_diff_diff(self, xx: tf.Tensor,
                                          yy: tf.Tensor) -> tf.Tensor:
        """
        To be implemented, compute the derivative of the kernel covariance
        matrix between xx and yy with respect to xx twice and twice to yy,
        for each state:
                    diff_c_phi[n_s, i, j] =
                    d^2/dxx^2 d^2/dyy^2 kernel(xx[i], yy[j])_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the tensor containing the covariance matrices.
        """
        dim = xx.shape[0]
        out = self.derivatives[2][2](*self._prepare_input(xx, yy))
        out = tf.reshape(out, [-1, dim, dim])
        return out


class RBFKernel(GenericKernel):
    """
    Implementation of the Radial Basis Function kernel.
    """
    def __init__(self, input_dim: int, use_single_gp: bool = False, m: int = 2):
        super(RBFKernel, self).__init__(input_dim, use_single_gp, m)

        self.x, self.y, self.l, self.v = sp.symbols('x y l v', real=True)
        function = self.v * sp.exp(-(self.x-self.y)**2.0 / 2.0 / self.l**2.0)
        self._initilize_kernel_derivatives(function, self.x, self.y)

        for i, j in itertools.product(range(self.m+1), range(self.m+1)):
            self.derivatives[i][j] = sp.lambdify([self.x, self.y, self.l, self.v], self.derivatives[i][j], "tensorflow")

        return


def tf_dirac_delta(inp, der="not_given"):
    return tf.zeros(inp.shape, dtype=tf.float64)


def tf_sign(inp):
    """
    Custom translation function of the sign-function for sympy lamdify.
    """
    cond = tf.less_equal(tf.abs(inp), 1e-10)
    out = tf.where(cond, tf.ones(shape=inp.shape, dtype=tf.float64), tf.cast(tf.sign(inp), dtype=tf.float64))
    return out


class Matern52Kernel(GenericKernel):
    """
    Implementation of the Matern 5/2 kernel.
    """
    def __init__(self, input_dim: int, use_single_gp: bool = False, m: int = 2):
        super(Matern52Kernel, self).__init__(input_dim, use_single_gp, m)
        self.x, self.y, self.l, self.v = sp.symbols('x y l v', real=True)
        function = self.v * (1.0 + sp.sqrt(5.0) / self.l * abs(self.x - self.y) + 5.0 / 3.0 / self.l**2 *
                             (self.x - self.y)**2) * sp.exp(-sp.sqrt(5.0) / self.l * abs(self.x - self.y))
        self._initilize_kernel_derivatives(function, self.x, self.y)

        for i, j in itertools.product(range(self.m+1), range(self.m+1)):
            self.derivatives[i][j] = sp.lambdify([self.x, self.y, self.l, self.v], self.derivatives[i][j],
                                                 modules=['tensorflow', {'DiracDelta': tf_dirac_delta,
                                                                         'sign': tf_sign}])


class Matern32Kernel(GenericKernel):
    """
    Implementation of the Matern 3/2 kernel.
    """
    def __init__(self, input_dim: int, use_single_gp: bool = False, m: int = 1):
        super(Matern32Kernel, self).__init__(input_dim, use_single_gp, m)
        self.x, self.y, self.l, self.v = sp.symbols('x y l v', real=True)
        function = self.v * (1.0 + sp.sqrt(3.0) / self.l * abs(self.x - self.y)) * sp.exp(-sp.sqrt(3.0) / self.l *
                                                                                          abs(self.x - self.y))
        self._initilize_kernel_derivatives(function, self.x, self.y)

        for i, j in itertools.product(range(self.m+1), range(self.m+1)):
            self.derivatives[i][j] = sp.lambdify([self.x, self.y, self.l, self.v], self.derivatives[i][j],
                                                 modules=['tensorflow', {'DiracDelta': tf_dirac_delta,
                                                                         'sign': tf_sign}])


class RationalQuadraticKernel(GenericKernel):
    """
    Implementation of the Rational-Quadratic kernel.
    """
    def __init__(self, input_dim: int, use_single_gp: bool = False,
                 alpha: float = 1.0, m: int = 2):
        super(RationalQuadraticKernel,
              self).__init__(input_dim, use_single_gp, sim=sp.simplify, m=m)
        self.alpha = tf.constant(alpha, dtype=tf.float64)
        self.x, self.y, self.l, self.v, self.a = sp.symbols('x y l v a', real=True)
        function = self.v*(1 + (self.x - self.y)**2 / (2 * self.a * self.l**2))**(-self.a)
        self._initilize_kernel_derivatives(function, self.x, self.y)

        for i, j in itertools.product(range(self.m + 1), range(self.m + 1)):
            self.derivatives[i][j] = sp.lambdify([self.x, self.y, self.l, self.v, self.a], self.derivatives[i][j],
                                                 "tensorflow")

        return

    def _prepare_input(self, x, y):
        """
        We need to overwrite the prepare input function here, since RQK has different variables.
        """
        shape = x.shape
        x = tf.reshape(x, [1, shape[0]])
        y = tf.reshape(y, [1, shape[0]])
        l = tf.reshape(self.lengthscales, [self.lengthscales.shape[0], 1])
        v = tf.reshape(self.variances, [self.variances.shape[0], 1])

        x_new = tf.repeat(x, shape[0], axis=1)
        x_new = tf.repeat(x_new, self.dimensionality, axis=0)

        y_new = tf.concat([y] * shape[0], axis=1)
        y_new = tf.repeat(y_new, self.dimensionality, axis=0)

        v_new = tf.repeat(v, shape[0] * shape[0], axis=1)
        l_new = tf.repeat(l, shape[0] * shape[0], axis=1)

        alpha_new = tf.reshape(self.alpha, [1, 1])
        alpha_new = tf.repeat(alpha_new, shape[0] * shape[0], axis=1)
        alpha_new = tf.repeat(alpha_new, self.dimensionality, axis=0)

        return x_new, y_new, l_new, v_new, alpha_new


class SigmoidKernel(GenericKernel):
    """
    Implementation of the Sigmoid kernel.
    """
    def __init__(self, input_dim: int, use_single_gp: bool = False, m: int = 2):
        super(SigmoidKernel,
              self).__init__(input_dim, use_single_gp, m)
        self.x, self.y, self.v, self.a_sp, self.b_sp = sp.symbols('x y v a b', real=True)
        function = self.v * sp.asin((self.a_sp + self.b_sp * self.x * self.y) /
                                    sp.sqrt((self.a_sp + self.b_sp * self.x**2 + 1) *
                                            (self.a_sp + self.b_sp * self.y**2 + 1)))
        self._initilize_kernel_derivatives(function, self.x, self.y)

        for i, j in itertools.product(range(self.m + 1), range(self.m + 1)):
            self.derivatives[i][j] = sp.lambdify([self.x, self.y, self.v, self.a_sp, self.b_sp],
                                                 self.derivatives[i][j], "tensorflow")

        return

    def _prepare_input(self, x, y):
        """
        We need to overwrite the prepare input function here, since Sigmoid Kernel has different variables.
        """
        shape = x.shape
        x = tf.reshape(x, [1, shape[0]])
        y = tf.reshape(y, [1, shape[0]])
        v = tf.reshape(self.variances, [self.variances.shape[0], 1])
        a = tf.reshape(self.a, [self.a.shape[0], 1])
        b = tf.reshape(self.b, [self.b.shape[0], 1])

        x_new = tf.repeat(x, shape[0], axis=1)
        x_new = tf.repeat(x_new, self.dimensionality, axis=0)

        y_new = tf.concat([y] * shape[0], axis=1)
        y_new = tf.repeat(y_new, self.dimensionality, axis=0)

        v_new = tf.repeat(v, shape[0] * shape[0], axis=1)
        a_new = tf.repeat(a, shape[0] * shape[0], axis=1)
        b_new = tf.repeat(b, shape[0] * shape[0], axis=1)

        return x_new, y_new, v_new, a_new, b_new

    def _initialize_variables(self, use_single_gp: bool = False) -> None:
        """
        We need to overwrite the prepare input function here, since the Sigmoid kernel has different variables.
        """
        with tf.variable_scope('gaussian_process_kernel'):
            if use_single_gp:
                self.log_a_single = tf.Variable(np.log(1.0), dtype=tf.float64,
                                                trainable=True,
                                                name='sigmoid_a')
                self.log_b_single = tf.Variable(np.log(1.0), dtype=tf.float64,
                                                trainable=True,
                                                name='sigmoid_b')
                self.log_variance = tf.Variable(np.log(1.0), dtype=tf.float64,
                                                trainable=True,
                                                name='log_variance')
                self.a = \
                    tf.exp(self.log_a_single) \
                    * tf.ones([self.dimensionality, 1, 1], dtype=tf.float64)
                self.b = \
                    tf.exp(self.log_b_single) \
                    * tf.ones([self.dimensionality, 1, 1], dtype=tf.float64)
                self.variances = \
                    tf.exp(self.log_variance) \
                    * tf.ones([self.dimensionality, 1, 1], dtype=tf.float64)
            else:
                self.log_a = tf.Variable(np.log(1.0) *
                                         tf.ones([self.dimensionality, 1, 1],
                                                 dtype=tf.float64),
                                         dtype=tf.float64,
                                         trainable=True,
                                         name='sigmoid_a')
                self.log_b = tf.Variable(np.log(1.0) *
                                         tf.ones([self.dimensionality, 1, 1],
                                                 dtype=tf.float64),
                                         dtype=tf.float64,
                                         trainable=True,
                                         name='sigmoid_b')
                self.log_variances = tf.Variable(
                    np.log(1.0) * tf.ones([self.dimensionality, 1, 1],
                                          dtype=tf.float64),
                    dtype=tf.float64, trainable=True, name='variances')
                self.a = tf.exp(self.log_a)
                self.b = tf.exp(self.log_b)
                self.variances = tf.exp(self.log_variances)
        return

    def compute_c_phi_diff(self, xx: tf.Tensor,
                           yy: tf.Tensor) -> tf.Tensor:
        """
        Need to redefine because Sigmoid kernel is not symmetric.
        """
        dim = xx.shape[0]
        out = self.derivatives[0][1](*self._prepare_input(xx, yy))
        out = tf.reshape(out, [-1, dim, dim])
        return out

    def compute_c_phi_diff_diff(self, xx: tf.Tensor,
                                yy: tf.Tensor) -> tf.Tensor:
        """
        Need to redefine because Sigmoid kernel is not symmetric.
        """
        dim = xx.shape[0]
        out = self.derivatives[0][2](*self._prepare_input(xx, yy))
        out = tf.reshape(out, [-1, dim, dim])
        return out

    def compute_diff_c_phi_diff_diff(self, xx: tf.Tensor,
                                     yy: tf.Tensor) -> tf.Tensor:
        """
        Need to redefine because Sigmoid kernel is not symmetric.
        """
        dim = xx.shape[0]
        out = self.derivatives[1][2](*self._prepare_input(xx, yy))
        out = tf.reshape(out, [-1, dim, dim])
        return out

