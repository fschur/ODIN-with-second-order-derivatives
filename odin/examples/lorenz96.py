"""
Example script that runs the ODIN with secon order derivatives regression on the Lorenz '96 model.

Felix Schur, ETH Zürich

based on code from

Gabriele Abbati, Machine Learning Research Group, University of Oxford
February 2019
"""


# Import libraries
import numpy as np
import tensorflow as tf
from odin import Lorenz96
from odin import TrainableLorenz96
from odin import ODIN


# Fix the random seeds for reproducibility
seed = 2
np.random.seed(seed)
tf.set_random_seed(seed)


# 1) Use the provided utilities class to simulate some noisy observations of
#    the Lorenz '96 model

n_states = 4  # Number of states, can be changed to the preferred setting

lorenz96_simulator = Lorenz96(n_states=n_states,
                              true_param=7.0,
                              noise_variance=1.0)

system_obs, t_obs = lorenz96_simulator.observe(initial_state=1.0,
                                               initial_time=0.0,
                                               final_time=5.0,
                                               t_delta_integration=0.01,
                                               t_delta_observation=0.03)

n_states, n_points = system_obs.shape


# 2) Initialize the provided TrainableLorenz96 class

# Trainable object
trainable_l96 = TrainableLorenz96(n_states, n_points)


# 3) Run the actual ODIN regression by initializing the optimizer, building the
#    model and calling the fit() function

# ODIN optimizer
odin_optimizer = ODIN(trainable_l96,
                      system_obs,
                      t_obs,
                      gp_kernel='Matern52',  # For L96 we use the Matern kernel
                      optimizer='L-BFGS-B',  # L-BFGS-B optimizer for the bounds
                      initial_gamma=1.0,  # initial gamma value
                      initial_gamma_prime=10.0,  # initial gamma' value
                      use_sec_grads=True,  # we use second order derivatives
                      train_gamma=True,  # gamma will be trained as well
                      train_gamma_prime=True,  # gamma' will be trained
                      single_gp=False,  # Here we use one GP per state
                      basinhopping=False,  # we don't use the basinhopping here
                      time_normalization=True,  # time normalization on
                      state_normalization=True)  # states normalization on

# Build the model
odin_optimizer.build_model()

# Fit the model
final_theta, final_gamma, final_x = odin_optimizer.fit()
print(final_theta)