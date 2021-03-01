"""
Example script that runs the ODIN with second order derivatives regression on the classic setting for the
protein transduction model.

Felix Schur, ETH Zürich

based on code from

Gabriele Abbati, Machine Learning Research Group, University of Oxford
February 2019
"""


# Import libraries
import numpy as np
import tensorflow as tf
from odin import ProteinTransduction
from odin import TrainableProteinTransduction
from odin import ODIN


# Fix the random seeds for reproducibility
seed = 2
np.random.seed(seed)
tf.set_random_seed(seed)


# 1) Use the provided utilities class to simulate some noisy observations of
#    the protein transduction model

# We specify the time stamps for the observations
t_observations = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
                           0.9, 0.95, 1.0, 1.1,  1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0,
                           15.0, 20.0, 25.0, 30.0])

protein_transduction_simulator = ProteinTransduction(
    true_param=[0.07, 0.6, 0.05, 0.3, 0.017, 0.3], noise_variance=1e-2**2)

system_obs, t_obs = protein_transduction_simulator.observe_at_t(
    initial_state=(1.0, 0.0, 1.0, 0.0, 0.0), initial_time=0.0,
    final_time=30.0, t_delta_integration=0.01, t_observations=t_observations)

n_states, n_points = system_obs.shape


# 2) Initialize the provided TrainableProteinTransduction class and set some
#    bounds for the theta variables

# Constraints on parameters
theta_bounds = np.array([[1e-8, 10.0], [1e-8, 10.0], [1e-8, 10.0], [1e-8, 10.0],
                         [1e-8, 10.0], [1e-8, 10.0]])

# Trainable object
trainable_protein_transduction = TrainableProteinTransduction(
    n_states, n_points, bounds=theta_bounds)


# 3) Run the actual ODIN regression by initializing the optimizer, building the
#    model and calling the fit() function

# Constraints on states
state_bounds = np.array([[0.0, 2.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0],
                         [0.0, 2.0]])

# ODIN optimizer
odin_optimizer = ODIN(trainable_protein_transduction,
                      system_obs,
                      t_obs,
                      gp_kernel='Sigmoid',  # For PT we use the Sigmoid kernel
                      optimizer='L-BFGS-B',  # L-BFGS-B optimizer for the bounds
                      initial_gamma=1e-1,  # initial gamma value
                      initial_gamma_prime=1.0,  # initial gamma' value
                      use_sec_grads=True,  # we will use second order derivatives
                      train_gamma=True,  # gamma will be trained as well
                      train_gamma_prime=True,  # gamma' will be trained
                      state_bounds=state_bounds,  # Pass the state bounds
                      single_gp=False,  # Here we use one GP per state
                      basinhopping=True,  # Here we do use basinhopping
                      time_normalization=False,  # Better fit if off (empirical)
                      state_normalization=True)  # states normalization on

# Build the model
odin_optimizer.build_model()

# Fit the model
final_theta, final_gamma, final_x = odin_optimizer.fit()
print(final_theta)