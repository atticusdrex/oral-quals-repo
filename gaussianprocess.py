import numpy as np 
import matplotlib.pyplot as plt 
import math 
from tqdm import tqdm
import jax 
import jax.numpy as jnp
from jax import vmap
from copy import copy

# Define the kernel function (e.g., RBF kernel)
def rbf_kernel(x1, x2, kernel_params):
    """
    Radial Basis Function (RBF) kernel.

    Parameters
    ----------
    x1, x2: array_like
        two d-dimensional vectors. 

    kernel_params: array_like
        a d-dimensional vector specifying the diagonals of a 
        covariance matrix which is inverted. 
    
    Returns: 
    ---------
    a scalar evaluating the RBF kernel at these two inputs 
    """
    h = (x1-x2).ravel()

    return jnp.exp(-jnp.sum(h**2 / kernel_params))

# Gaussian Process Regression Class 
def K(X1, X2, kernel_func, kernel_params):
    '''
    Function for computing a kernel matrix

    Parameters
    ----------
    X1: array_like
        a dxN1 array of inputs where each column is an
        observation of a specific input. 
    X2: array_like
        a dxN2 array of inputs where each column is an
        observation of a specific input. 

    Returns: 
    -----------
    array_like: the N1 x N2 kernel matrix with the kernel
    function evaluated at each entry ij. 
    '''
    return vmap(
        vmap(lambda x, y: kernel_func(x,y, kernel_params), in_axes=(None,1)),
        in_axes=(1,None)
    )(X1, X2)

# Objective function for GPs (derived from MLE of Prior) 
def loss(p, kernel_func, X, Y, noise_var):
    """
    A function computing the loss for a specific set of 
    training data and kernel function. 

    Parameters
    ----------
    p: dict
        a dictionary containing the parameters for which we 
        are optimizing. In this case it's just 'kernel_params'
    
    kernel_func: function
        a function for which to form the kernel matrix. 

    X: array_like 
        a dxN array of input training data 

    Y: an Nx1 array of corresponding output training data 

    noise_var: the variance of any gaussian white noise in Y 

    Returns
    ----------

    The scalar loss-function value of the Marginal Likelihood 
    of the training data. 
    """

    Ktrain = K(X, X, kernel_func, p['kernel_params']) + noise_var * jnp.eye(X.shape[1])
    jnp.linalg.cond(Ktrain)
    # Compute the scalar log-likelihood
    L = jnp.linalg.cholesky(Ktrain)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    quadratic_term = Y.T @ jax.scipy.linalg.cho_solve((L, True), Y) # Solve for (Ktrain^-1 f)

    # Combine terms into a scalar
    loss = 0.5*(quadratic_term + logdet)

    return loss.squeeze()  # Ensure the input is a scalar

def adam_step(params, grads, lr, t, m, v, beta1=0.99, beta2=0.9999, epsilon=1e-8):
    """
    Perform a single optimization step using the ADAM algorithm.

    Parameters:
        params (np.ndarray): Current parameter values.
        grads (np.ndarray): Gradient of loss with respect to parameters.
        lr (float): Learning rate.
        t (int): Current iteration count.
        m (np.ndarray): First moment vector (mean of gradients).
        v (np.ndarray): Second moment vector (mean of squared gradients).
        beta1 (float): Decay rate for the first moment.
        beta2 (float): Decay rate for the second moment.
        epsilon (float): Small constant for numerical stability.

    Returns:
        updated_params (np.ndarray): Updated parameters after the ADAM step.
        m (np.ndarray): Updated first moment vector.
        v (np.ndarray): Updated second moment vector.
    """

    # Update biased first and second moment estimates
    m = beta1 * m + (1 - beta1) * grads
    v = beta2 * v + (1 - beta2) * (grads ** 2)

    # Compute bias-corrected first and second moments
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    # Update parameters
    updated_params = params - lr * m_hat / (np.sqrt(v_hat) + epsilon)

    return updated_params, m, v

class GaussianProcess:
    """
    The main class for training, storing, and 
    optimizing Gaussian Processes. 
    """
    def __init__(self, kernel_func = rbf_kernel, double_precision = False, rcond=1e-12):
        """
        The constructor of the Gaussian Process class

        Parameters
        ----------

        kernel_func: array_like (default = rbf_kernel)
            a kernel function if the user would like to pass it in. 

        double_precision: bool (default = False)
            specifying whether the user would like to use double
            precision floating point arithmetic (default is 32-bit). 

        rcond: float (default = 1e-10) many of the algorithms require linear solves and for 
            numerical stability it's often desirable to regularize results 
            by cutting off relative singular values below a certain tolerance. 
        """
        # Set the kernel function 
        self.kernel_func = kernel_func

        # Set the matrix conditioning 
        self.rcond = rcond

        # Enable 64-bit floating point precision 
        if double_precision:
            jax.config.update("jax_enable_x64", True)


    def fit(self, X, Y, kernel_params, noise_var=0.0):
        """
        This is the function which trains the Gaussian Process model on its 
        training data. It does not optimize its hyperparameters. 

        Parameters
        ----------
        X: array_like
            The d x N array of input training data where each column represents 
            an observation of the input and d is the dimension of the 
            input. 

        Y: array_like
            the N x 1 array of output training data where the jth row represents 
            the output corresponding to the jth column of X. 

        kernel_params: array_like
            the parameters to be passed into the kernel function.


        noise_var: float (default = 0.0)
            the variance of any Gaussian White Noise in Y 
        """
        # Kernel_params is a 1d numpy array containing kernel parameters 
        self.kernel_params = jnp.array(kernel_params)

        # noise_var is the variance of random-noise in Y
        # Pass in X (d x N) 2d array and Y, (N) 1d vectors
        self.X = jnp.array(X) 
        self.Y = jnp.array(Y.ravel())
        self.noise_var = noise_var

        # Compute the training matrix 
        self.Ktrain = K(self.X, self.X, self.kernel_func, self.kernel_params) + noise_var * jnp.eye(self.X.shape[1])

        cond_num = jnp.linalg.cond(self.Ktrain)
        # Check condition number of kernel matrix 
        if cond_num > 1e8:
            print("Warning! Kernel Matrix is close to singular: K=%d" % (int(cond_num)))

        # Compute weights by solving linear system
        self.alpha = jnp.linalg.lstsq(self.Ktrain, self.Y, rcond=self.rcond)[0]

    
    def predict(self, Xtest, include_std=True):
        """
        This function is for the online prediction of the training 
        data. 

        Parameters
        ----------
        Xtest: array_like
            The d x M array of testing inputs for which we would like 
            to approximate the value of the Gaussian Process for M 
            inputs. 

        include_std: bool (default = True)
            Whether or not to include the analytical standard deviation 
            associated with the Gaussian Process predictions 

        Returns
        ----------
        Yhat: A length-M array of model predictions at the inputs 

        Ystd: A length-M array of the standard deviation associated 
            with each prediction. 
        """

        # Compute testing matrix 
        Ktest = K(Xtest, self.X, self.kernel_func, self.kernel_params) 

        # Expected value of test inputs 
        Yhat = Ktest @ self.alpha

        # Standard deviation of prediction at test inputs
        if include_std:
            Ystd = jnp.sqrt(jnp.diag(K(Xtest, Xtest, self.kernel_func, self.kernel_params) - Ktest @ jnp.linalg.lstsq(self.Ktrain, Ktest.T, rcond=self.rcond)[0]))
            return Yhat, Ystd 
        else:
            return Yhat
        
    def optimize_kernel_params(self, kernel_param_guess, lr=1e-2, tol = 1e-6, max_iter = 10000, verbose=True):
        """
        This function is for optimizing the hyperparameters of the 
        Gaussian Process. This step is not required, but is necessary
        for the generalizability and accuracy of the model. 

        Parameters
        ----------
        kernel_param_guess: array_like
            the initial guess at the kernel parameters. 

        lr: float (default = 1e-2)
            The learning rate of the algorithm. 

        tol: float (default = 1e-6)
            The stopping tolerance of the algorithm. 
        
        max_iter: int (default = 10000)
            The maximum number of iterations of the optimization
            before force interrupting. 

        verbose: bool (default = True)
            Whether or not to print out the progress of the optimization. 
        """
        # Defining a parameter dictionary
        p = {
            'kernel_params':kernel_param_guess
        }

        grad_func = jax.grad(lambda p: loss(p, self.kernel_func, self.X, self.Y, self.noise_var))
        
        # Compute the gradient function of our loss-function
        initial_loss = loss(p, self.kernel_func, self.X, self.Y, self.noise_var)

        # Print the loss at the parameter guess 
        if verbose:
            print(f"Initial Loss: {initial_loss:.5f}")

        # Defining an iterator
        if verbose:
            iterator = tqdm(range(max_iter))
        else:
            iterator = range(max_iter)

        # Initializing best_loss variable
        best_loss = 1e99

        # Initializing moment vectors 
        m = np.zeros_like(p['kernel_params'])
        v = np.zeros_like(p['kernel_params'])

        # Looping through the iterator 
        for t in iterator:
            # Printing the loss at the step
            this_loss = loss(p,self.kernel_func, self.X, self.Y, self.noise_var)

            # Checking stagnation 
            if this_loss < best_loss:
                best_loss = this_loss
                best_params = copy(p)
                stagnation_count = 0 
            else:
                stagnation_count += 1
            
            # Break if we have not improved in 10 steps
            if stagnation_count > 500:
                print("No Improvements Made! Breaking Loop...")
                break

            
            if verbose:
                iterator.set_postfix_str("Current Loss: %.5f Learning Rate: %.2e"  % (this_loss, lr))
            
            # Making a trial step
            grads = grad_func(p)

            trial_params, trial_m, trial_v = adam_step(p['kernel_params'], grads['kernel_params'], lr, t+1, m, v)
            
            
            trial_p = {
                'kernel_params':trial_params
            }

            # Taking the gradient at the trial step
            trial_grads = grad_func(trial_p)
            trial_loss = loss(trial_p,self.kernel_func, self.X, self.Y, self.noise_var)

            # Waiting until the trial step is valid i.e. no NaNs and we lower our loss-function
            while (jnp.isnan(trial_loss).any() or jnp.isnan(trial_grads['kernel_params']).any()):
                # Making the learning rate smaller
                lr *= 0.5

                # Making a trial step 
                trial_params, trial_m, trial_v = adam_step(p['kernel_params'], grads['kernel_params'], lr, t+1, m, v)
                trial_p = {
                    'kernel_params':trial_params
                }

                # Taking the gradient at the trial step
                trial_grads = grad_func(trial_p)
                trial_loss = loss(trial_p,self.kernel_func, self.X, self.Y, self.noise_var)

            # Saving the next parameter step as the trial p 
            p, m, v = copy(trial_p), trial_m, trial_v

            
            
        # Save the best parameters
        p = best_params
        
        # Print final loss
        if verbose:
            print(f"Final Loss: {loss(p, self.kernel_func, self.X, self.Y, self.noise_var):.5f}\n")

        # Save best kernel params 
        self.kernel_params = p['kernel_params']

        # Retrain the model 
        self.fit(self.X, self.Y, self.kernel_params, noise_var = self.noise_var)







    
    
    

