from gaussianprocess import * 
from sklearn.preprocessing import StandardScaler

class MultiFidelityRegressor:
    def __init__(self, K, Ns, data_dict):
        """
        For creating a multifidelity regressor object.

        Parameters
        ----------
        K: int
            The number of levels of fidelity. 

        Ns: list
            A list of the number of datapoints of each fidelity-level.

        data_dict: dict
            A dictionary where the keys are the fidelity-levels and the values
            are dictionaries with keys 'X' and 'Y' for the input and output data
            at that fidelity-level. 
        """
        self.d = data_dict # Setting the data dictionary 
        self.K = K # Setting the number of levels of fidelity 
        self.Ns = Ns # A list of the data-sizes of each fidelity-level 
        self.input_dim = self.d[0]['X'].shape[0] # Setting the input dimension        

class Kriging(MultiFidelityRegressor):
    def fit(self, sigma_guess = None, lr = 1e-6, max_iter = 500):
        """
        For fitting a Kriging model to the the high-fidelity data.

        Parameters
        ----------
        sigma_guess: array_like
            (default = None) An array for the initial guess for the kernel hyperparameters.

        lr: float
            (default = 1e-6) The learning rate for the optimization algorithm.

        max_iter: int
            (default = 500) The maximum number of iterations for the optimization algorithm.
        """
        # Assining a default sigma 
        if sigma_guess is None:
            sigma_guess = 1e-4*np.ones(self.input_dim)

        # Training gaussian process 
        model = GaussianProcess(double_precision=True)
        model.fit(self.d[0]['X'], self.d[0]['Y'], sigma_guess)
        model.optimize_kernel_params(sigma_guess, lr = lr, max_iter=max_iter)

        self.model = model 

    def predict(self, Xtest):
        """
        For making predictions with the Kriging model.

        Parameters
        ----------
        Xtest: array_like
            The input data for which to make predictions.

        Returns
        ----------
        Yhat: array_like
            The predicted output data.
        """
        # Make predictions 
        Yhat, Ystd = self.model.predict(Xtest, include_std = True)

        return Yhat, Ystd


class AR1(MultiFidelityRegressor):
    def fit(self, param_scale = 1e-4, lr = 1e-6, max_iter = 500):
        """
        For fitting an AR1 model to the data, based on the 
        algorithm described the 2000 paper by Kennedy and O'Hagan.

        Parameters
        ----------
        param_scale: float
            (default = 1e-4) The scale of the initial guess for the kernel hyperparameters.

        lr: float
            (default = 1e-6) The learning rate for the optimization algorithm.

        max_iter: int
            (default = 500) The maximum number of iterations for the optimization algorithm.

        """
        sigma_guess = param_scale*np.random.uniform(size = self.input_dim)


        self.model_params = {}

        # Training GPR on Lowest Fidelity Model
        model = GaussianProcess(double_precision=True)
        model.fit(self.d[self.K]['X'], self.d[self.K]['Y'], sigma_guess)
        model.optimize_kernel_params(sigma_guess, lr = lr, max_iter = max_iter)

        self.model_params[self.K] = {
            'model':model
        }

        # Iterating through the levels of fidelity 
        for k in reversed(range(1, self.K+1)):
            yk = self.d[k]['Y'][:len(self.d[k-1]['Y'].ravel())]
            yk1 = self.d[k-1]['Y']

            # Computing the correlation coefficient 
            rho = (yk1.ravel().dot(yk.ravel())) / (yk.ravel().dot(yk.ravel()))

            # Computing the difference function
            delta = yk1 - rho * yk 

            # Declaring GPR Model
            model = GaussianProcess(double_precision=True)
            model.fit(self.d[k-1]['X'],delta, sigma_guess)
            model.optimize_kernel_params(sigma_guess, lr = lr, max_iter = max_iter)

            # Saving model parameters 
            self.model_params[k-1] = {
                'rho':rho, 
                'model':model
            }

    def predict(self, Xtest):
        """
        For making predictions with the AR1 model.

        Parameters
        ----------
        Xtest: array_like  
            The input data for which to make predictions.

        Returns
        ----------
        Yhat: array_like
            The predicted output
        """
        Yhat, std = self.model_params[self.K]['model'].predict(Xtest, include_std = True)

        for k in range(self.K):
            k = self.K - k # Starting from lowest-fidelity function 

            delta, this_std = self.model_params[k-1]['model'].predict(Xtest, include_std = True)
            rho = self.model_params[k-1]['rho']

            Yhat = rho * Yhat + delta # Making prediction
            std = np.sqrt(rho ** 2 * std + this_std ** 2)  # Variance propagation

        return Yhat, std
        
class NARGP(MultiFidelityRegressor):
    def fit(self, param_scale = 1e-4, lr = 1e-8, max_iter = 500):
        """
        For fitting a NARGP model to the data, based on the 
        algorithm described in the 2017 paper by Perdikaris et al. 

        Parameters
        ----------
        param_scale: float
            (default = 1e-4) The scale of the initial guess for the kernel hyperparameters.

        lr: float
            (default = 1e-8) The learning rate for the optimization algorithm.

        max_iter: int
            (default = 500) The maximum number of iterations for the optimization algorithm.

        """
        # Initialize kernel parameter guess 
        sigma_guess = param_scale*np.ones(self.input_dim+1)

        # Initialize dictionary to store model parameters 
        self.model_params = {}

        # Training GPR on Lowest Fidelity Model
        model = GaussianProcess(double_precision=True)
        model.fit(self.d[self.K]['X'], self.d[self.K]['Y'], 1e-4*np.ones(self.input_dim))
        model.optimize_kernel_params(1e-4*np.ones(self.input_dim), lr = lr, max_iter = max_iter)

        # Store model to parameter dictionary
        self.model_params[self.K] = {
            'model':model
        }

        # Iterating through the levels of fidelity 
        for k in reversed(range(1, self.K+1)):
            yk = self.d[k]['Y'][:len(self.d[k-1]['Y'].ravel())].reshape(1,-1)
            yk1 = self.d[k-1]['Y']


            # Create train inputs by vertically stacking the input features
            train_inputs = np.vstack((
                self.d[k-1]['X'], 
                yk
            ))

            # Declaring GPR Model
            model = GaussianProcess(double_precision=True)
            model.fit(train_inputs,yk1, sigma_guess)
            model.optimize_kernel_params(sigma_guess, lr = lr, max_iter = max_iter)

            # Saving model parameters 
            self.model_params[k-1] = {
                'model':model
            }

    def predict(self, Xtest):
        """
        For making predictions with the NARGP model.

        Parameters
        ----------
        Xtest: array_like
            The input data for which to make predictions.

        Returns
        ----------
        Yhat: array_like
            The predicted output data.

        """
        # Make the base prediction
        Yhat = self.model_params[self.K]['model'].predict(Xtest, include_std = False)

        # Iterate through the levels of fidelity
        for k in reversed(range(1, self.K+1)):
            # Create test inputs by vertically stacking the past predictions
            test_inputs = np.vstack((
                Xtest, 
                Yhat.reshape(1,-1)
            ))

            # Make the model prediction at this fidelity-level
            Yhat, std = self.model_params[k-1]['model'].predict(test_inputs, include_std = True)

        return Yhat, std

class Hyperkriging(MultiFidelityRegressor):
    def fit(self, param_scale = 1e-4, lr = 1e-6, max_iter = 500):
        """
        For fitting a Hyperkriging model to the data, based on the approach 
        described by Rex & Qian. 

        Parameters
        ----------
        param_scale: float
            (default = 1e-4) The scale of the initial guess for the kernel hyperparameters.

        lr: float
            (default = 1e-6) The learning rate for the optimization algorithm.

        max_iter: int
            (default = 500) The maximum number of iterations for the optimization algorithm.


        """
        # Initialize kernel parameter guess
        self.model_params = {}

        # Training GPR on Lowest Fidelity Model
        model = GaussianProcess(double_precision=True)
        model.fit(self.d[self.K]['X'], self.d[self.K]['Y'], param_scale*np.ones(self.input_dim))
        model.optimize_kernel_params(param_scale*np.ones(self.input_dim), lr = lr, max_iter = max_iter)

        # Store model to parameter dictionary
        self.model_params[self.K] = {
            'model':model
        }

        # Iterating through the levels of fidelity 
        for k in reversed(range(1, self.K+1)):
            yk1 = self.d[k-1]['Y']

            # Initializing list of input features 
            feature_list = [self.d[k-1]['X']]

            # Creating the feature-space progressively 
            for ki in range(k, self.K+1):
                feature_list.append(
                    self.d[ki]['Y'][:len(self.d[k-1]['Y'].ravel())].reshape(1,-1)
                )

            # Vertically stack features 
            train_inputs = np.vstack(feature_list)

            # Standard scaling training inputs
            scaler = StandardScaler() 
            # train_inputs = scaler.fit_transform(train_inputs.T).T

            # Initialize kernel parameter guess
            sigma_guess = param_scale*np.ones(train_inputs.shape[0])

            # Declaring GPR Model
            model = GaussianProcess(double_precision=True)
            model.fit(train_inputs,yk1, sigma_guess)
            model.optimize_kernel_params(sigma_guess, lr = lr, max_iter = max_iter)

            # Saving model parameters 
            self.model_params[k-1] = {
                'model':model, 
                'scaler':scaler
            }

    def predict(self, Xtest):
        """
        For making predictions with the Hyperkriging model.

        Parameters
        ----------
        Xtest: array_like
            The input data for which to make predictions.

        Returns
        ----------
        Yhat: array_like
            The predicted output data.

        """
        # Make the base prediction
        Yhat = self.model_params[self.K]['model'].predict(Xtest, include_std = False)

        # Create list of model-inputs which grows throughout the 
        # approximation process
        input_list = [Xtest]

        for k in reversed(range(1, self.K+1)):
            # Creating the progressive feature-space for the test data
            input_list.insert(1, Yhat.reshape(1,-1))

            # Vertically concatenating the input features 
            test_inputs = np.vstack(input_list)

            # Making the model prediction at this fidelity-level 
            Yhat, std = self.model_params[k-1]['model'].predict(test_inputs, include_std = True)
        
        print(self.model_params[0]['model'].kernel_params)

        return Yhat, std


            

        


            


