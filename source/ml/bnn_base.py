"""
Bayesian Neural Network (BNN)

Sources:
    - Theory: G. Scalia, C. A. Grambow, B. Pernici, Y.-P. Li, and W. H. Green, “Evaluating Scalable Uncertainty
      Estimation Methods for DNN-Based Molecular Property Prediction,” Oct. 07, 2019, arXiv: arXiv:1910.03127.
      Accessed: Nov. 05, 2024. [Online]. Available: http://arxiv.org/abs/1910.03127
    - Initial model architecture: https://github.com/huyng/incertae by Huy Nguyen
      Accessed: Dec. 2024
"""

import os
from tabnanny import verbose
os.environ["KERAS_BACKEND"] = "jax"     # order matters !
import keras

import io
import time
from abc import ABC
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.exceptions import NotFittedError
from sklearn.base import RegressorMixin, BaseEstimator

from keras.src.layers import Dropout
from keras import Sequential
from keras import ops


# =========================================================================================================
# Utility Functions
# =========================================================================================================

def check_is_fitted(model) -> bool:
    """
    If the model is an instance of BnnBase, then return its is_fitted attribute.
    Else apply the check_is_fitted() of sci-kit learn.

    :param model: a machine learning model
    :return:
    """
    if isinstance(model, BnnBase):
        return model.is_fitted
    else:
        from sklearn.utils.validation import check_is_fitted as cif
        try:
            cif(model)
            return True
        except:
            return False



@keras.saving.register_keras_serializable()
def gaussian_nll(y_true, y_pred):
    """
    Gaussian negative log likelihood

    Note: to make training more stable, we optimize a modified loss by having our model
    predict log(sigma^2) rather than sigma^2.

    Source: https://github.com/huyng/incertae/blob/master/mcdrop_regression.ipynb

    :return: log(sigma^2)
    """
    y_true = ops.reshape(y_true, [-1])
    mu = y_pred[:, 0]
    si = y_pred[:, 1]
    loss = (si + ops.square(y_true - mu) / ops.exp(si)) / 2.0
    return ops.mean(loss)



# =========================================================================================================
# Bayesian Neural Network (BNN) base class
# =========================================================================================================

class BnnBase(RegressorMixin, BaseEstimator, ABC):
    """
    Bayesian neural networks with Monte Carlo dropout. This is the base class.

    EXPLANATION:
        The prediction consists of 2 values: mu and si. So a mean and a standard deviation.
        They are defined implicitly through the use of a special loss function "gaussian_nll()".
    """

    def __init__(self, n_features: int,  epochs: int, n_samples: int, model_name:str = None, random_seed: int = None):
        """
        :param epochs: train duration
        :param n_samples: number of predictions made to compute distribution
        """
        self.n_features: int = n_features  # number of nodes in the input layer
        self.epochs: int = epochs
        self.n_samples: int = n_samples
        self.is_fitted: bool = False
        self.model_name: str = model_name
        self.random_seed: int = None if random_seed is None else int(random_seed)

        # -------------------------------------------------------------------
        # Define Model
        # -------------------------------------------------------------------
        self.model: Sequential = self._build_model(n_features=self.n_features, random_seed=self.random_seed)
        if self.model_name:
            self.model.name = self.model_name


    def summary(self, model: Sequential | None = None) -> str:
        """
        Capture print output of `model.summary()` and return it as a clean string,
        including activation functions and dropout ratios.
        """
        target_model = model or self.model

        def capture_summary(model: Sequential) -> str:
            buffer = io.StringIO()
            print("*" * 76, file=buffer)
            model.summary(print_fn=lambda x: buffer.write(x + "\n"))
            return buffer.getvalue()

        # Capture the model's summary
        summary_text = capture_summary(target_model)

        # Gather extra info: activation functions + dropout ratios
        layer_info = []
        for layer in target_model.layers:
            activation = getattr(layer, "activation", None)
            activation_name = activation.__name__ if activation else "None"

            dropout_rate = getattr(layer, "rate", None)  # Dropout layers have 'rate'
            if dropout_rate is not None:
                info = f"{layer.__class__.__name__}: {dropout_rate}"
            else:
                n_nodes = getattr(layer, "units", None) 
                info = f"{layer.__class__.__name__}:   {activation_name}({n_nodes})"

            layer_info.append(info)
        
        def _get_learning_rate(opt) -> str:
            lr_obj = getattr(opt, "learning_rate", None)
            if lr_obj is None:
                return "unknown"
            try:
                return f"{float(lr_obj):.6g}"
            except Exception:
                pass
            try:
                return f"{float(getattr(lr_obj, 'numpy')()):.6g}"
            except Exception:
                pass
            try:
                cfg = lr_obj.get_config()
                for key in ("learning_rate", "initial_learning_rate"):
                    if key in cfg:
                        return f"{float(cfg[key]):.6g}"
            except Exception:
                pass
            return str(lr_obj)

        lr = _get_learning_rate(target_model.optimizer)

        additional_info = "\n".join([
            "Layer Details:",
            " - " + "\n - ".join(layer_info),
            f"Optimizer: {target_model.optimizer.get_config()['name']}",
            f"Learning Rate: {lr}",
            f"Epochs: {self.epochs}",
            f"Samples per prediction: {self.n_samples}",
            f"is_fitted: {self.is_fitted}",
            f"Random Seed: {self.random_seed}",
        ])

        return f"{summary_text}\n{additional_info}"


    def save_model_to_disk(self, base_path: str | Path):
        """
        Saves the custom estimator object and its Keras model.

        :param base_path: The base directory or filename prefix to save the model.
                          e.g., 'saved_models/my_bnn_model' will save:
                          - 'saved_models/my_bnn_model.keras' (Keras model)
                          - 'saved_models/my_bnn_model.joblib' (estimator attributes)
        """
        # get the base path as a string
        if isinstance(base_path, Path):
            base_path = str(base_path)
            
        # Ensure the directory exists
        os.makedirs(os.path.dirname(base_path), exist_ok=True)

        # 1. Save the Keras model
        keras_model_path = f"{base_path}.keras"
        if self.model: # Ensure model exists before saving
            self.model.save(keras_model_path)
            print(f"Keras model saved to: {keras_model_path}")
        else:
            raise ValueError("Keras model not initialized, cannot save.")

        # 2. Temporarily set self.model to None to avoid joblib trying to pickle it
        temp_model = self.model
        self.model: Sequential = None

        # 3. Save the rest of the object's attributes using joblib
        joblib_path = f"{base_path}.joblib"
        joblib.dump(self, joblib_path)
        print(f"Estimator attributes saved to: {joblib_path}")

        # 4. Restore the Keras model attribute (for continued use in current session)
        self.model = temp_model


    @classmethod
    def load_model_from_disk(cls, base_path: str):
        """
        Loads the custom estimator object and its Keras model.

        :param base_path: The base directory or filename prefix used during saving.
        :return: Loaded BayesianNN instance.
        """
        joblib_path = f"{base_path}.joblib"
        keras_model_path = f"{base_path}.keras"

        # 1. Load the estimator's attributes
        if not os.path.exists(joblib_path):
            raise FileNotFoundError(f"Joblib file not found: {joblib_path}")
        loaded_instance = joblib.load(joblib_path)
        print(f"Estimator attributes loaded from: {joblib_path}")

        # 2. Load the Keras model into the loaded instance
        if not os.path.exists(keras_model_path):
            raise FileNotFoundError(f"Keras model file not found: {keras_model_path}")
        # When loading a custom loss function, you need to provide it in custom_objects
        loaded_keras_model = keras.models.load_model(keras_model_path, custom_objects={'gaussian_nll': gaussian_nll})
        loaded_instance.model = loaded_keras_model
        print(f"Keras model loaded from: {keras_model_path}")

        if check_is_fitted(loaded_instance):
            print(f"Model {loaded_instance} is fitted.")
        else:
            print(f"Model {loaded_instance} is not fitted.")

        return loaded_instance


    @classmethod
    def _build_model(cls, n_features: int, random_seed: int=None) -> Sequential:
        """
        Defines the model architecture. Child classes can override this method if needed.

        USAGE:
            model = Sequential()
            model.add(Input(shape=(n_features,)))
            optim = optimizers.RMSprop()
            model.compile(optimizer=optim, loss=gaussian_nll)
            return model

        :param n_features: number of nodes in the input layer
        :return: Compiled Sequential model
        """
        raise NotImplementedError("Please implement this method in the subclass.") 


    def _get_fit_kwargs(self) -> dict:
        """Hook for subclasses to customize keras.Model.fit keyword arguments."""
        return {}


    def fit(self, X, Y):
        """
        PHASE 2 MODEL FIT FOR DEPLOYMENT

        ---
        Fit the model to the training data.
        :param X: Training input data.
        :param Y: Training target data.
        :return: Training history.
        """
        if self.is_fitted:
            raise UserWarning("This model is already fitted!")
        start_time = time.time()
        print(f"Fitting started at:  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

        n_data_features = np.asarray(X).shape[1]
        assert n_data_features == self.n_features, f"n_features ({self.n_features}) does not match number of features in train data dimensions ({n_data_features})!"
        fit_kwargs = self._get_fit_kwargs() or {}
        self.history = self.model.fit(
            X,
            Y,
            batch_size=32,
            epochs=self.epochs,
            verbose=0,
            **fit_kwargs,
        )
        self.is_fitted = True
        end_time = time.time()
        print(f"Fitting finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Total fitting time of {self.__class__.__name__}.{self.model_name}: {end_time - start_time:.2f} seconds")
        print(f"-"*70)
        return self.history


    def predict(self, X) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        An array (mu, si) is returned. Thus, the prediction describes a distribution instead af a deterministic value.
        The values mu and si are defined implicitly through the use of a special loss function "gaussian_nll()".

        REFERENCE:
            # Reference: Scalia et al., 2019, p. 11

        :param X: sample matrix with features to predict on
        :return: (y_mean_pred, std_total, std_aleatoric, std_epistemic)
        """
        if not self.is_fitted:
            raise NotFittedError("Please fit the model before you predict.")

        # Predict `n_samples` times and return the average.
        mu_arr, si_arr = [], []
        for _ in range(self.n_samples):
            y_pred = self.model(X, training=True) # IMPORTANT: `training=True` enables dropout at prediction time
            mu_arr.append(y_pred[:, 0])  # mu
            si_arr.append(y_pred[:, 1])  # si sigma   # IMPORTANT: expected from Gaussian NLL loss function!

        # Convert lists to arrays
        mu_arr  = np.array(mu_arr)  # mu: array of y_pred from n_samples forward passes
        si_arr  = np.array(si_arr)  # sigma: array of predicted log(sigma^2) from n_samples forward passes
        var_arr = np.exp(si_arr)    # sigma^2_a -> e^si, because we predicted log(sigma^2) in the loss function for numerical stability 

        # prediction value 
        y_mean = np.mean(mu_arr, axis=0)  # predicted value <- mean prediction ove n_samples

        # Uncertainty decomposition: Split std into aleatoric and epistemic
        var_al = np.mean(var_arr, axis=0) # aleatoric variance
        var_ep = np.var(mu_arr, axis=0)   # epistemic variance

        # Take root of variances to receive standard deviations
        std_total     = np.sqrt(var_al + var_ep) 
        std_aleatoric = np.sqrt(var_al)
        std_epistemic = np.sqrt(var_ep)
        return y_mean, std_total, std_aleatoric, std_epistemic


    def predict_rel_std(self, X) -> tuple[np.ndarray, np.ndarray]:
        """
        Call predict and return only mean and standard deviation scaled by mean.

        :param X:
        :return: (mean, std/mean)
        """
        y_mean, y_std, _, _ = self.predict(X)
        return y_mean, (y_std / y_mean)


    @classmethod
    def cross_validate(cls, X: pd.DataFrame, y: pd.DataFrame, n_splits: int = 5, random_seed: int = None
                       ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[float, float, float]]:
        """
        Computes and returns all three metrics (MSE, MAE, and R²) and also returns the predicted results from the
        same cross-validation. 

        :param n_splits: The number of folds. Default=5
        :param random_seed: random seed for fold definition
        :return: (y_pred_cv, y_std_cv), (mae_cv_avg, mse_cv_avg, r2_cv_avg)
        """
        print("Cross Validation running...")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        mae_scores, mse_scores, r2_scores = [], [], []

        # Initialize arrays to store predictions in the original order
        y_pred_cv = np.zeros_like(y, dtype=float)  # Placeholder for predictions
        y_std_cv = np.zeros_like(y, dtype=float)   # Placeholder for std deviations
        std_al_cv = np.zeros_like(y, dtype=float)  # Placeholder for std deviations
        std_ep_cv = np.zeros_like(y, dtype=float)  # Placeholder for std deviations

        for i, (train_index, val_index) in enumerate(kf.split(X)):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            # Initialize a new model for each split
            reg = cls(n_features=X.shape[1])
            fold_seed = random_seed + i if random_seed is not None else None
            print(f"Using random seed {fold_seed} for fold {i + 1}")
            reg.random_seed = fold_seed
            reg.epochs = cls.epochs  # Overwrite epochs if ensemble shall save time compared to sub-models

            # Fit new model
            print(f"\nFitting for cross-validation split {i}/{n_splits} ...\n")
            reg.fit(X_train, y_train)

            # Get predictions and standard deviations
            y_pred, y_std, std_al, std_ep = reg.predict(X_val)

            # Store predictions in the correct positions
            y_pred_cv[val_index] = y_pred
            y_std_cv[val_index] = y_std
            std_al_cv[val_index] = std_al
            std_ep_cv[val_index] = std_ep

            # Get scores
            mae_scores.append(mean_absolute_error(y_val, y_pred))
            mse_scores.append(mean_squared_error(y_val, y_pred))
            r2_scores.append(r2_score(y_val, y_pred))

        # -------------------------------------------------------------------
        # Handle Scores
        # -------------------------------------------------------------------
        mae_avg = np.mean(mae_scores).round(3)
        mse_avg = np.mean(mse_scores).round(3)
        r2_avg = np.mean(r2_scores).round(3)

        print(f'Average R2  across {n_splits}-fold CV: {r2_avg}')
        print(f'Average MAE across {n_splits}-fold CV: {mae_avg}')
        print(f'Average MSE across {n_splits}-fold CV: {mse_avg}')
        print("Cross-validation finished\n")

        return (y_pred_cv, y_std_cv, std_al_cv, std_ep_cv), (mae_avg, mse_avg, r2_avg)
