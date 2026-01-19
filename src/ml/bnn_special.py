"""
Special BNN models for certain projects 
"""

import os
os.environ["KERAS_BACKEND"] = "jax"     # order matters !

from keras.src.layers import Input, Dense, Dropout
from keras import optimizers
from keras import Sequential

from .bnn_base import BnnBase, gaussian_nll


# ===================================
# For MagLearn project
# ===================================

class BnnMs(BnnBase):
    random_seed: int = None
    epochs: int = 800
    name: str = "Bnn_Magnetization"

    def __init__(self, n_features: int, n_samples: int = 50, random_seed: int = None):
        """
        Magnetisation model
        """
        super().__init__(n_features=n_features, epochs=self.epochs, n_samples=n_samples, model_name=self.name,
                         random_seed=random_seed)

    @classmethod
    def _build_model(cls, n_features: int, random_seed: int) -> Sequential:
        """
        Defines the model architecture. Child classes can override this method if needed.

        :param n_features: number of nodes in the input layer
        :return: Compiled Sequential model
        """
        model = Sequential()
        # Input layer
        model.add(Input(shape=(n_features,)))

        # Hidden layers - used in paper (2025-11-28)
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='tanh'))
        model.add(Dropout(0.5, seed=random_seed))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='tanh'))
        model.add(Dropout(0.5, seed=random_seed))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='tanh'))
        model.add(Dropout(0.5, seed=random_seed))

        # Output layer
        model.add(Dense(2, activation=None))
        # Compile the model
        optim = optimizers.RMSprop()
        model.compile(optimizer=optim, loss=gaussian_nll)
        return model


class BnnHa(BnnBase):
    random_seed: int = None
    epochs: int = 400
    name: str = "Bnn_Anisotropy"

    def __init__(self, n_features: int, n_samples: int = 50, random_seed: int = None):
        """
        Anisotropy model
        """
        super().__init__(n_features=n_features, epochs=self.epochs, n_samples=n_samples, model_name=self.name,
                         random_seed=random_seed)

    @classmethod
    def _build_model(cls, n_features: int, random_seed: int) -> Sequential:
        """
        Defines the model architecture. Child classes can override this method if needed.

        :param n_features: number of nodes in the input layer
        :return: Compiled Sequential model
        """
        model = Sequential()
        # Input layer
        model.add(Input(shape=(n_features,)))
        # Hidden layers

        # SOTA (2025-11-27)
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='tanh'))
        model.add(Dropout(0.5, seed=random_seed))

        # Output layer
        model.add(Dense(2, activation=None))
        # Compile the model
        optim = optimizers.RMSprop()
        model.compile(optimizer=optim, loss=gaussian_nll)
        return model
