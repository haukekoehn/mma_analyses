from dataclasses import dataclass, field
import pickle
import time
from typing import Sequence, Callable

import matplotlib.pyplot as plt
import pandas as pd

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, Float
from flax import linen as nn  # Linen API
from flax.training.train_state import TrainState
from fiesta.utils import train_test_split
from fiesta.scalers import MinMaxScalerJax, StandardScalerJax
from fiesta.train.neuralnets import serialize

import optax

@dataclass
class Config:
    layer_sizes: list[int]
    input_ndim: int
    learning_rate: float
    nb_epochs: int
    optimizer: str = "adam"
    batch_size: int = 2048
    key: jax.Array = field(default_factory=lambda: jax.random.key(18936892))

    def __post_init__(self):
        self.nb_report = self.nb_epochs // 10


class MLP(nn.Module):
    """Basic multi-layer perceptron: a feedforward neural network with multiple Dense layers."""
    layer_sizes: Sequence[int]
    act_func: Callable = nn.relu

    def setup(self):
        self.layers = [nn.Dense(n) for n in self.layer_sizes]

    @nn.compact
    def __call__(self, x: Array, train: bool = False):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.act_func(x)

        x = self.layers[-1](x)
        #x = nn.relu(x)
        return x
    
def bce(y, pred):
    """
    binary cross entropy between y and the predicted array pred
    """
    return -jnp.mean(y * jnp.log(pred) + (1-y) * jnp.log(1-pred))

def mse(y, pred):
    """
    mse error between y and the prediceted array
    """
    return jnp.mean((y-pred)**2) 

def load_training_data(file: str):

    df = pd.read_csv(file, sep=" ")

    X = df[["mass_1_source", "mass_2_source", "theta_jn", "alt", "az", "psi"]].to_numpy()
    X[:,0] = df["mass_1_source"] + df["mass_2_source"]
    X[:,1] = df["mass_2_source"] / df["mass_1_source"]
    y = df["snr"].to_numpy().astype(float)

    y = jnp.log10(y).reshape(-1, 1)

    scaler = StandardScalerJax()
    yscaler = StandardScalerJax()

    X_scaled = scaler.fit_transform(X)
    y_scaled = yscaler.fit_transform(y)

    train_X, val_X, train_y, val_y = train_test_split(X_scaled, y_scaled, 0.9)

    return train_X, val_X, train_y, val_y, scaler, yscaler


@staticmethod
@jax.jit
def train_step(state, batch_X, batch_y):
    def loss_fn(params):
        pred_y = state.apply_fn({'params': params}, batch_X, train=True)
        return mse(batch_y, pred_y)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@staticmethod
@jax.jit
def eval_step(state, X, y):
    pred_y = state.apply_fn({'params': state.params}, X, train=False)
    return mse(y, pred_y)

@staticmethod 
def train_loop(config,
               train_X,
               train_y,
               val_X=None,
               val_y=None,
               verbose=True):

    net = MLP(layer_sizes=config.layer_sizes)
    key, subkey = jax.random.split(config.key)

    params = net.init(subkey, jnp.ones(config.input_ndim), train=False)['params']
    
    # switch optimizer if you want true SGD
    if config.optimizer == "sgd":
        tx = optax.sgd(config.learning_rate)
    else:
        tx = optax.adam(config.learning_rate)

    state = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

    train_losses, val_losses = [], []
    best_state = state
    best_val_loss = jnp.inf

    n_samples = train_X.shape[0]
    batch_size = config.batch_size

    start = time.time()

    for epoch in range(config.nb_epochs):

        # Shuffle dataset each epoch
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, n_samples)

        train_X_shuffled = train_X[perm]
        train_y_shuffled = train_y[perm]

        epoch_loss = 0.0
        n_batches = n_samples // batch_size

        # Mini-batch loop (this is SGD)
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            batch_X = train_X_shuffled[start_idx:end_idx]
            batch_y = train_y_shuffled[start_idx:end_idx]

            state, batch_loss = train_step(state, batch_X, batch_y)
            epoch_loss += batch_loss

        epoch_loss /= n_batches

        # Validation
        if val_X is not None:
            val_loss = eval_step(state, val_X, val_y)
        else:
            val_loss = jnp.zeros_like(epoch_loss)

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)

        # Track best model
        if val_X is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = state

        if epoch % config.nb_report == 0 and verbose:
            print(f"Epoch {epoch+1}")
            print(f"Train loss: {epoch_loss}")
            print(f"Valid loss: {val_loss}")
            print(f"Best valid loss: {best_val_loss}")
            print("---")

    end = time.time()

    if verbose:
        print(f"Training took {end-start} seconds.")
        if val_X is not None:
            print(f"Best validation loss: {best_val_loss}")
        print("\n \n")

    trained_state = best_state if val_X is not None else state

    return trained_state, train_losses, val_losses

def save_model(trained_state, config, outfile: str):

    serialize_dict = serialize(trained_state, config)
    with open(outfile, 'wb') as handle:
        pickle.dump(serialize_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main(file: str):

    if "ETL" in file:
        detector = "ETL"
    else:
        detector = "ETT"

    train_X, val_X, train_y, val_y, scaler, yscaler = load_training_data(file)
    nnconfig = Config(
        layer_sizes=[32, 32, 1],
        input_ndim= train_X.shape[1],
        learning_rate=1e-2,
        nb_epochs=2000,
        batch_size=2**16,
        optimizer="sgd"
    )

    trained_state, train_losses, val_losses = train_loop(
        nnconfig,
        train_X,
        train_y,
        val_X,
        val_y,
        verbose=True,
    )

    # plot loss
    plt.plot(train_losses, color="blue")
    plt.plot(val_losses, color="red")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.savefig(f"./networks/{detector}_loss_curve.png", bbox_inches="tight")
    plt.close()

    # plot AUC
    pred_y = trained_state.apply_fn({'params': trained_state.params}, val_X, train=False)
    plt.scatter(yscaler.inverse_transform(val_y), yscaler.inverse_transform(pred_y), color="blue")
    plt.xlabel("validation data")
    plt.ylabel("predicted data")
    plt.plot(jnp.linspace(yscaler.inverse_transform(val_y).min(), yscaler.inverse_transform(val_y).max(), 2), jnp.linspace(yscaler.inverse_transform(val_y).min(), yscaler.inverse_transform(val_y).max(), 2), color="black", linestyle="dashed")

    plt.savefig(f"./networks/prediction_test_{detector}.png", bbox_inches="tight")
    plt.close()


    save_model(trained_state, nnconfig, outfile=f"./networks/{detector}_snr_nn.pkl")

    scalers_dict = dict(Xscaler=scaler, yscaler=yscaler)
    with open(f"./networks/{detector}_snr_nn_scalers.pkl", 'wb') as handle:
        pickle.dump(scalers_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__=="__main__":
    main("./training_data/train_ETL.dat")
    main("./training_data/train_ETT.dat")
