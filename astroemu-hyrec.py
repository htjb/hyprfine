"""Tinkering."""

import glob

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from astroemu.dataloaders import SpectrumDataset
from astroemu.network import mlp
from astroemu.normalisation import log_base_10, standardise
from astroemu.serialisation import load, save
from astroemu.train import train
from astroemu.utils import compute_mean_std

files = glob.glob("hyrec-data/*.npz")[:2500]
print(f"Found {len(files)} files.")
train_files = files[: int(len(files) / 100 * 80)]
val_files = files[int(len(files) / 100 * 80) : int(len(files) / 100 * 90)]
test_files = files[int(len(files) / 100 * 90) :]

for label in ['xe', 'tk']:
    variable_input = ['H0', 'omb', 'omc', 'yhe']
    log10 = log_base_10(log_all_y=True, log_all_params=True)
    train_dataset = SpectrumDataset(
        files=train_files,
        x="z",
        y=label,
        variable_input=variable_input,
        tiling=False,
        allow_pickle=True,
        forward_pipeline=log10,
    )

    _, x, _ = train_dataset[0]

    mean_spec, std_spec, mean_x, std_x, mean_params, std_params = compute_mean_std(
        train_dataset.get_batch_iterator(batch_size=1024, shuffle=False)
    )

    standard = standardise(
        y_mean=mean_spec,
        y_std=std_spec,
        x_mean=mean_x,
        x_std=std_x,
        params_mean=mean_params,
        params_std=std_params,
    )

    train_dataset.tiling = True
    train_dataset.forward_pipeline = [log10, standard]

    val_dataset = SpectrumDataset(
        files=val_files,
        x="z",
        y=label,
        variable_input=variable_input,
        tiling=True,
        allow_pickle=True,
        forward_pipeline=[log10, standard],
    )
    test_dataset = SpectrumDataset(
        files=test_files,
        x="z",
        y=label,
        variable_input=variable_input,
        tiling=True,
        allow_pickle=True,
        forward_pipeline=[log10, standard]
    )

    config = {
        "hidden_size": 32,
        "nlayers": 4,
        "act": "tanh",
        "epochs": 1000,
        "patience": 50,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
    }

    best_params, train_losses, val_losses = train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        **config,
        batch_size=512
    )

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.savefig(f"docs/hyrec-emulators/hyrec_training_curve_{label}.png")
    plt.close()

    save(
        f"hyprfine/data/hyrec_{label}.astroemu",
        best_params,
        train_losses,
        val_losses,
        **config,
        loss="mse",
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )

    loaded = load(f"hyprfine/data/hyrec_{label}.astroemu")

    predictions = []
    true_values = []
    for batch in test_dataset.get_batch_iterator(batch_size=32, shuffle=False):
        y, params = batch
        preds = mlp(loaded["params"], params, act=loaded["hyperparams"]["act"])
        # reshape from tiled (batch*len_x,) to (batch, len_x) before the
        # backward pass so per-frequency statistics broadcast correctly
        preds = preds.reshape(-1, len(x))
        y = y.reshape(-1, len(x))
        for pipe in reversed(test_dataset.forward_pipeline):
            preds, _, _ = pipe.backward(preds, x, params)
            y, _, _ = pipe.backward(y, x, params)
        predictions.append(preds)
        true_values.append(y)

    predictions = jnp.vstack(predictions)
    true_values = jnp.vstack(true_values)
    print(predictions.shape, true_values.shape)

    [plt.plot(x, predictions[i, :], c='r', ls='--') for i in range(10)]
    [plt.plot(x, true_values[i, :], c='k', ls='-') for i in range(10)]
    plt.loglog()
    plt.savefig(f"docs/hyrec-emulators/hyrec_predictions_{label}.png")
    plt.close()
