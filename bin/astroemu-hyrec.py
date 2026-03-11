"""Train HYREC emulators for xe and Tk."""

import glob
import os
from collections import Counter
from pathlib import Path

import numpy as np

os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=8"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from astroemu.dataloaders import SpectrumDataset
from astroemu.network import mlp
from astroemu.normalisation import log_base_10, standardise
from astroemu.serialisation import load, save
from astroemu.train import train
from astroemu.utils import compute_mean_std

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalisation import focus_on_recombination, downsample

ROOT = Path(__file__).resolve().parent.parent

_all_files = glob.glob(str(ROOT / "hyrec-data" / "*.npz"))[:1000]
print(f"Found {len(_all_files)} files.")

# Determine the most common spectrum shape and drop malformed files.
def _shape(path: str) -> tuple:
    try:
        d = np.load(path, allow_pickle=True)
        return (len(d["z"]), len(d["xe"]), len(d["tk"]))
    except Exception:
        return (-1, -1, -1)

_shapes = [_shape(f) for f in _all_files]
_expected = Counter(_shapes).most_common(1)[0][0]
files = [f for f, s in zip(_all_files, _shapes) if s == _expected]
print(
    f"Kept {len(files)} files with shape {_expected} "
    f"(dropped {len(_all_files) - len(files)})."
)
train_files = files[: int(len(files) / 100 * 80)]
val_files = files[int(len(files) / 100 * 80) : int(len(files) / 100 * 90)]
test_files = files[int(len(files) / 100 * 90) :]

for label in ['xe', 'tk']:
    variable_input = ['H0', 'omb', 'omc', 'yhe']
    if label == 'tk':
        focus = downsample()
    else:
        focus = focus_on_recombination()
    log10 = log_base_10(log_all_y=True, log_all_x=True)
    train_dataset = SpectrumDataset(
        files=train_files,
        x="z",
        y=label,
        variable_input=variable_input,
        tiling=False,
        allow_pickle=True,
        forward_pipeline=[focus, log10],
    )

    _, x, _ = train_dataset[0]

    mean_spec, std_spec, mean_x, std_x, mean_params, std_params = (
        compute_mean_std(
            train_dataset.get_batch_iterator(
                batch_size=1024, shuffle=False
            )
        )
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
    train_dataset.forward_pipeline = [focus, log10, standard]

    val_dataset = SpectrumDataset(
        files=val_files,
        x="z",
        y=label,
        variable_input=variable_input,
        tiling=True,
        allow_pickle=True,
        forward_pipeline=[focus, log10, standard],
    )
    test_dataset = SpectrumDataset(
        files=test_files,
        x="z",
        y=label,
        variable_input=variable_input,
        tiling=True,
        allow_pickle=True,
        forward_pipeline=[focus, log10, standard],
    )

    config = {
        "hidden_size": 64,
        "nlayers": 2,
        "act": "tanh",
        "epochs": 500,
        "patience": 20,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
    }

    best_params, train_losses, val_losses = train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        **config,
        batch_size=5120,
    )

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.savefig(ROOT / "docs" / "hyrec-emulators" / f"hyrec_training_curve_{label}.png")
    plt.close()

    save(
        str(ROOT / "hyprfine" / "data" / f"hyrec_{label}.astroemu"),
        best_params,
        train_losses,
        val_losses,
        **config,
        loss="mse",
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )

    loaded = load(str(ROOT / "hyprfine" / "data" / f"hyrec_{label}.astroemu"))

    predictions = []
    true_values = []
    for batch in test_dataset.get_batch_iterator(batch_size=3200, shuffle=False):
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

    ylabels = {'xe': '$x_e$', 'tk': '$T_k$ [K]'}
    [plt.plot(x, predictions[i, :], c='r', ls='--') for i in range(10)]
    [plt.plot(x, true_values[i, :], c='k', ls='-') for i in range(10)]
    plt.loglog()
    plt.xlabel('Redshift $z$')
    plt.ylabel(ylabels[label])
    plt.savefig(ROOT / "docs" / "hyrec-emulators" / f"hyrec_predictions_{label}.png")
    plt.close()

    # percentage error across all test samples as a function of z
    percent_err = jnp.abs(
        (predictions - true_values) / true_values
    ) * 100
    median_err = jnp.median(percent_err, axis=0)
    upper = jnp.percentile(percent_err, 84, axis=0)
    lower = jnp.percentile(percent_err, 16, axis=0)

    plt.plot(x, median_err, c='k')
    plt.fill_between(
        x, lower, upper, alpha=0.3, color='k', label='16th–84th percentile'
    )
    plt.xlabel('Redshift $z$')
    plt.ylabel(f'Percentage error in {ylabels[label]} [%]')
    plt.xscale('log')
    plt.legend()
    plt.savefig(
        ROOT / "docs" / "hyrec-emulators" / f"hyrec_percent_error_{label}.png"
    )
    plt.close()
