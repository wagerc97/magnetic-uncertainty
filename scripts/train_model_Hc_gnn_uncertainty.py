#!/usr/bin/env python3

"""Train and evaluate the adapted GNN uncertainty workflow using the immutable submodule."""

from copy import deepcopy
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.nn import GaussianNLLLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
_GNN_PATH = _ROOT / "GNN_Uncertainty"
if not _GNN_PATH.exists():
    raise FileNotFoundError(f"Required submodule directory not found: {_GNN_PATH}. Please run `bash setup.sh` to initialize the submodule.")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_GNN_PATH) not in sys.path:
    sys.path.insert(0, str(_GNN_PATH))
print(f"Project root added to python path:\n{_ROOT}")
print(f"GNN path added to python path:\n{_GNN_PATH}")

from evaluate import evaluate_with_uncertainty
from models import GNNModel
from scaler_helper import fit_and_apply_scalers, get_dataset
from src.evaluation.confidence_styled import plot_confidence_curve
from src.evaluation.residual_styled import plot_cv_predictions_hexbin


LABEL = "Hc"
SYMBOL = r"$\mu_0 H_c$"
SEED = 123
N_EVAL_SEEDS = 10
EVAL_SEEDS = list(range(SEED, SEED + N_EVAL_SEEDS))
TEST_SIZE = 0.3
VAL_SIZE = 0.1
SCRIPT_NAME = Path(__file__).stem
OUT_PATH = Path(__file__).parent / f"out_{SCRIPT_NAME}"
BEST_EPOCH_PATH = OUT_PATH / "best_epoch.txt"


def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(dataset, config, device):
    return GNNModel(
        orig_node_fea_len=dataset[0].num_node_features,
        edge_fea_len=config["edge_fea_len"],
        node_fea_len=config["node_fea_len"],
        n_conv=config["n_conv"],
        h_fea_len=config["h_fea_len"],
        n_h=config["n_h"],
    ).to(device)


def build_optimizer_and_scheduler(model, config):
    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=config["scheduler"].get("mode", "min"),
        factor=config["scheduler"].get("factor", 0.5),
        patience=config["scheduler"].get("patience", 5),
        threshold=float(config["scheduler"].get("threshold", 1e-4)),
        verbose=config["scheduler"].get("verbose", True),
    )
    return optimizer, scheduler


def split_dataset(dataset, seed, include_validation):
    indices = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=seed,
        shuffle=True,
    )

    dataset_train_full = [dataset[i] for i in train_idx]
    dataset_test = [dataset[i] for i in test_idx]

    if not include_validation:
        return dataset_train_full, [], dataset_test

    if len(dataset_train_full) < 2:
        raise ValueError("Need at least two training samples to create a validation split.")

    val_fraction = min(VAL_SIZE, (len(dataset_train_full) - 1) / len(dataset_train_full))
    dataset_train, dataset_val = train_test_split(
        dataset_train_full,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
    )
    return dataset_train, dataset_val, dataset_test


def make_scaled_datasets_and_loaders(dataset_train, dataset_val, dataset_test, batch_size):
    dataset_train_s, dataset_val_s, dataset_test_s, label_scaler = fit_and_apply_scalers(
        dataset_train,
        dataset_val,
        dataset_test,
    )
    loader_train = DataLoader(dataset_train_s, batch_size=batch_size, shuffle=True)
    loader_val = None
    if dataset_val_s:
        loader_val = DataLoader(dataset_val_s, batch_size=batch_size, shuffle=False)
    loader_test = DataLoader(dataset_test_s, batch_size=batch_size, shuffle=False)
    return dataset_train_s, dataset_val_s, dataset_test_s, loader_train, loader_val, loader_test, label_scaler


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            mu, log_var = model(data)
            loss = criterion(data.y.view(-1, 1), mu, torch.exp(log_var))
            total_loss += loss.item()
    return total_loss / len(loader.dataset)


def train_with_validation(model, train_loader, val_loader, optimizer, scheduler, device, config):
    criterion = GaussianNLLLoss()
    patience = config.get("patience", 30)
    best_val_loss = float("inf")
    best_epoch = config["epochs"]
    epochs_without_improvement = 0
    save_path = Path(config["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        for data in train_loader:
            data = data.to(device)
            mu, log_var = model(data)
            loss = criterion(data.y.view(-1, 1), mu, torch.exp(log_var))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_loss = evaluate_loss(model, train_loader, criterion, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model at epoch {epoch}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered. Restoring best model from epoch {best_epoch}")
            model.load_state_dict(torch.load(save_path, map_location=device))
            break
    else:
        model.load_state_dict(torch.load(save_path, map_location=device))

    return best_epoch


def train_fixed_epochs(model, train_loader, optimizer, scheduler, device, epochs, save_path):
    criterion = GaussianNLLLoss()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        for data in train_loader:
            data = data.to(device)
            mu, log_var = model(data)
            loss = criterion(data.y.view(-1, 1), mu, torch.exp(log_var))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_loss = evaluate_loss(model, train_loader, criterion, device)
        scheduler.step(train_loss)
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f}")

    torch.save(model.state_dict(), save_path)


def scale_uncertainty(values, label_scaler):
    return np.asarray(values).reshape(-1) * float(label_scaler.scale_[0])


def evaluate_split(dataset, config, device, seed, epochs):
    set_all_seeds(seed)
    dataset_train, _, dataset_test = split_dataset(dataset, seed=seed, include_validation=False)
    _, _, _, loader_train, _, loader_test, label_scaler = make_scaled_datasets_and_loaders(
        dataset_train,
        [],
        dataset_test,
        batch_size=config.get("batch_size", 100),
    )

    model = build_model(dataset, config, device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    train_fixed_epochs(
        model,
        loader_train,
        optimizer,
        scheduler,
        device,
        epochs=epochs,
        save_path=OUT_PATH / f"weights_seed_{seed}.pt",
    )

    num_samples = config.get("mc_dropout_samples", 50)
    train_pred, train_true, train_epi, train_alea = evaluate_with_uncertainty(
        loader_train,
        model,
        num_samples,
        device,
        label_scaler,
    )
    test_pred, test_true, test_epi, test_alea = evaluate_with_uncertainty(
        loader_test,
        model,
        num_samples,
        device,
        label_scaler,
    )

    train_pred = np.asarray(train_pred).reshape(-1)
    train_true = np.asarray(train_true).reshape(-1)
    test_pred = np.asarray(test_pred).reshape(-1)
    test_true = np.asarray(test_true).reshape(-1)

    train_std_ep = scale_uncertainty(train_epi, label_scaler)
    train_std_al = scale_uncertainty(train_alea, label_scaler)
    test_std_ep = scale_uncertainty(test_epi, label_scaler)
    test_std_al = scale_uncertainty(test_alea, label_scaler)

    train_std_total = np.sqrt(train_std_ep ** 2 + train_std_al ** 2)
    test_std_total = np.sqrt(test_std_ep ** 2 + test_std_al ** 2)

    metrics_rows = []
    for split_name, y_true, y_pred in (
        ("train", train_true, train_pred),
        ("test", test_true, test_pred),
    ):
        metrics_rows.append(
            {
                "seed": seed,
                "split": split_name,
                "n_samples": len(y_true),
                "MAE": mean_absolute_error(y_true, y_pred),
                "MSE": mean_squared_error(y_true, y_pred),
                "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "R2": r2_score(y_true, y_pred),
            }
        )

    artifacts = {
        "train_true": train_true,
        "train_pred": train_pred,
        "train_std_total": train_std_total,
        "train_std_al": train_std_al,
        "train_std_ep": train_std_ep,
        "test_true": test_true,
        "test_pred": test_pred,
        "test_std_total": test_std_total,
        "test_std_al": test_std_al,
        "test_std_ep": test_std_ep,
    }
    return metrics_rows, artifacts


def determine_best_epoch(dataset, config, device):
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    if BEST_EPOCH_PATH.exists():
        best_epoch = int(BEST_EPOCH_PATH.read_text().strip())
        print(f"Reusing saved best epoch count: {best_epoch}")
        return best_epoch

    print("Determining best epoch count from one validation split ...")
    set_all_seeds(SEED)
    dataset_train, dataset_val, dataset_test = split_dataset(dataset, seed=SEED, include_validation=True)
    _, _, _, loader_train, loader_val, _, _ = make_scaled_datasets_and_loaders(
        dataset_train,
        dataset_val,
        dataset_test,
        batch_size=config.get("batch_size", 100),
    )

    model = build_model(dataset, config, device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    fit_config = deepcopy(config)
    fit_config["save_path"] = str(OUT_PATH / "best_model_validation.pt")
    best_epoch = train_with_validation(
        model,
        loader_train,
        loader_val,
        optimizer,
        scheduler,
        device,
        fit_config,
    )
    BEST_EPOCH_PATH.write_text(f"{best_epoch}\n")
    print(f"Saved best epoch count to {BEST_EPOCH_PATH}: {best_epoch}")
    return best_epoch


def save_metrics(metrics_rows):
    metrics_by_seed_df = pd.DataFrame(metrics_rows)
    metrics_by_seed_df.to_csv(OUT_PATH / "metrics_by_seed.csv", index=False)
    metrics_summary_df = (
        metrics_by_seed_df.groupby("split", as_index=False)
        .agg(
            n_samples_mean=("n_samples", "mean"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            MSE_mean=("MSE", "mean"),
            MSE_std=("MSE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
        )
    )
    metrics_summary_df.to_csv(OUT_PATH / "metrics_summary_across_seeds.csv", index=False)
    return metrics_summary_df


def print_metrics(metrics_summary_df, best_epoch):
    train_summary = metrics_summary_df.loc[metrics_summary_df["split"] == "train"].iloc[0]
    test_summary = metrics_summary_df.loc[metrics_summary_df["split"] == "test"].iloc[0]

    print("\nTrain set performance across seeds:")
    print(
        f" MAE: {train_summary['MAE_mean']:.4f} +- {train_summary['MAE_std']:.4f},"
        f" MSE: {train_summary['MSE_mean']:.4f} +- {train_summary['MSE_std']:.4f},"
        f" R2: {train_summary['R2_mean']:.4f} +- {train_summary['R2_std']:.4f}"
    )
    print("Test set performance across seeds:")
    print(
        f" MAE: {test_summary['MAE_mean']:.4f} +- {test_summary['MAE_std']:.4f},"
        f" MSE: {test_summary['MSE_mean']:.4f} +- {test_summary['MSE_std']:.4f},"
        f" R2: {test_summary['R2_mean']:.4f} +- {test_summary['R2_std']:.4f}"
    )
    print(f"\nEpochs reused for evaluation: {best_epoch}")
    print(f"MC samples: {CONFIG.get('mc_dropout_samples', 50)}")
    print(f"Representative seed for plots: {SEED}\n")


def save_representative_plots(artifacts):
    for split_name in ("train", "test"):
        y_true = artifacts[f"{split_name}_true"]
        y_pred = artifacts[f"{split_name}_pred"]
        for uncertainty_type, key in (
            ("total", f"{split_name}_std_total"),
            ("epistemic", f"{split_name}_std_ep"),
            ("aleatoric", f"{split_name}_std_al"),
        ):
            title = f"Parity plot {split_name}-set ({SYMBOL})"
            plot_cv_predictions_hexbin(
                y_pred=y_pred,
                y_true=y_true,
                std=artifacts[key],
                title=title,
                uncertainty_type=uncertainty_type,
                save_path=OUT_PATH / f"{LABEL}_{split_name}_{uncertainty_type}_unc_hexbin.png",
                show=False,
            )


def save_confidence_plot(test_trials):
    plot_confidence_curve(
        regressor=None,
        df=None,
        features=None,
        label=None,
        save_path=OUT_PATH / f"confidence_curve_{LABEL}_all_samples.png",
        uncertainty_types=["total", "aleatoric", "epistemic"],
        plot_linear_fit=False,
        title=f"Confidence curve of GNN model with all samples ({SYMBOL})",
        metric="mae",
        step=5,
        min_remaining=10,
        show=False,
        y_test_arr=np.vstack(test_trials["y_true"]),
        y_pred_arr=np.vstack(test_trials["y_pred"]),
        uncertainty_arr_by_type={
            "total": np.vstack(test_trials["std_total"]),
            "aleatoric": np.vstack(test_trials["std_al"]),
            "epistemic": np.vstack(test_trials["std_ep"]),
        },
    )


def main():
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    dataset = get_dataset()
    best_epoch = determine_best_epoch(dataset, CONFIG, DEVICE)

    metrics_rows = []
    representative_artifacts = None
    test_trials = {
        "y_true": [],
        "y_pred": [],
        "std_total": [],
        "std_al": [],
        "std_ep": [],
    }

    print(f"[*] evaluating GNN across {len(EVAL_SEEDS)} random seeds ...")
    for seed in EVAL_SEEDS:
        print(f"[*] evaluating seed {seed} ...")
        seed_rows, artifacts = evaluate_split(
            dataset=dataset,
            config=CONFIG,
            device=DEVICE,
            seed=seed,
            epochs=best_epoch,
        )
        metrics_rows.extend(seed_rows)
        test_trials["y_true"].append(artifacts["test_true"])
        test_trials["y_pred"].append(artifacts["test_pred"])
        test_trials["std_total"].append(artifacts["test_std_total"])
        test_trials["std_al"].append(artifacts["test_std_al"])
        test_trials["std_ep"].append(artifacts["test_std_ep"])
        if seed == SEED:
            representative_artifacts = artifacts

    if representative_artifacts is None:
        raise RuntimeError(f"Representative artifacts for seed {SEED} were not captured.")

    metrics_summary_df = save_metrics(metrics_rows)
    print_metrics(metrics_summary_df, best_epoch)
    save_representative_plots(representative_artifacts)
    save_confidence_plot(test_trials)
    print(f"FINISHED TRAINING AND EVALUATION. All results saved to {OUT_PATH}")


torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(_GNN_PATH / "config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)
CONFIG.setdefault("mc_dropout_samples", 50)


if __name__ == "__main__":
    main()
