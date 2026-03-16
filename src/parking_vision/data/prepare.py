from __future__ import annotations

from pathlib import Path

from parking_vision.config import load_yaml
from parking_vision.data.adapters.cnrpark_ext import CNRParkEXTAdapter
from parking_vision.data.adapters.pklot import PKLotAdapter
from parking_vision.data.splits import build_splits
from parking_vision.utils.io import ensure_dir, save_dataframe


def get_adapter(cfg: dict):
    name = cfg["dataset"]["name"].lower()
    if name == "pklot":
        return PKLotAdapter(cfg)
    if name == "cnrpark_ext":
        return CNRParkEXTAdapter(cfg)
    raise ValueError(f"Unsupported dataset adapter: {name}")


def prepare_dataset(config_path: str) -> dict:
    cfg = load_yaml(config_path)
    adapter = get_adapter(cfg)
    adapter.download()
    manifest = adapter.build_manifest()

    out_manifest = Path(cfg["dataset"]["output_manifest"])
    save_dataframe(manifest, out_manifest)

    train_df, val_df, test_df = build_splits(
        manifest,
        train_size=cfg["dataset"]["train_size"],
        val_size=cfg["dataset"]["val_size"],
        test_size=cfg["dataset"]["test_size"],
        seed=cfg["dataset"]["random_seed"],
        split_policy=cfg["dataset"].get("split_policy", "group"),
    )
    split_dir = ensure_dir(cfg["dataset"]["output_split_dir"])
    save_dataframe(train_df, split_dir / "train.csv")
    save_dataframe(val_df, split_dir / "val.csv")
    save_dataframe(test_df, split_dir / "test.csv")

    return {
        "manifest_path": str(out_manifest),
        "train_path": str(split_dir / "train.csv"),
        "val_path": str(split_dir / "val.csv"),
        "test_path": str(split_dir / "test.csv"),
        "num_samples": len(manifest),
    }
