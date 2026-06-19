#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DEFAULT_CSV = REPO_DIR / "day-3-flight-2-logs-hovering-deep/20260618_074356_rfd_doa_gps_matched.csv"

ARRAY_MAX = 14.9
INPUT_BITS = 8
DOA_PREFIX = "doa"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def unsigned_scale(maximum: float, bits: int) -> float:
    return maximum / float(2**bits - 1)


def quantize_unsigned(x: np.ndarray, scale: float, bits: int) -> np.ndarray:
    return np.rint(x / scale).clip(0, 2**bits - 1).astype(np.int64)


def read_scalar(path: Path, dtype=float):
    values = pd.read_csv(path, header=None).to_numpy().reshape(-1)
    if len(values) != 1:
        raise ValueError(f"{path} must contain exactly one value")
    return dtype(values[0])


def read_column(path: Path) -> np.ndarray:
    values = pd.read_csv(path, header=None).to_numpy().reshape(-1)
    return np.rint(values).astype(np.int64)


def infer_label(raw_label: object) -> str | None:
    label = str(raw_label).strip()
    if label in {"A", "B"}:
        return label
    return None


def parse_timestamp_offsets_seconds(series: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(series.astype(str), format="%H:%M:%S.%f", errors="coerce")
    if timestamps.notna().any():
        first_timestamp = timestamps[timestamps.notna()].iloc[0]
        return (timestamps - first_timestamp).dt.total_seconds()
    return pd.Series(np.nan, index=series.index)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render integer SLP predictions over DOA polar maps."
    )
    parser.add_argument(
        "-c",
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV,
        help="Input CSV containing true labels and doa_0..doa_359 columns.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output MP4 path. Defaults to Kraken_Binaries/integer_slp_polar_predictions.mp4.",
    )
    parser.add_argument(
        "--prediction-csv",
        type=Path,
        help="Optional prediction CSV output path.",
    )
    parser.add_argument("--doa-prefix", default=DOA_PREFIX, help="DOA column prefix.")
    parser.add_argument("--label-column", default="transmission", help="True label column.")
    parser.add_argument("--seconds-per-map", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=2.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    csv_path = args.csv_path.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else SCRIPT_DIR / "integer_slp_polar_predictions.mp4"
    )
    prediction_csv = (
        args.prediction_csv.expanduser().resolve()
        if args.prediction_csv
        else output_path.with_name(f"{output_path.stem}_predictions.csv")
    )

    if args.seconds_per_map <= 0:
        raise ValueError("--seconds-per-map must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    weights = read_column(SCRIPT_DIR / "w1.csv")
    bias = read_scalar(SCRIPT_DIR / "b1.csv", int)
    alpha = read_scalar(SCRIPT_DIR / "m1.csv", int)
    beta = read_scalar(SCRIPT_DIR / "e1.csv", int)
    sy = read_scalar(SCRIPT_DIR / "sy.csv", float)

    doa_columns = [f"{args.doa_prefix}_{idx}" for idx in range(360)]
    df = pd.read_csv(csv_path)
    missing = sorted({args.label_column, *doa_columns} - set(df.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
    if len(weights) != len(doa_columns):
        raise ValueError(f"Expected {len(doa_columns)} weights, got {len(weights)}")

    df = df.copy()
    df["true_label"] = df[args.label_column].map(infer_label)
    df = df[df["true_label"].isin(["A", "B"])].reset_index(drop=True)
    if df.empty:
        raise ValueError("No A/B rows found.")

    x_real = df[doa_columns].to_numpy(dtype=float)
    sx = unsigned_scale(ARRAY_MAX, INPUT_BITS)
    xq = quantize_unsigned(x_real, sx, INPUT_BITS)

      
    mac = xq @ weights
    integer_accumulator = mac + bias
    ratio = alpha / float(2**beta)
    integer_output = np.rint(integer_accumulator * ratio).astype(np.int64)
    logits = integer_output.astype(float) * sy
    output_activation_b = sigmoid(logits)
    predicted_label = np.where(output_activation_b >= 0.5, "B", "A")

    pred_df = pd.DataFrame(
        {
            "row": np.arange(len(df), dtype=int),
            "true_label": df["true_label"],
            "predicted_label": predicted_label,
            "integer_mac": mac,
            "integer_bias": bias,
            "integer_accumulator": integer_accumulator,
            "alpha": alpha,
            "beta": beta,
            "integer_output": integer_output,
            "sy": sy,
            "logit": logits,
            "output_activation_B": output_activation_b,
        }
    )
    for column in ("transmission_timestamp", "doa_timestamp"):
        if column in df.columns:
            pred_df[column] = df[column]
    pred_df.to_csv(prediction_csv, index=False)

    correct = int((pred_df["predicted_label"] == pred_df["true_label"]).sum())
    total = len(pred_df)

    if "transmission_timestamp" in df.columns:
        df["message_time_s"] = parse_timestamp_offsets_seconds(df["transmission_timestamp"])
    else:
        df["message_time_s"] = np.nan

    radial_limit = float(np.nanmax(x_real) * 1.08)
    if not math.isfinite(radial_limit) or radial_limit <= 0:
        radial_limit = 1.0

    angles = np.deg2rad(np.r_[np.arange(360), 0])
    repeated_frames = int(math.ceil(args.seconds_per_map * args.fps))
    total_frames = len(df) * repeated_frames

    figure = plt.figure(figsize=(8, 9), facecolor="white")
    axis = figure.add_subplot(111, projection="polar")
    axis.set_position([0.09, 0.25, 0.82, 0.62])
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)
    axis.set_ylim(0, radial_limit)
    axis.grid(alpha=0.35)

    (line,) = axis.plot([], [], color="#2457a6", linewidth=2.2)
    pred_text = figure.text(0.5, 0.145, "", ha="center", va="center", fontsize=30, fontweight="bold")
    true_text = figure.text(0.5, 0.095, "", ha="center", va="center", fontsize=28)
    meta_text = figure.text(0.5, 0.045, "", ha="center", va="center", fontsize=16, color="#444444")

    writer = FFMpegWriter(
        fps=args.fps,
        metadata={"title": "Integer SLP DOA predictions"},
        bitrate=1800,
    )

    print(f"Accuracy={correct}/{total} = {correct / total:.6f}")
    print(f"Rendering {len(df)} maps ({total_frames} frames) to {output_path}")
    with writer.saving(figure, str(output_path), dpi=150):
        for frame_idx in range(total_frames):
            row_idx = frame_idx // repeated_frames
            values = np.r_[x_real[row_idx], x_real[row_idx, 0]]
            line.set_data(angles, values)

            true_label = pred_df.loc[row_idx, "true_label"]
            pred_label = pred_df.loc[row_idx, "predicted_label"]
            is_correct = true_label == pred_label

            axis.set_title(f"DOA Polar Map\n{row_idx + 1} / {len(df)}", fontsize=14, pad=12)
            pred_text.set_text(f"Predicted: {pred_label}")
            pred_text.set_color("#1b7f3a" if is_correct else "#b8322a")
            true_text.set_text(f"True: {true_label}")

            if pd.notna(df.loc[row_idx, "message_time_s"]):
                time_text = f"t={df.loc[row_idx, 'message_time_s']:.1f}s"
            else:
                time_text = "t=n/a"
            meta_text.set_text(
                f"{time_text} | y_int={integer_output[row_idx]} | "
                f"logit={logits[row_idx]:.4f} | sigmoid={output_activation_b[row_idx]:.4f}"
            )

            writer.grab_frame()

    print(output_path)
    print(prediction_csv)


if __name__ == "__main__":
    main()
