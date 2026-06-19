#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
from astra_interface import AstraInterface

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DEFAULT_CSV ="/home/astra/Kraken_ASTRA/20260618_074356_rfd_doa_gps_matched.csv"

INSTRUCTION_MEMORY = "/home/astra/Kraken_ASTRA/instr_mem.bin"
PARAMETER_MEMORY = "/home/astra/Kraken_ASTRA/param_mem.bin"

ARRAY_MAX = 14.9
INPUT_BITS = 8
OUTPUT_BITS = 8
DOA_PREFIX = "doa"
NONINTERACTIVE_BACKENDS = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}

class AstraSerialProcessor:
    def __init__(self, port, baud, instruction_file, parameter_file, upload_config=True):
        self.astra = AstraInterface(port, baud)

        if upload_config:
            self.astra.write_instcfg(instruction_file)
            self.astra.write_paramcfg(parameter_file)

    def processor_run(self, activation_data, outputLength=9, startAddress=233, interrupt_timeout=10.0):
        return self.astra.run_activation(
            activation_data,
            read_size=outputLength,
            read_addr=startAddress,
            timeout=interrupt_timeout,
        )

    def close(self):
        self.astra.close()

def configure_matplotlib_backend() -> None:
    early_parser = argparse.ArgumentParser(add_help=False)
    early_parser.add_argument("--live", action=argparse.BooleanOptionalAction, default=True)
    early_parser.add_argument("--backend")
    early_args, _ = early_parser.parse_known_args()

    cache_dir = Path("/tmp") / f"matplotlib-{os.getuid()}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    requested_backend = early_args.backend
    if requested_backend:
        matplotlib.use(requested_backend, force=True)
        return

    if not early_args.live:
        return
    if os.environ.get("MPLBACKEND"):
        return
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return
    if matplotlib.get_backend().lower() not in NONINTERACTIVE_BACKENDS:
        return

    for backend in ("TkAgg", "QtAgg", "Qt5Agg"):
        try:
            matplotlib.use(backend, force=True)
            return
        except Exception:
            continue


configure_matplotlib_backend()

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def unsigned_scale(maximum: float, bits: int) -> float:
    return maximum / float(2**bits - 1)


def quantize_unsigned(x: np.ndarray, scale: float, bits: int) -> np.ndarray:
    return np.rint(x / scale).clip(0, 2**bits - 1).astype(np.int64)


def unsigned_bounds(bits: int) -> tuple[int, int]:
    return 0, 2**bits - 1


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
    parser.add_argument(
        "--backend",
        help="Matplotlib backend to use for live rendering, e.g. TkAgg or QtAgg.",
    )
    parser.add_argument(
        "--live",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a live matplotlib preview while frames are rendered.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Only render the live preview and prediction CSV; do not write an MP4.",
    )
    return parser

def signed_to_u8(values):
    return [int(x) & 0xff for x in values]

# def u8_to_signed(values):
#     return [int(x) - 256 if int(x) > 127 else int(x) for x in values]

def main() -> None:
    astra_processor = AstraSerialProcessor(
            port="/dev/ttyACM0",
            baud=115200,
            instruction_file=INSTRUCTION_MEMORY,
            parameter_file=PARAMETER_MEMORY,
        )

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
    if args.no_video and not args.live:
        raise ValueError("--no-video requires live rendering; remove --no-live")

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
    ratio = alpha / float(2**beta)

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

    backend = plt.get_backend().lower()
    live_render = args.live and backend not in NONINTERACTIVE_BACKENDS
    if args.live and not live_render:
        raise RuntimeError(
            f"Live preview needs an interactive matplotlib backend, but matplotlib is using "
            f"{plt.get_backend()!r}. Install one of these and run again:\n"
            "  sudo apt install python3-tk\n"
            "  pip install PyQt6\n"
            "Then run with --backend TkAgg or --backend QtAgg if matplotlib still defaults to Agg. "
            "Use --no-live for MP4-only rendering."
        )
    if args.no_video and not live_render:
        raise RuntimeError("Cannot use --no-video without an interactive matplotlib backend.")
    if live_render:
        print(f"Live preview using matplotlib backend: {plt.get_backend()}")
        plt.ion()
        if figure.canvas.manager is not None:
            figure.canvas.manager.set_window_title("Integer SLP DOA live rendering")
        plt.show(block=False)

    writer = None
    if not args.no_video:
        writer = FFMpegWriter(
            fps=args.fps,
            metadata={"title": "Integer SLP DOA predictions"},
            bitrate=1800,
        )

    if writer is not None:
        print(f"Rendering {len(df)} maps ({total_frames} frames) to {output_path}")
    else:
        print(f"Live rendering {len(df)} maps ({total_frames} frames); MP4 output disabled")
    prediction_rows: list[dict[str, object]] = []
    correct = 0
    save_context = writer.saving(figure, str(output_path), dpi=150) if writer else nullcontext()
    output_qmin, output_qmax = unsigned_bounds(OUTPUT_BITS)
    with save_context:
        for row_idx, row in df.iterrows():
            row_values = x_real[row_idx]
            xq = quantize_unsigned(row_values, sx, INPUT_BITS)
            xq = signed_to_u8(xq.flatten())
            output_data = astra_processor.processor_run(
                xq,
                startAddress=233,
                outputLength=1
            )
            # mac = int(xq @ weights)
            # integer_accumulator = mac + bias
            # scaled_output = integer_accumulator * ratio
            # integer_output = int(np.clip(round(scaled_output), output_qmin, output_qmax))
            logit = output_data[0] * sy
            output_activation_b = float(sigmoid(np.array([logit]))[0])
            pred_label = "B" if output_activation_b >= 0.5 else "A"
            true_label = row["true_label"]
            is_correct = true_label == pred_label
            correct += int(is_correct)

            prediction_row = {
                "row": row_idx,
                "true_label": true_label,
                "predicted_label": pred_label,
                "integer_bias": bias,
                "alpha": alpha,
                "beta": beta,
                "sy": sy,
                "logit": logit,
                "output_activation_B": output_activation_b,
            }
            for column in ("transmission_timestamp", "doa_timestamp"):
                if column in df.columns:
                    prediction_row[column] = row[column]
            prediction_rows.append(prediction_row)

            values = np.r_[row_values, row_values[0]]
            line.set_data(angles, values)
            if pd.notna(row["message_time_s"]):
                time_text = f"t={row['message_time_s']:.1f}s"
            else:
                time_text = "t=n/a"

            for _ in range(repeated_frames):
                axis.set_title(f"DOA Polar Map\n{row_idx + 1} / {len(df)}", fontsize=14, pad=12)
                pred_text.set_text(f"Predicted: {pred_label}")
                pred_text.set_color("#1b7f3a" if is_correct else "#b8322a")
                true_text.set_text(f"True: {true_label}")
                meta_text.set_text(
                    f"{time_text} | y_int={output_data[0]} | "
                    f"logit={logit:.4f} | sigmoid={output_activation_b:.4f}"
                )
                if live_render:
                    figure.canvas.draw_idle()
                    figure.canvas.flush_events()
                    plt.pause(max(1.0 / args.fps, 0.001))
                if writer is not None:
                    writer.grab_frame()

            print(
                f"row={row_idx} predicted={pred_label} true={true_label} "
                f"y_int={output_data[0]} sigmoid={output_activation_b:.6f}"
            )

    pred_df = pd.DataFrame(prediction_rows)
    pred_df.to_csv(prediction_csv, index=False)
    total = len(pred_df)
    print(f"Accuracy={correct}/{total} = {correct / total:.6f}")

    if writer is not None:
        print(output_path)
    print(prediction_csv)


if __name__ == "__main__":
    main()
