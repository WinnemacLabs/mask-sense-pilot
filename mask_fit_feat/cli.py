from __future__ import annotations

import argparse
import pandas as pd

from .io import load_trial
from .preprocess import bandpass
from .breath import detect_breath_cycles
from .features import breath_time_features
from .models import train_random_forest


def main() -> None:
    parser = argparse.ArgumentParser(prog="mask-fit-run")
    sub = parser.add_subparsers(dest="cmd")

    sub_load = sub.add_parser("load", help="Load trial CSV")
    sub_load.add_argument("path")

    sub_pre = sub.add_parser("preprocess", help="Bandpass filter Pa_Global")
    sub_pre.add_argument("path")

    sub_ex = sub.add_parser("extract", help="Run full extraction")
    sub_ex.add_argument("path")

    sub_train = sub.add_parser("train", help="Train model")
    sub_train.add_argument("features")
    sub_train.add_argument("labels")
    sub_train.add_argument("groups")

    args = parser.parse_args()

    if args.cmd == "load":
        data = load_trial(args.path)
        print("Channels:", list(data))
    elif args.cmd == "preprocess":
        data = load_trial(args.path)
        fs = 1000.0
        filt = bandpass(data["Pa_Global"], fs)
        print(f"Filtered length: {len(filt)}")
    elif args.cmd == "extract":
        data = load_trial(args.path)
        cycles = detect_breath_cycles(data["Pa_Global"])
        feats = breath_time_features(data["Pa_Global"], cycles)
        print(feats.head())
    elif args.cmd == "train":
        X = pd.read_csv(args.features, index_col=0)
        y = pd.read_csv(args.labels, index_col=0, squeeze=True)
        groups = pd.read_csv(args.groups, index_col=0, squeeze=True)
        model, res = train_random_forest(X, y, groups)
        print(res)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
