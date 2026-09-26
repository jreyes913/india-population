import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from .settings import *

def plot_series(df: pd.DataFrame, plot_path: Path) -> None:
    fig, axes = plt.subplots(figsize=(12,9), nrows=len(Vars), ncols=1)

    for idx, var in enumerate(Vars):
        axes[idx].plot(df[Year], df[var], c="k", marker=".", label=var)
        axes[idx].legend()
        axes[idx].grid(True)
        axes[idx].set_ylabel(var)
    plt.savefig(plot_path, dpi=300)

def main() -> None:
    df = pd.read_csv(proc_path)
    plot_series(df=df, plot_path=plot_path)

if __name__ == "__main__":
    main()