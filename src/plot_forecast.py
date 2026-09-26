import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from .settings import *
from .model import TARGET_STRESS_FRACTION

ForecastVars = ["PredictedPopulation", "CarryingCapacity", "Stress"]

def _combine_forecasts(forecast_df: pd.DataFrame, long_forecast_df: pd.DataFrame) -> pd.DataFrame:
    long_forecast_df = long_forecast_df.rename(columns={"ProjectedPopulation": "PredictedPopulation"})
    cols = [Year, "PredictedPopulation", "CarryingCapacity", "Stress"]
    combined = pd.concat(
        [forecast_df[cols], long_forecast_df[cols]],
        ignore_index=True,
    ).sort_values(Year).reset_index(drop=True)
    return combined

def plot_forecast(forecast_df: pd.DataFrame, long_forecast_df: pd.DataFrame, plot_path: Path) -> None:
    combined = _combine_forecasts(forecast_df, long_forecast_df)
    boundary_year = forecast_df[Year].max()  # last test year -- everything past this is pure projection, not evaluated against actual data

    fig, axes = plt.subplots(figsize=(12, 9), nrows=len(ForecastVars), ncols=1)

    for idx, var in enumerate(ForecastVars):
        axes[idx].plot(combined[Year], combined[var], c="k", marker=".", label=var)
        axes[idx].axvline(boundary_year, color="gray", linestyle=":", label="test / long-horizon boundary")
        if var == "Stress":
            axes[idx].axhline(TARGET_STRESS_FRACTION, color="r", linestyle="--",
                               label=f"{TARGET_STRESS_FRACTION*100:.0f}% threshold")
        axes[idx].legend()
        axes[idx].grid(True)
        axes[idx].set_ylabel(var)
    plt.savefig(plot_path, dpi=300)

def main() -> None:
    forecast_df = pd.read_csv(forecast_path)
    long_forecast_df = pd.read_csv(long_forecast_path)
    # forecast_plot_path: add this to settings.py alongside plot_path if it isn't there yet
    plot_forecast(forecast_df=forecast_df, long_forecast_df=long_forecast_df, plot_path=forecast_plot_path)

if __name__ == "__main__":
    main()