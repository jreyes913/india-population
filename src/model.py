import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from .settings import *
import copy
from dataclasses import dataclass


CURRENT_TARGET_STRESS = 0.5     # calibration target: latest year's stress level
TARGET_STRESS_FRACTION = 0.95   # the "95% of carrying capacity" threshold to project toward
MAX_FORECAST_YEARS = 500        # hard cap on the long-horizon simulation so it can't loop forever

WATER_MIN_PER_CAPITA = None     # set by calibrate_thresholds(), or hardcode a literature value here
ENERGY_MIN_PER_CAPITA = None    # set by calibrate_thresholds(), or hardcode a literature value here


@dataclass
class DataSet:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    r_intrinsic: float   # carried alongside the split so every downstream function gets it from one place


def estimate_r_intrinsic(df: pd.DataFrame) -> float:
    """
    India's own peak historical growth rate, taken directly from the data
    instead of a hardcoded world figure. Uses the max, not the mean,
    because r in the logistic model represents the UNCONSTRAINED growth
    rate (Stress -> 0), which is best approximated by the fastest growth
    actually observed, not an average dragged down by later, more
    constrained years.

    NOTE: idxmax() picks whichever single year had the highest growth. If
    that's a noisy one-off (data artifact, census correction, etc.) rather
    than a genuine sustained peak, it'll bake an outlier into the whole
    model. Print/inspect the years around the peak before trusting this,
    or switch to a percentile (e.g. GrowthRate.quantile(0.95)) if the max
    looks like an outlier.
    """
    peak_row = df.loc[df["GrowthRate"].idxmax()]
    r = float(peak_row["GrowthRate"])
    print(f"\nr_intrinsic estimated from data: {r:.6f} (peak Year: {int(peak_row['Year'])})")
    return r


def calibrate_thresholds(df: pd.DataFrame, target_stress: float = CURRENT_TARGET_STRESS) -> None:
    global WATER_MIN_PER_CAPITA, ENERGY_MIN_PER_CAPITA
    last_row = df.sort_values("Year").iloc[-1]
    WATER_MIN_PER_CAPITA = target_stress * last_row["TotalRenewableWaterResourcesPerCapita"]
    ENERGY_MIN_PER_CAPITA = target_stress * last_row["TesGJPC"]

    print("=== Threshold calibration ===")
    print(f"Calibrated against Year {int(last_row['Year'])} "
          f"(W={last_row['TotalRenewableWaterResourcesPerCapita']:.3f}, "
          f"E={last_row['TesGJPC']:.3f}) at target Stress={target_stress}")
    print(f"WATER_MIN_PER_CAPITA  = {WATER_MIN_PER_CAPITA:.4f}")
    print(f"ENERGY_MIN_PER_CAPITA = {ENERGY_MIN_PER_CAPITA:.4f}")


def split_df(model_df: pd.DataFrame, r_intrinsic: float) -> DataSet:
    n = len(model_df)
    split_idx = int(np.floor(n * 0.8))

    train_df = model_df.iloc[:split_idx].copy()
    test_df = model_df.iloc[split_idx:].copy()

    print(f"Total usable rows: {n}")
    print(f"Train: {train_df['Year'].min()}-{train_df['Year'].max()} ({len(train_df)} rows)")
    print(f"Test : {test_df['Year'].min()}-{test_df['Year'].max()} ({len(test_df)} rows)")

    return DataSet(train_df, test_df, r_intrinsic)


def compute_carrying_capacity(df: pd.DataFrame) -> pd.DataFrame:
    if WATER_MIN_PER_CAPITA is None or ENERGY_MIN_PER_CAPITA is None:
        raise RuntimeError("Call calibrate_thresholds(df) before compute_carrying_capacity().")

    water_capacity = df["Population"] * (
        df["TotalRenewableWaterResourcesPerCapita"] / WATER_MIN_PER_CAPITA
    )
    energy_capacity = df["Population"] * (
        df["TesGJPC"] / ENERGY_MIN_PER_CAPITA
    )
    df["CarryingCapacity"] = np.minimum(water_capacity, energy_capacity)
    df["BindingResource"] = np.where(
        water_capacity <= energy_capacity, "water", "energy"
    )
    df["Stress"] = df["Population"] / df["CarryingCapacity"]
    return df


def print_stress_diagnostics(train_df: pd.DataFrame) -> None:
    corr_stress_year = train_df["Stress"].corr(train_df["Year"])
    corr_stress_growth = train_df["Stress"].corr(train_df["GrowthRate"])
    corr_growth_year = train_df["GrowthRate"].corr(train_df["Year"])

    print("\n=== Diagnostics (train set, informational only -- nothing is fit) ===")
    print(f"corr(Stress, Year)        : {corr_stress_year:.4f}")
    print(f"corr(Stress, GrowthRate)  : {corr_stress_growth:.4f}")
    print(f"corr(GrowthRate, Year)    : {corr_growth_year:.4f}")
    if abs(corr_stress_year) > 0.9:
        print("NOTE: Stress remains nearly collinear with Year, confirming "
              "r cannot be reliably estimated via regression -- this is "
              "why r_intrinsic is taken as a fixed peak-observed value "
              "instead of fit.")


def prepare_data(df: pd.DataFrame) -> DataSet:
    _df = copy.deepcopy(df)
    _df = _df.sort_values("Year").reset_index(drop=True)
    _df["GrowthRate"] = _df["Population"].pct_change().shift(-1)

    r_intrinsic = estimate_r_intrinsic(_df)
    calibrate_thresholds(_df)
    _df = compute_carrying_capacity(_df)
    model_df = _df.dropna(subset=["GrowthRate"]).reset_index(drop=True)

    dataset = split_df(model_df, r_intrinsic)
    print_stress_diagnostics(dataset.train_df)

    print(f"\nUsing r_intrinsic = {r_intrinsic:.6f} (estimated from data, not hardcoded)")
    return dataset


def create_prediction(df: pd.DataFrame, dataset: DataSet) -> pd.DataFrame:
    r_intrinsic = dataset.r_intrinsic  # pulled from the dataset, not a bare module-level name

    y_test = dataset.test_df["GrowthRate"].values
    y_pred_test = r_intrinsic * (1.0 - dataset.test_df["Stress"].values)

    print("\n=== r_intrinsic growth-rate predictions vs. actual (test years) ===")
    print(f"R^2  : {r2_score(y_test, y_pred_test):.4f}")
    print(f"MAE  : {mean_absolute_error(y_test, y_pred_test):.6f}")

    last_train_row = dataset.train_df.iloc[-1]
    pop = float(last_train_row["Population"])

    threshold_year = None
    threshold_stress = None

    forecast_rows = []
    for _, row in dataset.test_df.iterrows():
        water = row["TotalRenewableWaterResourcesPerCapita"]
        energy = row["TesGJPC"]

        water_capacity = pop * (water / WATER_MIN_PER_CAPITA)
        energy_capacity = pop * (energy / ENERGY_MIN_PER_CAPITA)
        K_t = min(water_capacity, energy_capacity)
        binding = "water" if water_capacity <= energy_capacity else "energy"
        stress = pop / K_t

        g_pred = r_intrinsic * (1.0 - stress)
        pop_next = pop * (1 + g_pred)
        stress_next = pop_next / K_t  # against same-year K_t for reporting

        if threshold_year is None and stress_next >= TARGET_STRESS_FRACTION:
            threshold_year = int(row["Year"]) + 1
            threshold_stress = stress_next

        forecast_rows.append({
            "Year": int(row["Year"]) + 1,
            "CarryingCapacity": K_t,
            "BindingResource": binding,
            "Stress": stress,
            "PredictedGrowthRate": g_pred,
            "PredictedPopulation": pop_next,
            "ActualPopulation": float(df.loc[df["Year"] == row["Year"] + 1, "Population"].iloc[0])
                                if (df["Year"] == row["Year"] + 1).any() else np.nan,
        })
        pop = pop_next

    forecast_df = pd.DataFrame(forecast_rows)
    forecast_df["AbsPctError"] = (
        (forecast_df["PredictedPopulation"] - forecast_df["ActualPopulation"]).abs()
        / forecast_df["ActualPopulation"]
    )

    print("\n=== Multi-step (compounding) population simulation on test years ===")
    print(forecast_df.to_string(index=False))

    valid = forecast_df.dropna(subset=["ActualPopulation"])
    if len(valid) > 0:
        print(f"\nMean Absolute Percentage Error over test years: "
              f"{mean_absolute_percentage_error(valid['ActualPopulation'], valid['PredictedPopulation'])*100:.3f}%")

    if threshold_year is not None:
        print(f"\nProjected to reach {TARGET_STRESS_FRACTION*100:.0f}% of carrying "
              f"capacity in {threshold_year} (Stress={threshold_stress:.4f}). "
              f"NOTE: within the test window given -- extend the test/forecast "
              f"range if this threshold isn't crossed here.")
    else:
        print(f"\n{TARGET_STRESS_FRACTION*100:.0f}% of carrying capacity not reached "
              f"within the given forecast window.")

    return forecast_df


def project_resource_trends(df: pd.DataFrame) -> tuple[float, float, pd.Series]:
    #water_model = LinearRegression().fit(df[["Year"]], df["TotalRenewableWaterResourcesPerCapita"])
    energy_model = LinearRegression().fit(df[["Year"]], df["TesGJPC"])
    slope_water = -12.0  # m3/inhab/year per year (source: CWC "Reassessment of Water Availability in India using Space Inputs, 2019")
    #slope_water = float(water_model.coef_[0])
    slope_energy = float(energy_model.coef_[0])

    last_row = df.sort_values("Year").iloc[-1]

    print("\n=== Resource trend extrapolation (slope fit on full history, anchored to last actual year) ===")
    print(f"Water per capita : slope {slope_water:+.4f} m3/inhab/year per year, "
          f"anchored at Year {int(last_row['Year'])} = {last_row['TotalRenewableWaterResourcesPerCapita']:.3f}")
    print(f"Energy per capita: slope {slope_energy:+.4f} GJ/inhab/year per year, "
          f"anchored at Year {int(last_row['Year'])} = {last_row['TesGJPC']:.3f}")

    return slope_water, slope_energy, last_row


def create_long_horizon_prediction(df: pd.DataFrame, r_intrinsic: float) -> pd.DataFrame:
    if WATER_MIN_PER_CAPITA is None or ENERGY_MIN_PER_CAPITA is None:
        raise RuntimeError("Call calibrate_thresholds(df) (e.g. via prepare_data) before this.")

    slope_water, slope_energy, last_row = project_resource_trends(df)

    last_year = int(last_row["Year"])
    last_water = float(last_row["TotalRenewableWaterResourcesPerCapita"])
    last_energy = float(last_row["TesGJPC"])
    pop = float(last_row["Population"])

    threshold_year = None
    threshold_stress = None
    stopped_reason = None

    rows = []
    for step in range(1, MAX_FORECAST_YEARS + 1):
        year = last_year + step

        water = last_water + slope_water * step
        energy = last_energy + slope_energy * step

        if water <= 0 or energy <= 0:
            stopped_reason = (
                f"Linear trend projected a non-positive per-capita resource value "
                f"at Year {year} (Water={water:.3f}, Energy={energy:.3f}) -- linear "
                f"extrapolation isn't physically meaningful past this point. Stopping."
            )
            break

        water_capacity = pop * (water / WATER_MIN_PER_CAPITA)
        energy_capacity = pop * (energy / ENERGY_MIN_PER_CAPITA)
        K_t = min(water_capacity, energy_capacity)
        binding = "water" if water_capacity <= energy_capacity else "energy"
        stress = pop / K_t

        g_pred = r_intrinsic * (1.0 - stress)
        pop_next = pop * (1 + g_pred)
        stress_next = pop_next / K_t

        rows.append({
            "Year": year,
            "ProjectedWaterPC": water,
            "ProjectedEnergyPC": energy,
            "CarryingCapacity": K_t,
            "BindingResource": binding,
            "Stress": stress,
            "PredictedGrowthRate": g_pred,
            "ProjectedPopulation": pop_next,
        })

        if threshold_year is None and stress_next >= TARGET_STRESS_FRACTION:
            threshold_year = year
            threshold_stress = stress_next
            pop = pop_next
            break

        pop = pop_next

    long_forecast_df = pd.DataFrame(rows)

    print(f"\n=== Long-horizon simulation ({len(rows)} years projected) ===")
    print(long_forecast_df.to_string(index=False))

    if threshold_year is not None:
        print(f"\nProjected to reach {TARGET_STRESS_FRACTION*100:.0f}% of carrying "
              f"capacity in {threshold_year} (Stress={threshold_stress:.4f}), "
              f"{threshold_year - int(last_row['Year'])} years past the last "
              f"historical year ({int(last_row['Year'])}).")
    elif stopped_reason:
        print(f"\n{TARGET_STRESS_FRACTION*100:.0f}% of carrying capacity not reached. "
              f"{stopped_reason}")
    else:
        print(f"\n{TARGET_STRESS_FRACTION*100:.0f}% of carrying capacity not reached "
              f"within {MAX_FORECAST_YEARS} projected years. Under this linear "
              f"resource-trend assumption, Stress is trending toward "
              f"{long_forecast_df['Stress'].iloc[-1]:.4f} by Year "
              f"{int(long_forecast_df['Year'].iloc[-1])} rather than rising toward 1.0 -- "
              f"i.e. the projected energy/water trend outpaces population growth over "
              f"this horizon under r_intrinsic={r_intrinsic:.6f}.")

    return long_forecast_df


def main() -> None:
    df = pd.read_csv(proc_path)
    dataset = prepare_data(df)
    forecast_df = create_prediction(df, dataset)
    forecast_df.to_csv(forecast_path, index=False)
    print(f"\nSaved {forecast_path}")
    long_forecast_df = create_long_horizon_prediction(df, dataset.r_intrinsic)
    long_forecast_df.to_csv(long_forecast_path, index=False)
    print(f"\nSaved {long_forecast_path}")


if __name__ == "__main__":
    main()