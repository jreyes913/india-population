# India Population vs. Water/Energy Carrying Capacity

Models India's population growth against a resource-constrained carrying capacity (water + energy availability) using a logistic growth framework, then projects when population reaches a target fraction of that capacity.

## Equations

**1. Logistic growth (continuous form)**

$$\frac{dP}{dt} = rP\left(1 - \frac{P}{K}\right)$$

| Variable | Description |
|---|---|
| $P$ | Population |
| $r$ | Intrinsic growth rate — the unconstrained growth rate as Stress $\to 0$ |
| $K$ | Carrying capacity — the population level resources can sustain |
| $t$ | Year |

**2. Discretized growth rate, reparametrized by Stress**

$$g_t = \frac{P_{t+1} - P_t}{P_t} \approx r\left(1 - \text{Stress}_t\right)$$

| Variable | Description |
|---|---|
| $g_t$ | Year-over-year fractional population growth rate |
| $\text{Stress}_t$ | $P_t / K_t$ — how close population is to that year's carrying capacity (0 = no pressure, 1 = at capacity) |

**3. Carrying capacity — Liebig's Law of the Minimum**

Population is capped by whichever resource (water or energy) runs out first, not an average of both.

$$K_t = \min\left(K_t^{W},\ K_t^{E}\right)$$

$$K_t^{W} = P_t \cdot \frac{W_t}{w_{\min}}, \qquad K_t^{E} = P_t \cdot \frac{E_t}{e_{\min}}$$

| Variable | Description |
|---|---|
| $K_t^{W}$ | Water-only carrying capacity in year $t$ |
| $K_t^{E}$ | Energy-only carrying capacity in year $t$ |
| $W_t$ | Water per capita available in year $t$ (m³/person/year) |
| $E_t$ | Energy per capita available in year $t$ (GJ/person/year) |
| $w_{\min}$ | Minimum sustainable water per capita — the scarcity threshold |
| $e_{\min}$ | Minimum sustainable energy per capita — the scarcity threshold |

Since $K_t = \min(\cdot)$, $\text{Stress}_t = P_t/K_t$ simplifies to a form independent of population size:

$$\text{Stress}_t = \max\left(\frac{w_{\min}}{W_t},\ \frac{e_{\min}}{E_t}\right)$$

**4. Threshold calibration**

$w_{\min}$ and $e_{\min}$ are not fit from data — they're calibrated so the most recent actual year lands at a chosen target stress level (data-unit issues in the source columns made literature thresholds unreliable; see project notes):

$$w_{\min} = s_{\text{target}} \cdot W_{\text{last year}}, \qquad e_{\min} = s_{\text{target}} \cdot E_{\text{last year}}$$

where $s_{\text{target}}$ is the calibration target (e.g. 0.5).

**5. Intrinsic growth rate $r$**

Estimated directly from the data as the peak observed year-over-year growth rate (the closest available proxy for unconstrained growth), rather than fit by regression — Stress and Year are too collinear in a single-country time series for $r$ to be reliably estimated that way.

$$r = \max_t \left(g_t\right) \text{ over all historical years}$$

**6. Resource trend extrapolation (anchored)**

Future water/energy per capita are extrapolated linearly from the last actual observation, not from a raw regression line's own intercept (which would create a discontinuity at the history/projection boundary):

$$W_t = W_{\text{last year}} + m_W (t - t_{\text{last}}), \qquad E_t = E_{\text{last year}} + m_E (t - t_{\text{last}})$$

| Variable | Description |
|---|---|
| $m_W$, $m_E$ | Linear trend rate of change in per-capita water/energy, fit vs. Year |

**7. Forward simulation**

$$P_{t+1} = P_t \left(1 + g_t\right), \qquad g_t = r\left(1 - \text{Stress}_t\right)$$

Iterated year-by-year until $\text{Stress}_t$ crosses the target fraction (e.g. 0.95) of carrying capacity, or a max forecast horizon is reached.

## Folder structure

```
population/
├── main.py                     # entry point: clean data → build/plot model
├── src/
│   ├── __init__.py
│   ├── settings.py             # paths, shared constants
│   ├── clean_data.py           # load & merge raw water/energy/population sources
│   ├── model.py                # carrying-capacity model, calibration, simulation
│   ├── plot_series.py          # plots raw historical series
│   └── plot_forecast.py        # plots test + long-horizon forecast
├── data/
│   ├── raw/                    # source CSVs (AQUASTAT, Statistical Review of World Energy, population)
│   ├── processed/               # merged/cleaned dataset
│   └── forecast/                # test_forecast.csv, long_horizon_forecast.csv
└── figs/                        # series.png, forecast.png
```

## Requirements

- Python 3.10+
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

Install:
```
pip install numpy pandas scikit-learn matplotlib
```

## Usage

```
python -m main
```

Runs the full pipeline: cleans and merges raw data, fits/calibrates the model, runs the test-year and long-horizon simulations, and writes forecast CSVs and plots.