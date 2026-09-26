from pathlib import Path

Year="Year"
TotalRenewableWaterResourcesPerCapita="TotalRenewableWaterResourcesPerCapita"
WaterUnit="WaterUnit"
TesGJPC="TesGJPC"
EnergyUnit="EnergyUnit"
Population="Population"

Vars = [
    Population,
    TotalRenewableWaterResourcesPerCapita,
    TesGJPC
]

proc_path = Path(r".\data\processed\india.csv")
plot_path = Path(r".\figs\series.png")
forecast_path = Path(r".\data\forecast\test_forecast.csv")
long_forecast_path = Path(r".\data\forecast\long_horizon_forecast.csv")
forecast_plot_path = Path(r".\figs\forecast.png") 