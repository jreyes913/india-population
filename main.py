from src.clean_data import *
from src.plot_series import *
from src.model import *
from src.plot_forecast import *

data_path = DataPath(
    water=Path(r".\data\raw\AQUASTAT Dissemination System.csv"),
    energy=Path(r".\data\raw\Statistical Review of World Energy Narrow format.csv"),
    population=Path(r".\data\raw\India-Population-Population-2026-09-25-22-10.csv")
)
data_group = load_data(data_path=data_path)
df = merge_data(data_group=data_group)
df.to_csv(proc_path, index=False)
plot_series(df, plot_path)
dataset = prepare_data(df)
forecast_df = create_prediction(df, dataset)
forecast_df.to_csv(forecast_path, index=False)
print(f"\nSaved {forecast_path}")
long_forecast_df = create_long_horizon_prediction(df, dataset.r_intrinsic)
long_forecast_df.to_csv(long_forecast_path, index=False)
print(f"\nSaved {long_forecast_path}")
plot_forecast(forecast_df=forecast_df, long_forecast_df=long_forecast_df, plot_path=forecast_plot_path)