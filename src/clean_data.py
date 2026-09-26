import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import copy

@dataclass
class DataPath:
    water: Path
    energy: Path
    population: Path

@dataclass
class DataGroup:
    water: pd.DataFrame
    energy: pd.DataFrame
    population: pd.DataFrame

def load_data(data_path: DataPath) -> DataGroup:
    _data_path = copy.deepcopy(data_path)
    data_group = DataGroup(
        water=pd.read_csv(_data_path.water),
        energy=pd.read_csv(_data_path.energy),
        population=pd.read_csv(_data_path.population),
    )
    return data_group

def merge_data(data_group: DataGroup) -> pd.DataFrame:
    _data_group = copy.deepcopy(data_group)
    _data_group.water.rename(columns={
        "Area" : "Country",
        "Value" : "TotalRenewableWaterResourcesPerCapita",
        "Unit" : "WaterUnit"
    }, inplace=True)
    _data_group.population["Country"] = "India"
    _data_group.energy = _data_group.energy[
        (_data_group.energy["Var"] == "tes_gj_pc") &
        (_data_group.energy["Country"] == "India")
    ]
    _data_group.energy.rename(columns={
        "Value" : "TesGJPC"
    }, inplace=True)
    _data_group.energy["EnergyUnit"] = "gigajoule"
    output_df = _data_group.water.merge(
        _data_group.energy, on=['Country', 'Year'], how='inner'
    ).merge(
    _data_group.population, on=['Country', 'Year'], how='inner'
    )
    keep_columns = [
        "Year",
        "TotalRenewableWaterResourcesPerCapita",
        "WaterUnit",
        "TesGJPC",
        "EnergyUnit",
        "Population"
    ]
    output_df = output_df[keep_columns]
    output_df["Year"] = output_df["Year"].astype(int)
    return output_df

def main() -> None:
    data_path = DataPath(
        water=Path(r".\data\raw\AQUASTAT Dissemination System.csv"),
        energy=Path(r".\data\raw\Statistical Review of World Energy Narrow format.csv"),
        population=Path(r".\data\raw\India-Population-Population-2026-09-25-22-10.csv")
    )
    data_group = load_data(data_path=data_path)
    output_df = merge_data(data_group=data_group)
    output_path = Path(r".\data\processed\india.csv")
    output_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()