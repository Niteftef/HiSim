""" Retrieves weather data from the UTSP """

from enum import Enum
import io
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List

import numpy as np
import pandas as pd
import pvlib
from utspclient import client, datastructures

from hisim import loadtypes as lt
from hisim import log, utils, utsp_utils
from hisim.component import Component, ComponentOutput, ConfigBase, SingleTimeStepValues
from hisim.simulationparameters import SimulationParameters

# time zone used for calculation
TIME_ZONE = "Europe/Berlin"


class AtmosphericClimateVariable(str, Enum):
    DIRECT_NORMAL_IRRADIANCE = "dni"  # [W/m^2]
    DIRECT_NORMAL_IRRADIANCE_EXTRATERRESTRIAL = (
        "direct_normal_irradiance_extraterrestrial"  # [W/m^2]
    )
    DIRECT_HORIZONTAL_IRRADIANCE = "direct_horizontal_irradiance"  # [W/m^2]
    DIFFUSE_HORIZONTAL_IRRADIANCE = "dhi"  # [W/m^2]
    GLOBAL_HORIZONTAL_IRRADIANCE = "ghi"  #  [W/m^2]
    SURFACE_AIR_TEMPERATURE = "temp_air"  # [degC]
    WIND_SPEED = "wind_speed"  # [m/s]
    AIR_PRESSURE = "air_pressure"  # [hPa]
    WIND_DIRECTION = "wind_direction"  # [deg]
    AZIMUTH = "azimuth"
    ALTITUDE = "altitude"
    APPARENT_ZENITH = "apparent_zenith"


@dataclass
class UtspWeatherConfig(ConfigBase, utsp_utils.UtspConfig):
    """Configuration class for Weather."""

    weather_request: str

    @classmethod
    def get_main_classname(cls):
        """Get the name of the main class."""
        return UtspWeather.get_full_classname()

    @classmethod
    def get_default_config(
        cls, url: str = "http://localhost:443/api/v1/profilerequest", api_key: str = ""
    ) -> "UtspWeatherConfig":
        config = UtspWeatherConfig(
            url,
            api_key,
            name="UtspWeather_1",
            weather_request="""{
            "reference_region": 1,
            "reference_condition": "a",
            "reference_projection": 2015,
            "resolution_in_min": 15
        }""",
        )
        return config


class UtspWeather(Component):

    """Provide thermal and solar conditions of local weather."""

    # Inputs
    # None

    # Outputs
    TemperatureOutside = "TemperatureOutside"
    DirectNormalIrradiance = "DirectNormalIrradiance"
    DiffuseHorizontalIrradiance = "DiffuseHorizontalIrradiance"
    DirectNormalIrradianceExtra = "DirectNormalIrradianceExtra"
    GlobalHorizontalIrradiance = "GlobalHorizontalIrradiance"
    Altitude = "Altitude"
    Azimuth = "Azimuth"
    ApparentZenith = "ApparentZenith"
    WindSpeed = "WindSpeed"
    Weather_Temperature_Forecast_24h = "Weather_Temperature_Forecast_24h"

    Weather_TemperatureOutside_yearly_forecast = (
        "Weather_TemperatureOutside_yearly_forecast"
    )
    Weather_DirectNormalIrradiance_yearly_forecast = (
        "Weather_DirectNormalIrradiance_yearly_forecast"
    )
    Weather_DiffuseHorizontalIrradiance_yearly_forecast = (
        "Weather_DiffuseHorizontalIrradiance_yearly_forecast"
    )
    Weather_DirectNormalIrradianceExtra_yearly_forecast = (
        "Weather_DirectNormalIrradianceExtra_yearly_forecast"
    )
    Weather_GlobalHorizontalIrradiance_yearly_forecast = (
        "Weather_GlobalHorizontalIrradiance_yearly_forecast"
    )
    Weather_Azimuth_yearly_forecast = "Weather_Azimuth_yearly_forecast"
    Weather_ApparentZenith_yearly_forecast = "Weather_ApparentZenith_yearly_forecast"
    Weather_WindSpeed_yearly_forecast = "Weather_WindSpeed_yearly_forecast"

    @utils.measure_execution_time
    def __init__(
        self, my_simulation_parameters: SimulationParameters, config: UtspWeatherConfig
    ):
        """Initializes the entire class."""
        super().__init__(
            name="Weather", my_simulation_parameters=my_simulation_parameters
        )
        if my_simulation_parameters is None:
            raise Exception("Simparameters was none")
        self.last_timestep_with_update = -1
        self.weather_config = config
        self.parameter_string = my_simulation_parameters.get_unique_key()

        self.air_temperature_output: ComponentOutput = self.add_output(
            self.component_name,
            self.TemperatureOutside,
            lt.LoadTypes.TEMPERATURE,
            lt.Units.CELSIUS,
        )

        self.DNI_output: ComponentOutput = self.add_output(
            self.component_name,
            self.DirectNormalIrradiance,
            lt.LoadTypes.IRRADIANCE,
            lt.Units.WATT_PER_SQUARE_METER,
        )

        self.DNI_extra_output: ComponentOutput = self.add_output(
            self.component_name,
            self.DirectNormalIrradianceExtra,
            lt.LoadTypes.IRRADIANCE,
            lt.Units.WATT_PER_SQUARE_METER,
        )

        self.DHI_output: ComponentOutput = self.add_output(
            self.component_name,
            self.DiffuseHorizontalIrradiance,
            lt.LoadTypes.IRRADIANCE,
            lt.Units.WATT_PER_SQUARE_METER,
        )

        self.GHI_output: ComponentOutput = self.add_output(
            self.component_name,
            self.GlobalHorizontalIrradiance,
            lt.LoadTypes.IRRADIANCE,
            lt.Units.WATT_PER_SQUARE_METER,
        )

        self.altitude_output: ComponentOutput = self.add_output(
            self.component_name, self.Altitude, lt.LoadTypes.ANY, lt.Units.DEGREES
        )

        self.azimuth_output: ComponentOutput = self.add_output(
            self.component_name, self.Azimuth, lt.LoadTypes.ANY, lt.Units.DEGREES
        )

        self.apparent_zenith_output: ComponentOutput = self.add_output(
            self.component_name, self.ApparentZenith, lt.LoadTypes.ANY, lt.Units.DEGREES
        )

        self.wind_speed_output: ComponentOutput = self.add_output(
            self.component_name,
            self.WindSpeed,
            lt.LoadTypes.SPEED,
            lt.Units.METER_PER_SECOND,
        )
        self.temperature_list: List[float]
        self.DNI_list: List[float]
        self.DNIextra_list: List[float]
        self.altitude_list: List[float]
        self.azimuth_list: List[float]
        self.wind_speed_list: List[float]
        self.GHI_list: List[float]
        self.apparent_zenith_list: List[float]
        self.DHI_list: List[float]

    def write_to_report(self):
        """Write configuration to the report."""
        lines = []
        lines.append("Weather")
        lines.append(self.weather_config.get_string_dict())  # type: ignore
        return lines

    def i_save_state(self) -> None:
        """Saves the current state."""
        pass

    def i_restore_state(self) -> None:
        """Restores the previous state. Not needed for weather."""
        pass

    def i_doublecheck(self, timestep: int, stsv: SingleTimeStepValues) -> None:
        """Double chekc."""
        pass

    def i_simulate(
        self, timestep: int, stsv: SingleTimeStepValues, force_convergence: bool
    ) -> None:
        """Performs the simulation."""
        if self.last_timestep_with_update == timestep:
            return
        if force_convergence:
            return
        """ Performs the simulation. """
        stsv.set_output_value(
            self.air_temperature_output, self.temperature_list[timestep]
        )
        stsv.set_output_value(self.DNI_output, self.DNI_list[timestep])
        stsv.set_output_value(self.DNI_extra_output, self.DNIextra_list[timestep])
        stsv.set_output_value(self.DHI_output, self.DHI_list[timestep])
        stsv.set_output_value(self.GHI_output, self.GHI_list[timestep])
        stsv.set_output_value(self.altitude_output, self.altitude_list[timestep])
        stsv.set_output_value(self.azimuth_output, self.azimuth_list[timestep])
        stsv.set_output_value(self.wind_speed_output, self.wind_speed_list[timestep])
        stsv.set_output_value(
            self.apparent_zenith_output, self.apparent_zenith_list[timestep]
        )

        # set the temperature forecast
        if self.my_simulation_parameters.system_config.predictive:
            timesteps_24h = (
                24 * 3600 / self.my_simulation_parameters.seconds_per_timestep
            )
            last_forecast_timestep = int(timestep + timesteps_24h)
            if last_forecast_timestep > len(self.temperature_list):
                last_forecast_timestep = len(self.temperature_list)
            # log.information( type(self.temperature))
            temperatureforecast = self.temperature_list[timestep:last_forecast_timestep]
            self.simulation_repository.set_entry(
                self.Weather_Temperature_Forecast_24h, temperatureforecast
            )
        self.last_timestep_with_update = timestep

    def get_data_from_utsp(self) -> str:

        # Prepare the time series request
        request = datastructures.TimeSeriesRequest(
            self.weather_config.weather_request, "weather_provider"
        )

        log.information("Requesting weather profiles from the UTSP.")
        # Request the time series
        result = client.request_time_series_and_wait_for_delivery(
            self.weather_config.url, request, self.weather_config.api_key
        )

        data: str = result.data["weather_data.csv"].decode()
        return data

    def i_prepare_simulation(self) -> None:
        """Generates the lists to be used later."""
        seconds_per_timestep = self.my_simulation_parameters.seconds_per_timestep
        log.information(self.weather_config.to_json())
        # TODO: somehow provide the used weather location
        location = {
            "name": "Bremerhaven",
            "latitude": 53.5591,
            "longitude": 8.5872,
        }
        self.simulation_repository.set_entry("weather_location", location)
        cachefound, cache_filepath = utils.get_cache_file(
            "Weather", self.weather_config, self.my_simulation_parameters
        )
        if cachefound:
            # read cached files
            my_weather = pd.read_csv(
                cache_filepath, sep=",", decimal=".", encoding="cp1252"
            )
            self.temperature_list = my_weather[
                AtmosphericClimateVariable.SURFACE_AIR_TEMPERATURE
            ].tolist()
            self.DHI_list = my_weather[
                AtmosphericClimateVariable.DIFFUSE_HORIZONTAL_IRRADIANCE
            ].tolist()
            self.DNI_list = my_weather[
                AtmosphericClimateVariable.DIRECT_NORMAL_IRRADIANCE
            ].tolist()  # self np.float64( maybe not needed? - Noah
            self.DNIextra_list = my_weather[
                AtmosphericClimateVariable.DIRECT_NORMAL_IRRADIANCE_EXTRATERRESTRIAL
            ].tolist()
            self.GHI_list = my_weather[
                AtmosphericClimateVariable.GLOBAL_HORIZONTAL_IRRADIANCE
            ].tolist()
            self.altitude_list = my_weather[
                AtmosphericClimateVariable.ALTITUDE
            ].tolist()
            self.azimuth_list = my_weather[AtmosphericClimateVariable.AZIMUTH].tolist()
            self.apparent_zenith_list = my_weather[
                AtmosphericClimateVariable.APPARENT_ZENITH
            ].tolist()
            self.wind_speed_list = my_weather[
                AtmosphericClimateVariable.WIND_SPEED
            ].tolist()
        else:
            raw_data: str = self.get_data_from_utsp()
            data: pd.DataFrame = read_data(raw_data)

            direct_normal_irradiance = self.interpolate(
                data[AtmosphericClimateVariable.DIRECT_NORMAL_IRRADIANCE],
                self.my_simulation_parameters.year,
            )
            # calculate extra terrestrial radiation- n eeded for perez array diffuse irradiance models
            direct_normal_irradiance_extraterrestrial = pd.Series(pvlib.irradiance.get_extra_radiation(direct_normal_irradiance.index), index=direct_normal_irradiance.index)  # type: ignore
            # DNI_data = self.interpolate(tmy_data['DNI'], 2015)
            temperature = self.interpolate(
                data[AtmosphericClimateVariable.SURFACE_AIR_TEMPERATURE],
                self.my_simulation_parameters.year,
            )
            diffuse_horizontal_irradiance = self.interpolate(
                data[AtmosphericClimateVariable.DIFFUSE_HORIZONTAL_IRRADIANCE],
                self.my_simulation_parameters.year,
            )
            global_horizontal_irradiance = self.interpolate(
                data[AtmosphericClimateVariable.GLOBAL_HORIZONTAL_IRRADIANCE],
                self.my_simulation_parameters.year,
            )  # TODO: wrong column for testing (GHI column is missing)
            solar_position = pvlib.solarposition.get_solarposition(direct_normal_irradiance.index, location["latitude"], location["longitude"])  # type: ignore
            altitude = solar_position["elevation"]
            azimuth = solar_position["azimuth"]
            apparent_zenith = solar_position["apparent_zenith"]
            wind_speed = self.interpolate(
                data[AtmosphericClimateVariable.WIND_SPEED],
                self.my_simulation_parameters.year,
            )

            if seconds_per_timestep != 60:
                # resample to simulation resolution
                self.temperature_list = (
                    temperature.resample(str(seconds_per_timestep) + "S")
                    .mean()
                    .tolist()
                )
                self.DHI_list = (
                    diffuse_horizontal_irradiance.resample(
                        str(seconds_per_timestep) + "S"
                    )
                    .mean()
                    .tolist()
                )
                self.DNI_list = (
                    direct_normal_irradiance.resample(str(seconds_per_timestep) + "S")
                    .mean()
                    .tolist()
                )
                self.DNIextra_list = (
                    direct_normal_irradiance_extraterrestrial.resample(
                        str(seconds_per_timestep) + "S"
                    )
                    .mean()
                    .tolist()
                )
                self.GHI_list = (
                    global_horizontal_irradiance.resample(
                        str(seconds_per_timestep) + "S"
                    )
                    .mean()
                    .tolist()
                )
                self.altitude_list = (
                    altitude.resample(str(seconds_per_timestep) + "S").mean().tolist()
                )
                self.azimuth_list = (
                    azimuth.resample(str(seconds_per_timestep) + "S").mean().tolist()
                )
                self.apparent_zenith_list = (
                    apparent_zenith.resample(str(seconds_per_timestep) + "S")
                    .mean()
                    .tolist()
                )
                self.wind_speed_list = (
                    wind_speed.resample(str(seconds_per_timestep) + "S").mean().tolist()
                )
            else:
                # data already has the correct resolution (1 minute)
                self.temperature_list = temperature.tolist()
                self.DHI_list = diffuse_horizontal_irradiance.tolist()
                self.DNI_list = direct_normal_irradiance.tolist()
                self.DNIextra_list = direct_normal_irradiance_extraterrestrial.tolist()
                self.GHI_list = global_horizontal_irradiance.tolist()
                self.altitude_list = altitude.tolist()
                self.azimuth_list = azimuth.tolist()
                self.apparent_zenith_list = apparent_zenith.tolist()
                self.wind_speed_list = (
                    wind_speed.resample(str(seconds_per_timestep) + "S").mean().tolist()
                )

            resampled_data = [
                self.DNI_list,
                self.DHI_list,
                self.GHI_list,
                self.temperature_list,
                self.altitude_list,
                self.azimuth_list,
                self.apparent_zenith_list,
                self.wind_speed_list,
                self.DNIextra_list,
            ]

            database = pd.DataFrame(
                np.transpose(resampled_data),
                columns=[
                    AtmosphericClimateVariable.DIRECT_NORMAL_IRRADIANCE,
                    AtmosphericClimateVariable.DIFFUSE_HORIZONTAL_IRRADIANCE,
                    AtmosphericClimateVariable.GLOBAL_HORIZONTAL_IRRADIANCE,
                    AtmosphericClimateVariable.SURFACE_AIR_TEMPERATURE,
                    AtmosphericClimateVariable.ALTITUDE,
                    AtmosphericClimateVariable.AZIMUTH,
                    AtmosphericClimateVariable.APPARENT_ZENITH,
                    AtmosphericClimateVariable.WIND_SPEED,
                    AtmosphericClimateVariable.DIRECT_NORMAL_IRRADIANCE_EXTRATERRESTRIAL,
                ],
            )
            database.to_csv(cache_filepath)

        # write one year forecast to simulation repository for PV processing -> if PV forecasts are needed
        if self.my_simulation_parameters.system_config.predictive:
            self.simulation_repository.set_entry(
                self.Weather_TemperatureOutside_yearly_forecast, self.temperature_list
            )
            self.simulation_repository.set_entry(
                self.Weather_DiffuseHorizontalIrradiance_yearly_forecast, self.DHI_list
            )
            self.simulation_repository.set_entry(
                self.Weather_DirectNormalIrradiance_yearly_forecast, self.DNI_list
            )
            self.simulation_repository.set_entry(
                self.Weather_DirectNormalIrradianceExtra_yearly_forecast,
                self.DNIextra_list,
            )
            self.simulation_repository.set_entry(
                self.Weather_GlobalHorizontalIrradiance_yearly_forecast, self.GHI_list
            )
            self.simulation_repository.set_entry(
                self.Weather_Azimuth_yearly_forecast, self.azimuth_list
            )
            self.simulation_repository.set_entry(
                self.Weather_ApparentZenith_yearly_forecast, self.apparent_zenith_list
            )
            self.simulation_repository.set_entry(
                self.Weather_WindSpeed_yearly_forecast, self.wind_speed_list
            )

    def interpolate(self, data: Any, year: int) -> Any:
        """Interpolates a timeseries to 1-minute resolution and interpolates missing data
        in the beginning or end of the simulation time frame."""
        # if the data starts after the simulation start date or ends before the simulation
        # end date, add the respective date to interpolate the missing parts
        start_date = pd.Timestamp(datetime(year, 1, 1, 0, 0), tz=TIME_ZONE)
        if start_date < data.index[0]:
            firstday = pd.Series(
                [0.0],
                index=[start_date],
            )
            data = pd.concat([firstday, data])
        end_date = pd.Timestamp(datetime(year, 12, 31, 23, 59), tz=TIME_ZONE)
        if end_date > data.index[-1]:
            lastday = pd.Series(
                data[-1],
                index=[end_date],
            )
            data = pd.concat([data, lastday])
        # resample to 1 minute resolution
        # TODO: why always resample to 1 Minute and not directly to the simulation resolution
        return data.resample("1T").asfreq().interpolate(method="linear")


def read_data(raw_data: str) -> pd.DataFrame:
    """
    Parses weather data from the UTSP weather_provider.
    """
    data_buffer = io.StringIO(raw_data)
    data = pd.read_csv(
        data_buffer,
        index_col=0,
        parse_dates=[0],
    )
    # convert to datetime index (needs to be done in UTC), and then change the time zone back to utc+1
    data.index = pd.to_datetime(data.index, utc=True).tz_convert(tz=TIME_ZONE)

    # map the column names in the data to the corresponding variable names
    general_name_mapping = {
        "temperature [degC]": AtmosphericClimateVariable.SURFACE_AIR_TEMPERATURE,
        "wind speed [m/s]": AtmosphericClimateVariable.WIND_SPEED,
        "pressure [hPa]": AtmosphericClimateVariable.AIR_PRESSURE,
        "wind direction [deg]": AtmosphericClimateVariable.WIND_DIRECTION,
    }
    if any(x not in data for x in general_name_mapping.keys()):
        raise Exception("Missing weather data")
    synthetic_name_mapping = {
        "synthetic diffuse irradiance [W/m^2]": AtmosphericClimateVariable.DIFFUSE_HORIZONTAL_IRRADIANCE,
        "synthetic global irradiance [W/m^2]": AtmosphericClimateVariable.GLOBAL_HORIZONTAL_IRRADIANCE,
    }
    measured_name_mapping = {
        "direct irradiance [W/m^2]": AtmosphericClimateVariable.DIRECT_HORIZONTAL_IRRADIANCE,
        "diffuse irradiance [W/m^2]": AtmosphericClimateVariable.DIFFUSE_HORIZONTAL_IRRADIANCE,
    }
    if all(x in data for x in synthetic_name_mapping.keys()):
        # synthetic irradiance data with higher resolution is available
        mapping = dict(general_name_mapping, **synthetic_name_mapping)
    elif all(x in data for x in measured_name_mapping.keys()):
        # measured irradiance data with low resolution is available
        mapping = dict(general_name_mapping, **measured_name_mapping)
    else:
        raise Exception("Missing irradiance data")
    data = data.rename(columns=mapping)

    # TODO: calculate missing data; depending on whether synthetic or measured is used, this
    # is direct_horizontal or global_horizontal, and in each case direct_normal
    # TODO: prepare_inputs can calculate all missing irradiance data automatically: can it be used without specifying a PV-System?
    # if not: calculate missing data 'manually'
    # pvlib.modelchain.ModelChain().prepare_inputs(data)

    # calculate direct horizontal irradiance
    direct_horizontal_irradiance = (
        data[AtmosphericClimateVariable.GLOBAL_HORIZONTAL_IRRADIANCE]
        - data[AtmosphericClimateVariable.DIFFUSE_HORIZONTAL_IRRADIANCE]
    )

    lon, lat = 53, 8  # TODO: somehow obtain coordinates
    # calculate direct normal irradiance
    data[
        AtmosphericClimateVariable.DIRECT_NORMAL_IRRADIANCE
    ] = calculate_direct_normal_radiation(direct_horizontal_irradiance, lon, lat)
    return data


def read_dwd_data(filepath: str, year: int) -> Any:
    """Reads the DWD data."""
    # get the geoposition
    with open(filepath + ".dat", encoding="utf-8") as file_stream:
        lines = file_stream.readlines()
        location_name = lines[0].split(maxsplit=2)[2].replace("\n", "")
        lat = float(lines[1][20:37])
        lon = float(lines[2][15:30])
    location_dict = {"name": location_name, "latitude": lat, "longitude": lon}
    # check if time series data already exists as .csv with DNI
    if os.path.isfile(filepath + ".csv"):
        data = pd.read_csv(
            filepath + ".csv", index_col=0, parse_dates=True, sep=";", decimal=","
        )
        data.index = pd.to_datetime(data.index, utc=True).tz_convert(TIME_ZONE)
    # else read from .dat and calculate DNI etc.
    else:
        # get data
        data = pd.read_csv(filepath + ".dat", sep=r"\s+", skiprows=list(range(0, 31)))
        data.index = pd.date_range(
            f"{year}-01-01 00:30:00", periods=8760, freq="H", tz=TIME_ZONE
        )
        data["GHI"] = data["D"] + data["B"]
        data = data.rename(
            columns={
                "D": "DHI",
                "t": "T",
                "WG": "Wspd",
                "MM": "Month",
                "DD": "Day",
                "HH": "Hour",
                "p": "Pressure",
                "WR": "Wdir",
            }
        )

        # calculate direct normal
        data["DNI"] = calculate_direct_normal_radiation(
            data["B"], lon, lat
        )  # data["DNI"] = data["B"]

        # save as .csv  # data.to_csv(filepath + ".csv",sep=";",decimal=",")
    return data, location_dict


def read_nsrdb_data(filepath, year):
    """Reads a set of NSRDB data."""
    with open(filepath + ".dat", encoding="utf-8") as file_stream:
        lines = file_stream.readlines()
        location_name = lines[0].split(maxsplit=2)[2].replace("\n", "")
        lat = float(lines[1][20:25])
        lon = float(lines[2][15:20])
    location_dict = {"name": location_name, "latitude": lat, "longitude": lon}
    # get data
    data = pd.read_csv(filepath + ".dat", sep=",", skiprows=list(range(0, 11)))
    data = data.drop(data.index[8761:8772])
    data.index = pd.date_range(
        f"{year}-01-01 00:30:00", periods=8760, freq="H", tz=TIME_ZONE
    )
    data = data.rename(
        columns={
            "DHI": "DHI",
            "Temperature": "T",
            "Wind Speed": "Wspd",
            "MM": "Month",
            "DD": "Day",
            "HH": "Hour",
            "Pressure": "Pressure",
            "Wind Direction": "Wdir",
            "GHI": "GHI",
            "DNI": "DNI",
        }
    )
    return data, location_dict


def calculate_direct_normal_radiation(
    direct_horizontal_irradation, lon, lat, zenith_tol=87.0
):
    """Calculates the direct NORMAL irradiance from the direct horizontal irradiance with the help of the PV lib.

    Based on the tsib project @[tsib-kotzur] (Check header)

    Parameters
    ----------
    direct_horizontal_irradation: pd.Series with time index
        Direct horizontal irradiance
    lon: float
        Longitude of the location
    lat: float
        Latitude of the location
    zenith_tol: float, optional
        Avoid cosines of values above a certain zenith angle of in order to avoid division by zero.

    Returns
    -------
    DNI: pd.Series

    """

    solar_pos = pvlib.solarposition.get_solarposition(
        direct_horizontal_irradation.index, lat, lon
    )
    solar_pos["apparent_zenith"][solar_pos.apparent_zenith > zenith_tol] = zenith_tol
    DNI = direct_horizontal_irradation.div(
        solar_pos["apparent_zenith"].apply(math.radians).apply(math.cos)
    )
    if DNI.isnull().values.any():
        raise ValueError("Something went wrong...")
    return DNI
