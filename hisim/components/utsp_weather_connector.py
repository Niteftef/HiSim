""" Retrieves weather data from the UTSP """

import datetime
import io
import math
import os
from dataclasses import dataclass
from typing import Any, List

import numpy as np
import pandas as pd
import pvlib
from utspclient import client, datastructures

from hisim import loadtypes as lt
from hisim import log, utils, utsp_utils
from hisim.component import Component, ComponentOutput, ConfigBase, SingleTimeStepValues
from hisim.simulationparameters import SimulationParameters


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
            "resolution_in_min": 60
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
        self.dry_bulb_list: List[float]

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
        # location_dict = get_coordinates(self.weather_config.source_path)
        # self.simulation_repository.set_entry("weather_location", location_dict)
        cachefound, cache_filepath = utils.get_cache_file(
            "Weather", self.weather_config, self.my_simulation_parameters
        )
        if cachefound:
            # read cached files
            my_weather = pd.read_csv(
                cache_filepath, sep=",", decimal=".", encoding="cp1252"
            )
            self.temperature_list = my_weather["t_out"].tolist()
            self.dry_bulb_list = self.temperature_list
            self.DHI_list = my_weather["DHI"].tolist()
            self.DNI_list = my_weather[
                "DNI"
            ].tolist()  # self np.float64( maybe not needed? - Noah
            self.DNIextra_list = my_weather["DNIextra"].tolist()
            self.GHI_list = my_weather["GHI"].tolist()
            self.altitude_list = my_weather["altitude"].tolist()
            self.azimuth_list = my_weather["azimuth"].tolist()
            self.apparent_zenith_list = my_weather["apparent_zenith"].tolist()
            self.wind_speed_list = my_weather["Wspd"].tolist()
        else:
            raw_data = self.get_data_from_utsp()
            tmy_data = read_data(raw_data)

            DNI = self.interpolate(tmy_data["DNI"], self.my_simulation_parameters.year)
            # calculate extra terrestrial radiation- n eeded for perez array diffuse irradiance models
            dni_extra = pd.Series(pvlib.irradiance.get_extra_radiation(DNI.index), index=DNI.index)  # type: ignore
            # DNI_data = self.interpolate(tmy_data['DNI'], 2015)
            temperature = self.interpolate(
                tmy_data["T"], self.my_simulation_parameters.year
            )
            DHI = self.interpolate(tmy_data["DHI"], self.my_simulation_parameters.year)
            GHI = self.interpolate(tmy_data["GHI"], self.my_simulation_parameters.year)
            solpos = pvlib.solarposition.get_solarposition(DNI.index, location["latitude"], location["longitude"])  # type: ignore
            altitude = solpos["elevation"]
            azimuth = solpos["azimuth"]
            apparent_zenith = solpos["apparent_zenith"]
            wind_speed = self.interpolate(
                tmy_data["Wspd"], self.my_simulation_parameters.year
            )

            if seconds_per_timestep != 60:
                self.temperature_list = (
                    temperature.resample(str(seconds_per_timestep) + "S")
                    .mean()
                    .tolist()
                )
                self.dry_bulb_list = (
                    temperature.resample(str(seconds_per_timestep) + "S")
                    .mean()
                    .to_list()
                )
                self.DHI_list = (
                    DHI.resample(str(seconds_per_timestep) + "S").mean().tolist()
                )
                # np.float64( ## not sure what this is fore. python float and npfloat 64 are the same.
                self.DNI_list = (
                    DNI.resample(str(seconds_per_timestep) + "S").mean().tolist()
                )  # )  # type: ignore
                self.DNIextra_list = (
                    dni_extra.resample(str(seconds_per_timestep) + "S").mean().tolist()
                )
                self.GHI_list = (
                    GHI.resample(str(seconds_per_timestep) + "S").mean().tolist()
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
                self.temperature_list = temperature.tolist()
                self.dry_bulb_list = temperature.to_list()
                self.DHI_list = DHI.tolist()
                self.DNI_list = DNI.tolist()
                self.DNIextra_list = dni_extra.tolist()
                self.GHI_list = GHI.tolist()
                self.altitude_list = altitude.tolist()
                self.azimuth_list = azimuth.tolist()
                self.apparent_zenith_list = apparent_zenith.tolist()
                self.wind_speed_list = (
                    wind_speed.resample(str(seconds_per_timestep) + "S").mean().tolist()
                )

            solardata = [
                self.DNI_list,
                self.DHI_list,
                self.GHI_list,
                self.temperature_list,
                self.altitude_list,
                self.azimuth_list,
                self.apparent_zenith_list,
                self.dry_bulb_list,
                self.wind_speed_list,
                self.DNIextra_list,
            ]

            database = pd.DataFrame(
                np.transpose(solardata),
                columns=[
                    "DNI",
                    "DHI",
                    "GHI",
                    "t_out",
                    "altitude",
                    "azimuth",
                    "apparent_zenith",
                    "DryBulb",
                    "Wspd",
                    "DNIextra",
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

    def interpolate(self, pd_database: Any, year: int) -> Any:
        """Interpolates a time series."""
        # firstday = pd.Series(
        #     [0.0],
        #     index=[
        #         pd.to_datetime(
        #             datetime.datetime(year - 1, 12, 31, 23, 0), utc=True
        #         ).tz_convert(tz="Europe/Berlin")
        #     ],
        # )
        lastday = pd.Series(
            pd_database[-1],
            index=[
                pd.to_datetime(
                    datetime.datetime(year, 12, 31, 22, 59), utc=True
                ).tz_convert(tz="Europe/Berlin")
            ],
        )
        # pd_database = pd_database.append(firstday)
        pd_database = pd_database.append(lastday)
        pd_database = pd_database.sort_index()
        return pd_database.resample("1T").asfreq().interpolate(method="linear")

    def calc_sun_position(self, latitude_deg, longitude_deg, year, hoy):
        """Calculates the Sun Position for a specific hour and location.

        :param latitude_deg: Geographical Latitude in Degrees
        :type latitude_deg: float
        :param longitude_deg: Geographical Longitude in Degrees
        :type longitude_deg: float
        :param year: year
        :type year: int
        :param hoy: Hour of the year from the start. The first hour of January is 1
        :type hoy: int
        :return: altitude, azimuth: Sun position in altitude and azimuth degrees [degrees]
        :rtype: tuple
        """
        # Convert to Radians
        latitude_rad = math.radians(latitude_deg)
        # longitude_rad = math.radians(longitude_deg)  # Note: this is never used

        # Set the date in UTC based off the hour of year and the year itself
        start_of_year = datetime.datetime(year, 1, 1, 0, 0, 0, 0)
        utc_datetime = start_of_year + datetime.timedelta(hours=hoy)

        # Angular distance of the sun north or south of the earths equator
        # Determine the day of the year.
        day_of_year = utc_datetime.timetuple().tm_yday

        # Calculate the declination angle: The variation due to the earths tilt
        # http://www.pveducation.org/pvcdrom/properties-of-sunlight/declination-angle
        declination_rad = math.radians(
            23.45 * math.sin((2 * math.pi / 365.0) * (day_of_year - 81))
        )

        # Normalise the day to 2*pi
        # There is some reason as to why it is 364 and not 365.26
        angle_of_day = (day_of_year - 81) * (2 * math.pi / 364)

        # The deviation between local standard time and true solar time
        equation_of_time = (
            (9.87 * math.sin(2 * angle_of_day))
            - (7.53 * math.cos(angle_of_day))
            - (1.5 * math.sin(angle_of_day))
        )

        # True Solar Time
        solar_time = (
            (utc_datetime.hour * 60)
            + utc_datetime.minute
            + (4 * longitude_deg)
            + equation_of_time
        ) / 60.0

        # Angle between the local longitude and longitude where the sun is at
        # higher altitude
        hour_angle_rad = math.radians(15 * (12 - solar_time))

        # Altitude Position of the Sun in Radians
        altitude_rad = math.asin(
            math.cos(latitude_rad)
            * math.cos(declination_rad)
            * math.cos(hour_angle_rad)
            + math.sin(latitude_rad) * math.sin(declination_rad)
        )

        # Azimuth Position fo the sun in radians
        azimuth_rad = math.asin(
            math.cos(declination_rad)
            * math.sin(hour_angle_rad)
            / math.cos(altitude_rad)
        )

        # I don't really know what this code does, it has been imported from
        # PySolar
        if math.cos(hour_angle_rad) >= (
            math.tan(declination_rad) / math.tan(latitude_rad)
        ):
            return math.degrees(altitude_rad), math.degrees(azimuth_rad)
        return math.degrees(altitude_rad), (180 - math.degrees(azimuth_rad))

    def calc_sun_position2(self, hoy: Any) -> Any:
        """Calculates the sun position."""
        return self.altitude_list[hoy], self.azimuth_list[hoy]


def get_coordinates(filepath: str) -> Any:
    """Reads a test reference year file and gets the GHI, DHI and DNI from it.

    Based on the tsib project @[tsib-kotzur] (Check header)
    """
    # get the correct file path
    # filepath = os.path.join(utils.HISIMPATH["weather"][location])

    # get the geoposition
    with open(filepath + ".dat", encoding="utf-8") as file_stream:
        lines = file_stream.readlines()
        location_name = lines[0].split(maxsplit=2)[2].replace("\n", "")
        lat = float(lines[1][20:37])
        lon = float(lines[2][15:30])
    return {"name": location_name, "latitude": lat, "longitude": lon}
    # self.index = pd.date_range(f"{year}-01-01 00:00:00", periods=60 * 24 * 365, freq="T", tz="Europe/Berlin")


def read_data(raw_data: str) -> pd.DataFrame:
    data_buffer = io.StringIO(raw_data)
    data = pd.read_csv(
        data_buffer,
        index_col=0,
        parse_dates=[0],
    )
    # convert to datetime index (needs to be done in UTC), and then change the time zone back to utc+1
    data.index = pd.to_datetime(data.index, utc=True).tz_convert(tz="Europe/Berlin")

    "temperature [degC]", "pressure [hPa]", "wind direction [deg]",
    "wind speed [m/s]", "cloud coverage [1/8]", "humidity [%]",
    "direct irradiance [W/m^2]", "diffuse irradiance [W/m^2]",
    "synthetic global irradiance [W/m^2]",
    "synthetic diffuse irradiance [W/m^2]", "clear sky irradiance [W/m^2]"

    data = data.rename(
        columns={
            "diffuse irradiance [W/m^2]": "DHI",
            "direct irradiance [W/m^2]": "DNI",
            "temperature [degC]": "T",
            "wind speed [m/s]": "Wspd",
            "MM": "Month",
            "DD": "Day",
            "HH": "Hour",
            "pressure [hPa]": "Pressure",
            "wind direction [deg]": "Wdir",
        }
    )
    return data


def read_test_reference_year_data(weatherconfig: UtspWeatherConfig, year: int) -> Any:
    """Reads a test reference year file and gets the GHI, DHI and DNI from it.

    Based on the tsib project @[tsib-kotzur] (Check header)
    """
    # get the correct file path
    filepath = os.path.join(weatherconfig.source_path)
    if weatherconfig.data_source == WeatherDataSourceEnum.NSRDB:
        data, location_dict = read_nsrdb_data(filepath, year)
    elif weatherconfig.data_source == WeatherDataSourceEnum.DWD:
        data, location_dict = read_dwd_data(filepath, year)

    return data, location_dict


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
        data.index = pd.to_datetime(data.index, utc=True).tz_convert("Europe/Berlin")
    # else read from .dat and calculate DNI etc.
    else:
        # get data
        data = pd.read_csv(filepath + ".dat", sep=r"\s+", skiprows=list(range(0, 31)))
        data.index = pd.date_range(
            f"{year}-01-01 00:30:00", periods=8760, freq="H", tz="Europe/Berlin"
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
        f"{year}-01-01 00:30:00", periods=8760, freq="H", tz="Europe/Berlin"
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
