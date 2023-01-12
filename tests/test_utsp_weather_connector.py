from typing import List, Tuple
from hisim.components import utsp_weather_connector
from hisim.simulationparameters import SimulationParameters
from datetime import datetime

import pytest

import pandas as pd

TIME_ZONE = "Europe/Berlin"


def create_utsp_weather_connector(
    seconds_per_timestep: int = 60,
) -> utsp_weather_connector.UtspWeather:
    mysim: SimulationParameters = SimulationParameters.full_year(
        year=2021, seconds_per_timestep=seconds_per_timestep
    )
    URL = "http://134.94.131.167:443/api/v1/profilerequest"
    API_KEY = "OrjpZY93BcNWw8lKaMp0BEchbCc"
    my_weather_config = utsp_weather_connector.UtspWeatherConfig.get_default_config(
        URL, API_KEY
    )
    my_weather = utsp_weather_connector.UtspWeather(mysim, my_weather_config)
    return my_weather


@pytest.mark.parametrize(
    "seconds_per_timestep,input_data,expected_data",
    [
        (
            60 * 5,
            [
                (datetime(2021, 1, 1, 0, 0), 0),
                (datetime(2021, 1, 1, 0, 15), 15),
                (datetime(2021, 1, 1, 0, 30), 30),
            ],
            [
                (datetime(2021, 1, 1, 0, 0), 0),
                (datetime(2021, 1, 1, 0, 5), 5),
                (datetime(2021, 1, 1, 0, 10), 10),
                (datetime(2021, 1, 1, 0, 15), 15),
                (datetime(2021, 1, 1, 0, 20), 20),
                (datetime(2021, 1, 1, 0, 25), 25),
                (datetime(2021, 1, 1, 0, 30), 30),
            ],
        ),
        (
            60 * 15,
            [
                (datetime(2021, 1, 1, 0, 0), 0),
                (datetime(2021, 1, 1, 0, 5), 5),
                (datetime(2021, 1, 1, 0, 10), 10),
                (datetime(2021, 1, 1, 0, 15), 15),
                (datetime(2021, 1, 1, 0, 20), 20),
                (datetime(2021, 1, 1, 0, 25), 25),
                (datetime(2021, 1, 1, 0, 30), 30),
            ],
            [
                (datetime(2021, 1, 1, 0, 0), 0),
                (datetime(2021, 1, 1, 0, 15), 15),
                (datetime(2021, 1, 1, 0, 30), 30),
            ],
        ),
        (
            60 * 15,
            [
                (datetime(2021, 1, 1, 0, 0), 0),
                (datetime(2021, 1, 1, 0, 15), 15),
                (datetime(2021, 1, 1, 0, 30), 30),
            ],
            [
                (datetime(2021, 1, 1, 0, 0), 0),
                (datetime(2021, 1, 1, 0, 15), 15),
                (datetime(2021, 1, 1, 0, 30), 30),
            ],
        ),
        (
            60 * 10,
            [
                (datetime(2021, 1, 1, 0, 5), 0),
                (datetime(2021, 1, 1, 0, 15), 15),
                (datetime(2021, 1, 1, 0, 25), 30),
            ],
            [
                (datetime(2021, 1, 1, 0, 0), 0),
                (datetime(2021, 1, 1, 0, 10), 7.5),
                (datetime(2021, 1, 1, 0, 20), 22.5),
                (datetime(2021, 1, 1, 0, 30), 30),
            ],
        ),
    ],
)
def test_interpolate(
    seconds_per_timestep: int,
    input_data: List[Tuple[datetime, float]],
    expected_data: List[Tuple[datetime, float]],
):
    """
    Tests the interpolate method of the utsp weather connector

    :param seconds_per_timestep: seconds per timestep of the simulation
    :type seconds_per_timestep: _type_
    :param input_data: _description_
    :type input_data: _type_
    :param expected_data: _description_
    :type expected_data: _type_
    """
    input_series = given_valid_tz_aware_timeseries(input_data)
    expected_series = given_valid_tz_aware_timeseries(expected_data)

    weather = create_utsp_weather_connector(seconds_per_timestep)
    interpolated = weather.interpolate(input_series)

    # Assume the year is no leap year (all test cases are for 2021)
    SECONDS_PER_YEAR = 365 * 24 * 60 * 60
    assert (
        len(interpolated) == SECONDS_PER_YEAR / seconds_per_timestep
    ), "Interpolated series has the wrong length"

    interpolated = interpolated.iloc[: len(expected_series)]

    pd.testing.assert_series_equal(
        expected_series,
        interpolated,
        check_names=False,
    )


def given_valid_tz_aware_timeseries(data: List[Tuple[datetime, float]]) -> pd.Series:
    """
    Create a pandas series from a list of date-value tuples. Sets timezone info as required
    for the interpolate method of the utsp_weather_connector.

    :param data: list of date indices and values
    :type data: List[Tuple[datetime, float]]
    :return: the series object
    :rtype: pd.Series
    """
    tz_data = [(pd.Timestamp(d, tz=TIME_ZONE), float(v)) for d, v in data]
    df = pd.DataFrame(tz_data, columns=["timestamp", "value"])
    df.set_index("timestamp", inplace=True)
    return df["value"]
