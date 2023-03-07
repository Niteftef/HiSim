"""Test for heat demand calculation in the building module.

The aim is to compare the calculated heat demand in the building module with the heat demand given by TABULA.
"""
# clean
import os
from typing import Optional
import numpy as np
import pandas as pd

import hisim.simulator as sim
from hisim.simulator import SimulationParameters
from hisim.components import loadprofilegenerator_connector
from hisim.components import weather
from hisim.components import building
from hisim.components import fake_heater
from hisim import utils
from hisim import log

__authors__ = "Katharina Rieck, Noah Pflugradt"
__copyright__ = "Copyright 2022, FZJ-IEK-3"
__credits__ = ["Noah Pflugradt"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Noah Pflugradt"
__status__ = "development"

# PATH and FUNC needed to build simulator, PATH is fake
PATH = "../examples/household_for_test_building_heat_demand_with_dummy_heater.py"
FUNC = "house_with_dummy_heater_for_heating_test"


def test_house_with_dummy_heater_for_heating_test(
    my_simulation_parameters: Optional[SimulationParameters] = None,
) -> None:  # noqa: too-many-statements
    """Test for heating energy demand.

    This setup function emulates an household including the basic components. Here the residents have their
    heating needs covered by the heat pump.

    - Simulation Parameters
    - Components
        - Occupancy (Residents' Demands)
        - Weather
        - Building
        - Dummy Heater
    """

    # =========================================================================================================================================================
    # System Parameters

    # Set Simulation Parameters
    year = 2021
    seconds_per_timestep = 60 * 60

    # Set Occupancy
    occupancy_profile = "CH01"

    # Set Dummy Heater
    set_heating_temperature_for_building_in_celsius = 19.5
    set_cooling_temperature_for_building_in_celsius = 20.5
    # =========================================================================================================================================================
    # Build Components

    # Build Simulation Parameters
    if my_simulation_parameters is None:
        my_simulation_parameters = SimulationParameters.full_year_all_options(
            year=year, seconds_per_timestep=seconds_per_timestep
        )
        my_simulation_parameters.post_processing_options.clear()

    # in case ou want to check on all TABULA buildings -> run test over all building_codes
    d_f = pd.read_csv(
        utils.HISIMPATH["housing"],
        decimal=",",
        sep=";",
        encoding="cp1252",
        low_memory=False,
    )
    d_f_1 = pd.read_csv(
        utils.HISIMPATH["housing_reference_temperatures"],
        decimal=",",
        sep=",",
        encoding="cp1252",
        low_memory=False,
    )


    with open(
        "test_building_heating_demand_dummy_heater_all_tabula_energy_needs.csv",
        "w", encoding="utf-8"
    ) as myfile:
        myfile.write(
            "Building Code"
            + ";"
            + "Energy need for heating from Dummy Heater [kWh/(a*m2)]"
            + ";"
            + "Energy need for heating from TABULA [kWh/(a*m2)]"
            + ";"
            + "Ratio HP/TABULA"
            + "\n"
        )

        for building_code in d_f["Code_BuildingVariant"]:
            buildingdata = d_f.loc[
                d_f["Code_BuildingVariant"] == building_code
            ]
            tabula_conditioned_floor_area = buildingdata["A_C_Ref"].values[0]
            if isinstance(building_code, str) and tabula_conditioned_floor_area != 0:

                country_abbreviation = building_code.split(".")[0]

                # check if country abbreviation in weather-LocationEnum or else take Aachen as location (DE)
                try: 
                    weather_location_enum = weather.LocationEnum[country_abbreviation]
                    log.information(str(weather_location_enum))
                except:
                    weather_location_enum = weather.LocationEnum.Aachen
                    log.information("  " + str (country_abbreviation))
                    country_abbreviation = "DE"

                # get heating reference temperatures from input data from 2019 weather data
                heating_reference_temperature_in_celsius = float(d_f_1.loc[d_f_1["Location"]==country_abbreviation]["HeatingReferenceTemperature"].values[0])

                # building_code = "DE.N.SFH.05.Gen.ReEx.001.002"
                # tabula_conditioned_floor_area = d_f.loc[
                #     d_f["Code_BuildingVariant"] == building_code
                # ]["A_C_Ref"].values[0]

                # this part is copied from hisim_main
                # Build Simulator
                normalized_path = os.path.normpath(PATH)
                path_in_list = normalized_path.split(os.sep)
                if len(path_in_list) >= 1:
                    path_to_be_added = os.path.join(os.getcwd(), *path_in_list[:-1])

                my_sim: sim.Simulator = sim.Simulator(
                    module_directory=path_to_be_added,
                    setup_function=FUNC,
                    my_simulation_parameters=my_simulation_parameters,
                )
                my_sim.set_simulation_parameters(my_simulation_parameters)

                # Build Occupancy
                my_occupancy_config = loadprofilegenerator_connector.OccupancyConfig(
                    profile_name=occupancy_profile, name="Occupancy"
                )
                my_occupancy = loadprofilegenerator_connector.Occupancy(
                    config=my_occupancy_config,
                    my_simulation_parameters=my_simulation_parameters,
                )

                # Build Weather
                my_weather_config = weather.WeatherConfig.get_default(
                    location_entry=weather_location_enum
                )
                my_weather = weather.Weather(
                    config=my_weather_config, my_simulation_parameters=my_simulation_parameters
                )

                # Build Building
                my_building_config = building.BuildingConfig(
                    name="TabulaBuilding",
                    heating_reference_temperature_in_celsius=heating_reference_temperature_in_celsius,
                    building_code=building_code,
                    building_heat_capacity_class="medium",
                    initial_internal_temperature_in_celsius=20.0,
                    absolute_conditioned_floor_area_in_m2=tabula_conditioned_floor_area,
                    total_base_area_in_m2=None,
                )
                log.information("building config " +str(my_building_config))
                log.information("weather config " + str(my_weather_config))

                my_building_config.absolute_conditioned_floor_area_in_m2 = (
                    tabula_conditioned_floor_area
                )
                my_building = building.Building(
                    config=my_building_config, my_simulation_parameters=my_simulation_parameters
                )

                # Build Dummy Heater
                my_dummy_heater = fake_heater.FakeHeater(
                    my_simulation_parameters=my_simulation_parameters,
                    set_heating_temperature_for_building_in_celsius=set_heating_temperature_for_building_in_celsius,
                    set_cooling_temperature_for_building_in_celsius=set_cooling_temperature_for_building_in_celsius,
                )
                # =========================================================================================================================================================
                # Connect Components

                # Building
                my_building.connect_input(
                    my_building.Altitude, my_weather.component_name, my_weather.Altitude
                )
                my_building.connect_input(
                    my_building.Azimuth, my_weather.component_name, my_weather.Azimuth
                )
                my_building.connect_input(
                    my_building.DirectNormalIrradiance,
                    my_weather.component_name,
                    my_weather.DirectNormalIrradiance,
                )
                my_building.connect_input(
                    my_building.DiffuseHorizontalIrradiance,
                    my_weather.component_name,
                    my_weather.DiffuseHorizontalIrradiance,
                )
                my_building.connect_input(
                    my_building.GlobalHorizontalIrradiance,
                    my_weather.component_name,
                    my_weather.GlobalHorizontalIrradiance,
                )
                my_building.connect_input(
                    my_building.DirectNormalIrradianceExtra,
                    my_weather.component_name,
                    my_weather.DirectNormalIrradianceExtra,
                )
                my_building.connect_input(
                    my_building.ApparentZenith,
                    my_weather.component_name,
                    my_weather.ApparentZenith,
                )
                my_building.connect_input(
                    my_building.TemperatureOutside,
                    my_weather.component_name,
                    my_weather.TemperatureOutside,
                )
                my_building.connect_input(
                    my_building.HeatingByResidents,
                    my_occupancy.component_name,
                    my_occupancy.HeatingByResidents,
                )
                my_building.connect_input(
                    my_building.ThermalPowerDelivered,
                    my_dummy_heater.component_name,
                    my_dummy_heater.ThermalPowerDelivered,
                )
                my_building.connect_input(
                    my_building.SetHeatingTemperature,
                    my_dummy_heater.component_name,
                    my_dummy_heater.SetHeatingTemperatureForBuilding,
                )
                my_building.connect_input(
                    my_building.SetCoolingTemperature,
                    my_dummy_heater.component_name,
                    my_dummy_heater.SetCoolingTemperatureForBuilding,
                )
                # Dummy Heater
                my_dummy_heater.connect_input(
                    my_dummy_heater.TheoreticalThermalBuildingDemand,
                    my_building.component_name,
                    my_building.TheoreticalThermalBuildingDemand,
                )
                # =========================================================================================================================================================
                # Add Components to Simulator and run all timesteps

                my_sim.add_component(my_weather)
                my_sim.add_component(my_occupancy)
                my_sim.add_component(my_building)
                my_sim.add_component(my_dummy_heater)

                my_sim.run_all_timesteps()

                # =========================================================================================================================================================
                # Calculate annual dummy heater heating energy
                # log.information("tabula floor area " + str(tabula_conditioned_floor_area))
                # log.information("absolute floor area " + str(my_building.buildingconfig.absolute_conditioned_floor_area_in_m2))
                results_dummy_heater_heating = my_sim.results_data_frame[
                    "FakeHeaterSystem - HeatingPowerDelivered [Heating - W]"
                ]
                # log.information(str(my_sim.results_data_frame["TabulaBuilding - TemperatureIndoorAir [Temperature - °C]"]))
                sum_heating_in_watt_timestep = sum(results_dummy_heater_heating)
                timestep_factor = seconds_per_timestep / 3600
                sum_heating_in_watt_hour = sum_heating_in_watt_timestep * timestep_factor
                sum_heating_in_kilowatt_hour = sum_heating_in_watt_hour / 1000
                # =========================================================================================================================================================
                # Test annual floor related heating demand

                energy_need_for_heating_given_by_tabula_in_kilowatt_hour_per_year_per_m2 = (
                    my_building.buildingdata["q_h_nd"].values[0]
                )

                energy_need_for_heating_from_dummy_heater_in_kilowatt_hour_per_year_per_m2 = (
                    np.round(
                        (
                            sum_heating_in_kilowatt_hour
                            / my_building_config.absolute_conditioned_floor_area_in_m2
                        ),
                        1,
                    )
                )

                ratio_hp_tabula = np.round(
                    energy_need_for_heating_from_dummy_heater_in_kilowatt_hour_per_year_per_m2
                    / energy_need_for_heating_given_by_tabula_in_kilowatt_hour_per_year_per_m2,
                    2,
                )

                with open(
                    "test_building_heating_demand_dummy_heater_all_tabula_energy_needs.csv",
                    "a", encoding="utf-8"
                ) as myfile:
                    myfile.write(
                        building_code
                        + ";"
                        + str(
                            energy_need_for_heating_from_dummy_heater_in_kilowatt_hour_per_year_per_m2
                        )
                        + ";"
                        + str(
                            energy_need_for_heating_given_by_tabula_in_kilowatt_hour_per_year_per_m2
                        )
                        + ";"
                        + str(ratio_hp_tabula)
                        + "\n"
                    )
