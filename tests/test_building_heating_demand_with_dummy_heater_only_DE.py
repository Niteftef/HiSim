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
from hisim.components import idealized_electric_heater
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
    seconds_per_timestep = 60*60
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
    # d_f_1 = pd.read_csv(
    #     utils.HISIMPATH["housing_reference_temperatures"],
    #     decimal=",",
    #     sep=",",
    #     encoding="cp1252",
    #     low_memory=False,
    # )



    with open(
            "test_building_heating_demand_dummy_heater_DE_energy_needs0.csv","w") as myfile:
            myfile.write(
                "Building Code"
                + ";"
                + "Seconds per Timestep"
                + ";"
                + "Heating demand for heating from Dummy Heater [kWh/(a*m2)]"
                + ";"
                + "Heating demand for heating from TABULA [kWh/(a*m2)]"
                + ";"
                + "Ratio HiSim/TABULA heating demand"
                + ";"
                + "Internal gains from Occupancy [kWh/(a*m2)]"
                + ";"
                + "Internal gains from TABULA [kWh/(a*m2)]"
                + ";"
                + "Ratio HiSim/TABULA internal gains"
                + ";"
                + "Solar gains from Windows [kWh/(a*m2)]"
                + ";"
                + "Solar gains from TABULA [kWh/(a*m2)]"
                + ";"
                + "Ratio HiSim/TABULA solar gains"
                + ";"
                + "Heat Loss from Building [kWh//a*m2)]"
                + ";"
                + "Total Heat Transfer from TABULA [kWh/(a*m2)]"
                + ";"
                + "Mean Outside Temperature HiSim Weather [°C]"
                + ";"
                + "Mean Outside Temperature HiSim Input Data [°C]"
                + ";"
                + "Ratio Weather/Input Outside Temperatures"
                + ";"
                + "Mean GHI HiSim Weather [W/m2]"
                + ";"
                + "Mean GHI HiSim Input Data [W/m2]"
                + ";"
                + "Ratio Weather/Input GHI"
                + ";"
                + "Sum GHI HiSim Weather [Wh/m2]"
                + ";"
                + "Sum GHI HiSim Input Data [Wh/m2]"
                + ";"
                + "Ratio Weather/Input GHI Sum"
                + ";"
                + "Mean Thermal Mass Temperature [°C]"
                + ";"
                + "Mean Indoor Air Temperature [°C]"
                + ";"
                + "Max Thermal Demand [W]"
                + ";"
                + "Transmission for Windows and Doors, based on ISO 13790 (H_tr_w) [W/K]"
                + ";"
                + "External Part of Transmission for Opaque Surfaces, based on ISO 13790 (H_tr_em) [W/K]"
                + ";"
                + "Internal Part of Transmission for Opaque Surfaces, based on ISO 13790 (H_tr_ms) [W/K]"
                + ";"
                + "Transmission between Indoor Air and Internal Surface, based on ISO 13790 (H_tr_is) [W/K]"
                + ";"
                + "Thermal Conductance by Ventilation, based on TABULA (H_ve) [W/K]"
                + ";"
                + "Conditioned Floor Area (A_f) [m2]"
                + ";"
                + "Floor Related Thermal Capacitance of Thermal Mass, based on ISO 13790 [kWh/m2.K]"
                + ";"
                + "Floor Related Thermal Capacitance of Thermal Mass, based on TABULA [kWh/m2.K]"
                + ";"
                + "Annual Floor Related Internal Heat Gain, based on TABULA (Q_int) [kWh/m2.a]"
                + ";"
                + "Annual Floor Related Solar Heat Gain, based on TABULA (Q_sol) [kWh/m2.a]"
                + ";"
                + "Annual Floor Related Heating Demand, based on TABULA (Q_h_nd) [kWh/m2.a]"
                + ";"
                + "\n"
            )
    for building_code in d_f["Code_BuildingVariant"]:
        country_abbreviation = building_code.split(".")[0]
        if country_abbreviation == "DE":

            buildingdata = d_f.loc[
                d_f["Code_BuildingVariant"] == building_code
            ]

            tabula_conditioned_floor_area = buildingdata["A_C_Ref"].values[0]
            if isinstance(building_code, str) and tabula_conditioned_floor_area != 0:

                log.information(str(country_abbreviation))
                log.information(str(tabula_conditioned_floor_area))
                weather_location_enum = weather.LocationEnum["Aachen"]
                heating_reference_temperature_in_celsius = -14.0

                # for building_code in d_f["Code_BuildingVariant"]:
                #     if isinstance(building_code, str): #and tabula_conditioned_floor_area != 0:
                #         country_abbreviation = ".".join(building_code.split(".")[0:-2])
                #         if country_abbreviation == "DE.N.SFH.03.Gen.ReEx":

                #             buildingdata = d_f.loc[
                #                 d_f["Code_BuildingVariant"] == building_code
                #             ]

                #             log.information(str(buildingdata))
                #             log.information(str(building_code))
                #             tabula_conditioned_floor_area = buildingdata["A_C_Ref"].values[0]
                #             weather_location_enum = weather.LocationEnum["Aachen"]
                #             heating_reference_temperature_in_celsius = -14.0
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
                    profile_name=occupancy_profile, name="Occupancy", country_name="DE", number_of_apartments=1
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

                my_building_config.absolute_conditioned_floor_area_in_m2 = (
                    tabula_conditioned_floor_area
                )
                my_building = building.Building(
                    config=my_building_config, my_simulation_parameters=my_simulation_parameters
                )

                # Build Dummy Heater
                my_idealized_electric_heater = idealized_electric_heater.IdealizedElectricHeater(
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
                    my_idealized_electric_heater.component_name,
                    my_idealized_electric_heater.ThermalPowerDelivered,
                )
                my_building.connect_input(
                    my_building.SetHeatingTemperature,
                    my_idealized_electric_heater.component_name,
                    my_idealized_electric_heater.SetHeatingTemperatureForBuilding,
                )
                my_building.connect_input(
                    my_building.SetCoolingTemperature,
                    my_idealized_electric_heater.component_name,
                    my_idealized_electric_heater.SetCoolingTemperatureForBuilding,
                )
                # Dummy Heater
                my_idealized_electric_heater.connect_input(
                    my_idealized_electric_heater.TheoreticalThermalBuildingDemand,
                    my_building.component_name,
                    my_building.TheoreticalThermalBuildingDemand,
                )
                # =========================================================================================================================================================
                # Add Components to Simulator and run all timesteps

                my_sim.add_component(my_weather)
                my_sim.add_component(my_occupancy)
                my_sim.add_component(my_building)
                my_sim.add_component(my_idealized_electric_heater)

                my_sim.run_all_timesteps()

                # =========================================================================================================================================================
                # Calculate annual dummy heater heating energy

                results_dummy_heater_heating = my_sim.results_data_frame[
                    "IdealizedElectricHeater - HeatingPowerDelivered [Heating - W]"
                ]
                results_dummy_heater_heating_and_cooling = my_sim.results_data_frame[
                    "IdealizedElectricHeater - ThermalPowerDelivered [Heating - W]"
                ]
                results_building_demand = my_sim.results_data_frame[
                    "TabulaBuilding - TheoreticalThermalBuildingDemand [Heating - W]"
                ]
                sum_heating_in_watt_timestep = sum(results_dummy_heater_heating)
                timestep_factor = seconds_per_timestep / 3600
                sum_heating_in_watt_hour = sum_heating_in_watt_timestep * timestep_factor
                sum_heating_in_kilowatt_hour = sum_heating_in_watt_hour / 1000
                # =========================================================================================================================================================
                # Test annual floor related heating demand

                heating_demand_given_by_tabula_in_kilowatt_hour_per_year_per_m2 = (
                    my_building.buildingdata["q_h_nd"].values[0]
                )

                heating_demand_from_dummy_heater_in_kilowatt_hour_per_year_per_m2 = (
                    np.round(
                        (
                            sum_heating_in_kilowatt_hour
                            / my_building_config.absolute_conditioned_floor_area_in_m2
                        ),
                        1,
                    )
                )

                ratio_hisim_tabula_heating_demand = np.round(
                    heating_demand_from_dummy_heater_in_kilowatt_hour_per_year_per_m2
                    / heating_demand_given_by_tabula_in_kilowatt_hour_per_year_per_m2,
                    2,
                )
                # =========================================================================================================================================================
                # Calculate annual internal heat gains

                results_occupancy_internal_heat_gains = my_sim.results_data_frame[
                    "Occupancy - HeatingByResidents [Heating - W]"
                ]
                sum_internal_heat_gains_in_watt_timestep = sum(results_occupancy_internal_heat_gains)
                sum_internal_heat_gains_in_watt_hour = sum_internal_heat_gains_in_watt_timestep * timestep_factor
                sum_internal_heat_gains_in_kilowatt_hour = sum_internal_heat_gains_in_watt_hour / 1000
                # =========================================================================================================================================================
                # Test annual floor related internal heat gains

                internal_gains_given_by_tabula_in_kilowatt_hour_per_year_per_m2 = (
                    my_building.buildingdata["q_int"].values[0]
                )

                internal_gains_given_by_dummy_heater_in_kilowatt_hour_per_year_per_m2 = (
                    np.round(
                        (
                            sum_internal_heat_gains_in_kilowatt_hour
                            / my_building_config.absolute_conditioned_floor_area_in_m2
                        ),
                        1,
                    )
                )

                ratio_hisim_tabula_internal_gains = np.round(
                    internal_gains_given_by_dummy_heater_in_kilowatt_hour_per_year_per_m2
                    / internal_gains_given_by_tabula_in_kilowatt_hour_per_year_per_m2,
                    2,
                )

                # =========================================================================================================================================================
                # Calculate annual solar heat gains

                results_building_solar_heat_gains = my_sim.results_data_frame[
                    "TabulaBuilding - SolarGainThroughWindows [Heating - W]"
                ]
                sum_solar_heat_gains_in_watt_timestep = sum(results_building_solar_heat_gains)
                sum_solar_heat_gains_in_watt_hour = sum_solar_heat_gains_in_watt_timestep * timestep_factor
                sum_solar_heat_gains_in_kilowatt_hour = sum_solar_heat_gains_in_watt_hour / 1000
                # =========================================================================================================================================================
                # Test annual floor related solar heat gains

                solar_gains_given_by_tabula_in_kilowatt_hour_per_year_per_m2 = (
                    my_building.buildingdata["q_sol"].values[0]
                )

                solar_gains_given_by_dummy_heater_in_kilowatt_hour_per_year_per_m2 = (
                    np.round(
                        (
                            sum_solar_heat_gains_in_kilowatt_hour
                            / my_building_config.absolute_conditioned_floor_area_in_m2
                        ),
                        1,
                    )
                )

                ratio_hisim_tabula_solar_gains = np.round(
                    solar_gains_given_by_dummy_heater_in_kilowatt_hour_per_year_per_m2
                    / solar_gains_given_by_tabula_in_kilowatt_hour_per_year_per_m2,
                    2,
                )
                # =========================================================================================================================================================
                # Calculate annual heat loss

                results_building_heat_loss = my_sim.results_data_frame[
                    "TabulaBuilding - HeatLoss [Heating - W]"
                ]
                sum_heat_loss_in_watt_timestep = sum(results_building_heat_loss)
                sum_heat_loss_in_watt_hour = sum_heat_loss_in_watt_timestep * timestep_factor
                sum_heat_loss_in_kilowatt_hour = sum_heat_loss_in_watt_hour / 1000
                # =========================================================================================================================================================
                # Test annual floor related heat loss

                total_heat_transfer_given_by_tabula_in_kilowatt_hour_per_year_per_m2 = (
                    my_building.buildingdata["q_ht"].values[0]
                )

                heat_loss_given_by_dummy_heater_in_kilowatt_hour_per_year_per_m2 = (
                    np.round(
                        (
                            sum_heat_loss_in_kilowatt_hour
                            / my_building_config.absolute_conditioned_floor_area_in_m2
                        ),
                        1,
                    )
                )
                # =========================================================================================================================================================
                # Calculate mean outside temperature 

                results_weather_outside_temperature = my_sim.results_data_frame[
                    "Weather - DailyAverageOutsideTemperatures [Temperature - °C]"
                ]
                mean_weather_outside_temperature = np.mean(results_weather_outside_temperature)
                # =========================================================================================================================================================
                # Test mean outside temperature

                weather_original_source = weather.read_test_reference_year_data(weatherconfig=my_weather_config, year=my_simulation_parameters.year)
                weather_outside_temperature_original = weather_original_source["T"]
                mean_weather_outside_temperature_original = np.mean(weather_outside_temperature_original)

                ratio_hisimweather_inputdata_outside_temperatures = np.round(
                    mean_weather_outside_temperature
                    / mean_weather_outside_temperature_original,
                    3,
                )

                # =========================================================================================================================================================
                # Calculate mean GHI and total GHI

                results_weather_ghi = my_sim.results_data_frame[
                    "Weather - GlobalHorizontalIrradiance [Irradiance - W per square meter]"
                ]
                mean_weather_ghi = np.mean(results_weather_ghi)
                sum_weather_ghi_in_watt_per_m2 = sum(results_weather_ghi)
                sum_weather_ghi_in_watt_hour_per_m2 = sum_weather_ghi_in_watt_per_m2 * timestep_factor
                # =========================================================================================================================================================
                # Test mean GHI and total GHI


                weather_ghi_original = weather_original_source["GHI"] # weather_original_source["D"] + weather_original_source["B"]
                mean_weather_ghi_original = np.mean(weather_ghi_original)
                sum_weather_ghi_original_in_watt_per_m2 = sum(weather_ghi_original)
                sum_weather_ghi_original_in_watt_hour_per_m2 = sum_weather_ghi_original_in_watt_per_m2
                ratio_hisimweather_inputdata_ghi = np.round(
                    mean_weather_ghi
                    / mean_weather_ghi_original,
                    3,
                )

                ratio_hisim_weather_inputdata_ghi_sum = np.round(sum_weather_ghi_in_watt_hour_per_m2 / sum_weather_ghi_original_in_watt_hour_per_m2, 3)

                # =========================================================================================================================================================
                # Calculate mean thermal mass temperature

                results_building_thermal_mass_temperature = my_sim.results_data_frame[
                    "TabulaBuilding - TemperatureMeanThermalMass [Temperature - °C]"
                ]
                mean_building_thermal_mass_temperature = np.mean(results_building_thermal_mass_temperature)

                # =========================================================================================================================================================
                # Calculate mean indoor air temperature

                results_building_indoor_air_temperature = my_sim.results_data_frame[
                    "TabulaBuilding - TemperatureIndoorAir [Temperature - °C]"
                ]
                mean_building_indoor_air_temperature = np.mean(results_building_indoor_air_temperature)


                building_report = my_building.write_for_heating_demand_test()

                with open(
                    "test_building_heating_demand_dummy_heater_DE_energy_needs1.csv", "a") as myfile:
                    myfile.write(
                        building_code
                        + ";"
                        + str(seconds_per_timestep)
                        + ";"
                        + str(
                            heating_demand_from_dummy_heater_in_kilowatt_hour_per_year_per_m2
                        )
                        + ";"
                        + str(
                            heating_demand_given_by_tabula_in_kilowatt_hour_per_year_per_m2
                        )
                        + ";"
                        + str(ratio_hisim_tabula_heating_demand)
                        + ";"
                        + str(
                            internal_gains_given_by_dummy_heater_in_kilowatt_hour_per_year_per_m2
                        )
                        + ";"
                        + str(
                            internal_gains_given_by_tabula_in_kilowatt_hour_per_year_per_m2
                        )
                        + ";"
                        + str(ratio_hisim_tabula_internal_gains)
                        + ";"
                        + str(
                            solar_gains_given_by_dummy_heater_in_kilowatt_hour_per_year_per_m2
                        )
                        + ";"
                        + str(
                            solar_gains_given_by_tabula_in_kilowatt_hour_per_year_per_m2
                        )
                        + ";"
                        + str(ratio_hisim_tabula_solar_gains)
                        + ";"
                        + str(heat_loss_given_by_dummy_heater_in_kilowatt_hour_per_year_per_m2)
                        + ";"
                        + str(total_heat_transfer_given_by_tabula_in_kilowatt_hour_per_year_per_m2)
                        + ";"
                        + str(mean_weather_outside_temperature)
                        + ";"
                        + str(mean_weather_outside_temperature_original)
                        + ";"
                        + str(ratio_hisimweather_inputdata_outside_temperatures)
                        + ";"
                        + str(mean_weather_ghi)
                        + ";"
                        + str(mean_weather_ghi_original)
                        + ";"
                        + str(ratio_hisimweather_inputdata_ghi)
                        + ";"
                        + str(sum_weather_ghi_in_watt_hour_per_m2)
                        + ";"
                        + str(sum_weather_ghi_original_in_watt_hour_per_m2)
                        + ";"
                        + str(ratio_hisim_weather_inputdata_ghi_sum)
                        + ";"
                        + str(mean_building_thermal_mass_temperature)
                        + ";"
                        + str(mean_building_indoor_air_temperature)
                        + ";"
                        + building_report
                        + "\n"
                    )

    csv_one=pd.read_csv("test_building_heating_demand_dummy_heater_DE_energy_needs1.csv")

    csv_one.to_csv("test_building_heating_demand_dummy_heater_DE_energy_needs0.csv", mode="a", index=False, header=True)
    csv_one.to_excel("test_building_heating_demand_dummy_heater_DE_energy_needs1.xlsx", index=None, header=None)
    # read_file = pd.read_csv(r'test_building_heating_demand_dummy_heater_DE_energy_needs0.csv', delimiter=';', encoding='unicode_escape')
    # read_file.to_excel(r'test_building_heating_demand_dummy_heater_DE_energy_needs0.xlsx', index=None, header=True)
