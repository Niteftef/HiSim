"""  Household example with generic gas heater. """

from typing import Optional, Any
from hisim.simulator import SimulationParameters
from hisim.components import loadprofilegenerator_utsp_connector
from hisim.components import loadprofilegenerator_connector
from hisim.components import weather
from hisim.components import generic_gas_heater
from hisim.components import building
from hisim.components import sumbuilder
from hisim import log
from hisim import utils
from dataclasses_json import dataclass_json
from dataclasses import dataclass
import os
from pathlib import Path
from hisim import loadtypes

__authors__ = "Vitor Hugo Bellotto Zago, Noah Pflugradt"
__copyright__ = "Copyright 2022, FZJ-IEK-3"
__credits__ = ["Noah Pflugradt"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Noah Pflugradt"
__status__ = "development"
from utspclient.helpers.lpgdata import (
    ChargingStationSets,
    Households,
    HouseTypes,
    LoadTypes,
    TransportationDeviceSets,
    TravelRouteSets,
)
from utspclient.helpers.lpgpythonbindings import CalcOption, JsonReference


@dataclass_json
@dataclass
class HouseholdPVConfig:
    PVSize: float
    BuildingType: str
    HouseholdType: JsonReference
    LPGUrl: str
    ResultPath: str
    TravelRouteSet: JsonReference
    simulation_parameters: SimulationParameters
    APIKey: str
    transportation_device_set: JsonReference
    charging_station_set: JsonReference
    PV_azimuth: float
    Tilt: float
    PV_Power: float
    total_base_area_in_m2: float

    @classmethod
    def get_default(cls):
        return HouseholdPVConfig(
            PVSize=5,
            BuildingType="blub",
            HouseholdType=Households.CHR01_Couple_both_at_Work,
            LPGUrl="http://134.94.131.167:443/api/v1/profilerequest",
            APIKey="OrjpZY93BcNWw8lKaMp0BEchbCc",
            simulation_parameters=SimulationParameters.one_day_only(2022),
            ResultPath="mypath",
            TravelRouteSet=TravelRouteSets.Travel_Route_Set_for_10km_Commuting_Distance,
            transportation_device_set=TransportationDeviceSets.Bus_and_one_30_km_h_Car,
            charging_station_set=ChargingStationSets.Charging_At_Home_with_11_kW,
            PV_azimuth=180,
            Tilt=30,
            PV_Power=10000,
            total_base_area_in_m2=121.2,
        )


def household_generic_gas_heater(
    my_sim: Any, my_simulation_parameters: Optional[SimulationParameters] = None
) -> None:  # noqa: too-many-statements
    """Basic household example.

    This setup function emulates a household with some basic components. Here the residents have their
    electricity and heating needs covered by a generic gas heater.

    - Simulation Parameters
    - Components
        - Occupancy (Residents' Demands)
        - Weather
        - Building
        - Gas Heater
    """

    config_filename = None  # "pv_hp_config.json"

    my_config: HouseholdPVConfig
    # if Path(config_filename).is_file():
    #     with open(config_filename, encoding='utf8') as system_config_file:
    #         my_config = HouseholdPVConfig.from_json(system_config_file.read())  # type: ignore
    #     log.information(f"Read system config from {config_filename}")
    # else:
    my_config = HouseholdPVConfig.get_default()

    # System Parameters #

    # Set simulation parameters
    year = 2021
    seconds_per_timestep = 60

    # Set weather
    location = "Aachen"

    # Set occupancy
    occupancy_profile = "CH01"

    # Set building
    building_code = "DE.N.SFH.05.Gen.ReEx.001.002"
    building_class = "medium"
    initial_temperature = 23
    heating_reference_temperature = -14
    absolute_conditioned_floor_area = None

    # Set generic gas heater
    temperature_delta_in_celsius = 10
    maximal_power_in_watt = 12_000
    is_modulating = True
    minimal_thermal_power_in_watt = 1_000
    maximal_thermal_power_in_watt = 12_000
    eff_th_min = 0.60
    eff_th_max = 0.90
    delta_temperature_in_celsius = 25
    maximal_mass_flow_in_kilogram_per_second = 12_000 / (4180 * 25) # -> ~0.07 P_th_max / (4180 * delta_T)
    maximal_temperature_in_celsius = 80

    # Build Components #

    # Build system parameters
    if my_simulation_parameters is None:
        my_simulation_parameters = SimulationParameters.full_year_all_options(
            year=year, seconds_per_timestep=seconds_per_timestep
        )
    my_sim.set_simulation_parameters(my_simulation_parameters)
    # # Build occupancy
    # lpgurl = "http://"
    # api_key = "asdf"
    # result_path = os.path.join(utils.get_input_directory(), "lpg_profiles")
    # my_occupancy_config = loadprofilegenerator_utsp_connector.UtspLpgConnectorConfig(url=my_config.LPGUrl,
    #                                                                                  api_key=my_config.APIKey,
    #                                                                                  household=my_config.HouseholdType,
    #                                                                                  result_path=my_config.ResultPath,
    #                                                                                  travel_route_set=my_config.TravelRouteSet,
    #                                                                                  transportation_device_set=my_config.transportation_device_set,
    #                                                                                  charging_station_set=my_config.charging_station_set
    #                                                                                  )

    # my_occupancy = loadprofilegenerator_utsp_connector.UtspLpgConnector(config=my_occupancy_config, my_simulation_parameters=my_simulation_parameters)
    # my_sim.add_component(my_occupancy)

    # Build occupancy (fro basic household example)
    my_occupancy_config = loadprofilegenerator_connector.OccupancyConfig(
        profile_name=occupancy_profile, name="Occupancy"
    )
    my_occupancy = loadprofilegenerator_connector.Occupancy(
        config=my_occupancy_config, my_simulation_parameters=my_simulation_parameters
    )
    my_sim.add_component(my_occupancy)

    # Build Weather
    my_weather_config = weather.WeatherConfig.get_default(
        location_entry=weather.LocationEnum.Aachen
    )
    my_weather = weather.Weather(
        config=my_weather_config, my_simulation_parameters=my_simulation_parameters
    )
    my_sim.add_component(my_weather)

    # Build Gasheater
    my_gasheater_config = generic_gas_heater.GenericGasHeaterConfig(
        temperature_delta_in_celsius=temperature_delta_in_celsius,
        maximal_power_in_watt=maximal_power_in_watt,
        is_modulating=is_modulating,
        minimal_thermal_power_in_watt=minimal_thermal_power_in_watt,
        maximal_thermal_power_in_watt=maximal_thermal_power_in_watt,
        eff_th_min=eff_th_min,
        eff_th_max=eff_th_max,
        delta_temperature_in_celsius=delta_temperature_in_celsius,
        maximal_mass_flow_in_kilogram_per_second=maximal_mass_flow_in_kilogram_per_second,
        maximal_temperature_in_celsius=maximal_temperature_in_celsius,
    )
    my_gasheater = generic_gas_heater.GasHeater(
        config=my_gasheater_config, my_simulation_parameters=my_simulation_parameters
    )

    # # electricity grid
    # my_base_electricity_load_profile = sumbuilder.ElectricityGrid(name="BaseLoad", grid=[my_occupancy, "Subtract", my_photovoltaic_system],
    #                                                               my_simulation_parameters=my_simulation_parameters)
    # my_sim.add_component(my_base_electricity_load_profile)

    # Build building
    my_building_config = building.BuildingConfig(
        building_code=building_code,
        building_heat_capacity_class=building_class,
        initial_internal_temperature_in_celsius=initial_temperature,
        heating_reference_temperature_in_celsius=heating_reference_temperature,
        name="Building1",
        total_base_area_in_m2=my_config.total_base_area_in_m2,
        absolute_conditioned_floor_area_in_m2=absolute_conditioned_floor_area,
    )
    my_building = building.Building(
        config=my_building_config, my_simulation_parameters=my_simulation_parameters
    )
    my_building.connect_only_predefined_connections(my_weather)
    my_building.connect_only_predefined_connections(my_occupancy)
    my_sim.add_component(my_building)

    my_building.connect_input(
        my_building.ThermalEnergyDelivered,
        my_gasheater.component_name,
        my_gasheater.thermal_output_power_channel,
    )
