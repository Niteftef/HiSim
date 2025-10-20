"""Simple Water Storage Module for dhw storage and hot water storage for heating."""

# clean
# Owned
import importlib
from dataclasses import dataclass
from typing import List, Any, Tuple, Union, Optional
from enum import IntEnum
import numpy as np
import pandas as pd
from dataclasses_json import dataclass_json

import hisim.component as cp
from hisim import loadtypes as lt
from hisim import utils
from hisim.component import (
    SingleTimeStepValues,
    ComponentInput,
    ComponentOutput,
    OpexCostDataClass,
    DisplayConfig,
    CapexCostDataClass,
)
from hisim.components.configuration import PhysicsConfig
from hisim.components import configuration
from hisim.sim_repository_singleton import SingletonSimRepository, SingletonDictKeyEnum
from hisim.simulationparameters import SimulationParameters
from hisim.postprocessing.kpi_computation.kpi_structure import KpiTagEnumClass, KpiEntry, KpiHelperClass
from hisim.postprocessing.cost_and_emission_computation.capex_computation import CapexComputationHelperFunctions

__authors__ = "Jonas Hoppe"
__copyright__ = ""
__credits__ = [""]
__license__ = ""
__version__ = ""
__maintainer__ = ""
__email__ = ""
__status__ = ""


class HotWaterStorageSizingEnum(IntEnum):
    """Set Simple Hot Water Storage sizing options."""

    SIZE_ACCORDING_TO_HEAT_PUMP = 1
    SIZE_ACCORDING_TO_GENERAL_HEATING_SYSTEM = 2
    SIZE_ACCORDING_TO_GAS_HEATER = 3
    SIZE_ACCORDING_TO_PELLET_HEATING = 4
    SIZE_ACCORDING_TO_WOOD_CHIP_HEATING = 5


class PositionHotWaterStorageInSystemSetup(IntEnum):
    """Set Simple Hot Water Storage Position options."""

    PARALLEL_TO_HEAT_SOURCE = 1
    SERIE_TO_HEAT_SOURCE = 2


@dataclass_json
@dataclass
class SimpleHotWaterStorageConfig(cp.ConfigBase):
    """Configuration of the SimpleHotWaterStorage class."""

    @classmethod
    def get_main_classname(cls):
        """Return the full class name of the base class."""
        return SimpleHotWaterStorage.get_full_classname()

    building_name: str
    name: str
    volume_heating_water_storage_in_liter: float
    heat_transfer_coefficient_in_watt_per_m2_per_kelvin: float
    heat_exchanger_is_present: bool
    position_hot_water_storage_in_system: Union[PositionHotWaterStorageInSystemSetup, int]
    # it should be checked how much energy the storage lost during the simulated period (see guidelines below, p.2, accepted loss in kWh/days)
    # https://www.bdh-industrie.de/fileadmin/user_upload/ISH2019/Infoblaetter/Infoblatt_Nr_74_Energetische_Bewertung_Warmwasserspeicher.pdf
    #: CO2 footprint of investment in kg
    device_co2_footprint_in_kg: Optional[float]
    #: cost for investment in Euro
    investment_costs_in_euro: Optional[float]
    #: lifetime in years
    lifetime_in_years: Optional[float]
    # maintenance cost in euro per year
    maintenance_costs_in_euro_per_year: Optional[float]
    # subsidies as percentage of investment costs
    subsidy_as_percentage_of_investment_costs: Optional[float]

    @classmethod
    def get_default_simplehotwaterstorage_config(
        cls,
        building_name: str = "BUI1",
    ) -> "SimpleHotWaterStorageConfig":
        """Get a default simplehotwaterstorage config."""
        volume_heating_water_storage_in_liter: float = 500
        position_hot_water_storage_in_system: Union[PositionHotWaterStorageInSystemSetup, int] = (
            PositionHotWaterStorageInSystemSetup.PARALLEL_TO_HEAT_SOURCE
        )
        config = SimpleHotWaterStorageConfig(
            building_name=building_name,
            name="SimpleHotWaterStorage",
            volume_heating_water_storage_in_liter=volume_heating_water_storage_in_liter,
            heat_transfer_coefficient_in_watt_per_m2_per_kelvin=2.0,
            heat_exchanger_is_present=True,  # until now stratified mode is causing problems, so heat exchanger mode is recommended
            position_hot_water_storage_in_system=position_hot_water_storage_in_system,
            # capex and device emissions are calculated in get_cost_capex function by default
            device_co2_footprint_in_kg=None,
            investment_costs_in_euro=None,
            lifetime_in_years=None,
            maintenance_costs_in_euro_per_year=None,
            subsidy_as_percentage_of_investment_costs=None,
        )
        return config

    @classmethod
    def get_scaled_hot_water_storage(
        cls,
        max_thermal_power_in_watt_of_heating_system: float,
        name: str = "SimpleHotWaterStorage",
        building_name: str = "BUI1",
        temperature_difference_between_flow_and_return_in_celsius: float = 7.0,
        sizing_option: HotWaterStorageSizingEnum = HotWaterStorageSizingEnum.SIZE_ACCORDING_TO_GENERAL_HEATING_SYSTEM,
    ) -> "SimpleHotWaterStorageConfig":
        """Gets a default storage with scaling according to heating load of the building_name.

        The information for scaling the buffer storage is taken from the heating system guidelines from Buderus:
        https://www.baunetzwissen.de/heizung/fachwissen/speicher/dimensionierung-von-pufferspeichern-161296

        - If the heating system is a heat pump -> use formular:
        buffer storage size [m3] =
        (max. thermal power of heat pump [kW]* bridging time [h])
        /
        (spec. heat capacity water [Wh/(kg*K)]* temperature difference flow-return [K])
        with bridging time = 1h
        You can also check the paper:
        https://www.sciencedirect.com/science/article/pii/S2352152X2201533X?via%3Dihub.

        - If the heating system is something else (e.g. gasheater, ...), use approximation: 60 l per kW thermal power.
        """

        # if the used heating system is a heat pump use formular
        if sizing_option == HotWaterStorageSizingEnum.SIZE_ACCORDING_TO_HEAT_PUMP:
            volume_heating_water_storage_in_liter: float = (
                max_thermal_power_in_watt_of_heating_system
                * 1e-3
                / (
                    PhysicsConfig.get_properties_for_energy_carrier(
                        energy_carrier=lt.LoadTypes.WATER
                    ).specific_heat_capacity_in_watthour_per_kg_per_kelvin
                    * temperature_difference_between_flow_and_return_in_celsius
                )
            ) * 1000  # 1m3 = 1000l

        # otherwise use approximation: 60l per kw thermal power
        elif sizing_option == HotWaterStorageSizingEnum.SIZE_ACCORDING_TO_GENERAL_HEATING_SYSTEM:
            volume_heating_water_storage_in_liter = max_thermal_power_in_watt_of_heating_system / 1e3 * 60

        # large storage for pellet heating to avoid frequent on-off
        elif sizing_option == HotWaterStorageSizingEnum.SIZE_ACCORDING_TO_PELLET_HEATING:
            volume_heating_water_storage_in_liter = max_thermal_power_in_watt_of_heating_system / 1e3 * 40

        # large storage even more important than for pellets, as on-off behavior should be avoided
        elif sizing_option == HotWaterStorageSizingEnum.SIZE_ACCORDING_TO_WOOD_CHIP_HEATING:
            volume_heating_water_storage_in_liter = max_thermal_power_in_watt_of_heating_system / 1e3 * 100

        # or for gas heaters make hws smaller because gas heaters are a bigger inertia than heat pump
        elif sizing_option == HotWaterStorageSizingEnum.SIZE_ACCORDING_TO_GAS_HEATER:
            volume_heating_water_storage_in_liter = max_thermal_power_in_watt_of_heating_system / 1e3 * 20

        else:
            raise ValueError(f"Sizing option for Simple Hot Water Storage {sizing_option} is unvalid.")

        position_hot_water_storage_in_system: Union[PositionHotWaterStorageInSystemSetup, int] = (
            PositionHotWaterStorageInSystemSetup.PARALLEL_TO_HEAT_SOURCE
        )

        config = SimpleHotWaterStorageConfig(
            building_name=building_name,
            name=name,
            volume_heating_water_storage_in_liter=round(volume_heating_water_storage_in_liter, 2),
            heat_transfer_coefficient_in_watt_per_m2_per_kelvin=2.0,
            heat_exchanger_is_present=True,  # until now stratified mode is causing problems, so heat exchanger mode is recommended
            position_hot_water_storage_in_system=position_hot_water_storage_in_system,
            # capex and device emissions are calculated in get_cost_capex function by default
            device_co2_footprint_in_kg=None,
            investment_costs_in_euro=None,
            lifetime_in_years=None,
            maintenance_costs_in_euro_per_year=None,
            subsidy_as_percentage_of_investment_costs=None,
        )
        return config


@dataclass_json
@dataclass
class SimpleHotWaterStorageControllerConfig(cp.ConfigBase):
    """Configuration of the SimpleHotWaterStorageController class."""

    @classmethod
    def get_main_classname(cls):
        """Return the full class name of the base class."""
        return SimpleHotWaterStorageController.get_full_classname()

    building_name: str
    name: str

    @classmethod
    def get_default_simplehotwaterstoragecontroller_config(
        cls,
    ) -> Any:
        """Get a default simplehotwaterstorage controller config."""
        config = SimpleHotWaterStorageControllerConfig(
            building_name="BUI1",
            name="SimpleHotWaterStorageController",
        )
        return config


@dataclass_json
@dataclass
class SimpleDHWStorageConfig(cp.ConfigBase):
    """Configuration of the SimpleHotWaterStorage class."""

    @classmethod
    def get_main_classname(cls):
        """Return the full class name of the base class."""
        return SimpleDHWStorage.get_full_classname()

    building_name: str
    name: str
    volume_heating_water_storage_in_liter: float
    heat_transfer_coefficient_in_watt_per_m2_per_kelvin: float
    #: CO2 footprint of investment in kg
    device_co2_footprint_in_kg: Optional[float]
    #: cost for investment in Euro
    investment_costs_in_euro: Optional[float]
    #: lifetime in years
    lifetime_in_years: Optional[float]
    # maintenance cost in euro per year
    maintenance_costs_in_euro_per_year: Optional[float]
    # subsidies as percentage of investment costs
    subsidy_as_percentage_of_investment_costs: Optional[float]

    @classmethod
    def get_default_simpledhwstorage_config(
        cls,
        building_name: str = "BUI1",
    ) -> "SimpleDHWStorageConfig":
        """Get a default simplehotwaterstorage config."""
        volume_heating_water_storage_in_liter: float = 250

        config = SimpleDHWStorageConfig(
            building_name=building_name,
            name="DHWStorage",
            volume_heating_water_storage_in_liter=volume_heating_water_storage_in_liter,
            heat_transfer_coefficient_in_watt_per_m2_per_kelvin=0.36,
            # capex and device emissions are calculated in get_cost_capex function by default
            device_co2_footprint_in_kg=None,
            investment_costs_in_euro=None,
            lifetime_in_years=None,
            maintenance_costs_in_euro_per_year=None,
            subsidy_as_percentage_of_investment_costs=None,
        )
        return config

    @classmethod
    def get_scaled_dhw_storage(
        cls,
        number_of_apartments: int = 1,
        default_volume_in_liter: float = 250.0,
        name: str = "DHWStorage",
        building_name: str = "BUI1",
    ) -> "SimpleDHWStorageConfig":
        """Gets a default storage with scaling according to number of apartments."""

        # if the used heating system is a heat pump use formular

        volume = default_volume_in_liter * max(number_of_apartments, 1)
        config = SimpleDHWStorageConfig(
            building_name=building_name,
            name=name,
            volume_heating_water_storage_in_liter=volume,
            heat_transfer_coefficient_in_watt_per_m2_per_kelvin=0.36,
            # capex and device emissions are calculated in get_cost_capex function by default
            device_co2_footprint_in_kg=None,
            investment_costs_in_euro=None,
            lifetime_in_years=None,
            maintenance_costs_in_euro_per_year=None,
            subsidy_as_percentage_of_investment_costs=None,
        )
        return config


@dataclass
class SimpleWaterStorageState:
    """SimpleHotWaterStorageState class."""

    mean_water_temperature_in_celsius: float = 25.0
    temperature_loss_in_celsius_per_timestep: float = 0.0
    heat_loss_in_watt: float = 0.0

    def self_copy(self):
        """Copy the Simple Hot Water Storage State."""
        return SimpleWaterStorageState(
            self.mean_water_temperature_in_celsius,
            self.temperature_loss_in_celsius_per_timestep,
            self.heat_loss_in_watt,
        )


class SimpleWaterStorage(cp.Component):
    """SimpleWaterStorage class with generic functions."""

    @utils.measure_execution_time
    def __init__(
        self,
        my_simulation_parameters: SimulationParameters,
        name: str,
        my_config: cp.ConfigBase,
        my_display_config: DisplayConfig,
    ) -> None:
        """Construct all the neccessary attributes."""
        super().__init__(name, my_simulation_parameters, my_config, my_display_config)
        self.my_simulation_parameters = my_simulation_parameters
        self.seconds_per_timestep = my_simulation_parameters.seconds_per_timestep


    def calculate_mean_water_temperature_in_water_storage(
        self,
        water_temperature_input_of_secondary_side_in_celsius: float,
        water_temperature_from_heat_generator_in_celsius: float,
        mass_of_input_water_flows_from_heat_generator_in_kg: float,
        mass_of_input_water_flows_of_secondary_side_in_kg: float,
        water_mass_in_storage_in_kg: float,
        previous_mean_water_temperature_in_water_storage_in_celsius: float,
        water_temperature_from_secondary_heat_generator_in_celsius: float = 0,
        mass_of_input_water_flows_from_secondary_heat_generator_in_kg: float = 0,
    ) -> float:
        """Calculate the mean temperature of the water in the water boiler."""
        # prepare
        t_prev = previous_mean_water_temperature_in_water_storage_in_celsius
        mass_sum_inputs = (mass_of_input_water_flows_from_heat_generator_in_kg
            + mass_of_input_water_flows_from_secondary_heat_generator_in_kg
            + mass_of_input_water_flows_of_secondary_side_in_kg)
        mass_sum = mass_sum_inputs + water_mass_in_storage_in_kg
        # first calc the (weighted) average temperature of the storage + all inflows        
        t_intermediate = (
            water_mass_in_storage_in_kg * previous_mean_water_temperature_in_water_storage_in_celsius
            + mass_of_input_water_flows_from_heat_generator_in_kg * water_temperature_from_heat_generator_in_celsius
            + mass_of_input_water_flows_from_secondary_heat_generator_in_kg * water_temperature_from_secondary_heat_generator_in_celsius
            + mass_of_input_water_flows_of_secondary_side_in_kg * water_temperature_input_of_secondary_side_in_celsius
        ) / mass_sum
        # now the t_intermediate is also the weighted average of the final temperature
        # plus all outflows (they are all at t_prev). Solving for final temperature gives:
        # ! This leads to a swinging system, I can't use that unfortunately
        # result = (t_intermediate * mass_sum - t_prev * mass_sum_inputs) / water_mass_in_storage_in_kg
        result = t_intermediate
        return result

    def calculate_mixing_factor_for_water_temperature_outputs(self) -> Any:
        """Calculate mixing factor for water outputs."""

        # mixing factor depends on seconds per timestep
        # if one timestep = 1h (3600s) or more, the factor for the water storage portion is one

        if 0 <= self.seconds_per_timestep <= 3600:
            factor_for_water_storage_portion = self.seconds_per_timestep / 3600
            factor_for_water_input_portion = 1 - factor_for_water_storage_portion

        elif self.seconds_per_timestep > 3600:
            factor_for_water_storage_portion = 1
            factor_for_water_input_portion = 0

        else:
            raise ValueError("unknown value for seconds per timestep")

        return factor_for_water_storage_portion, factor_for_water_input_portion

    def calculate_water_output_temperature(
        self,
        t_water_mean_in_water_storage_in_c: float,
        mixing_factor_water_storage_portion: float,
        mixing_factor_water_input_portion: float,
        water_input_temperature_in_celsius: float,
    ) -> float:
        """Calculate the water output temperature of the water storage."""

        water_temperature_output_in_celsius = (
            mixing_factor_water_input_portion * water_input_temperature_in_celsius
            + mixing_factor_water_storage_portion * t_water_mean_in_water_storage_in_c
        )

        return water_temperature_output_in_celsius

    def calculate_heat_loss_and_temperature_loss(
        self,
        storage_surface_in_m2: float,
        mean_water_temperature_in_water_storage_in_celsius: float,
        heat_transfer_coefficient_in_watt_per_m2_per_kelvin: float,
        ambient_temperature_in_celsius: float,
        mass_in_storage_in_kg: float,
    ) -> Tuple[float, float]:
        """Calculates the heat loss in watt and the temperature loss in Kelvin per second
        of the storage and the water inside the storage."""

        heat_loss_in_watt = self.calculate_heat_loss_in_watt(
            mean_temperature_in_storage_in_celsius=mean_water_temperature_in_water_storage_in_celsius,
            storage_surface_in_m2=storage_surface_in_m2,
            heat_transfer_coefficient_in_watt_per_m2_per_kelvin=heat_transfer_coefficient_in_watt_per_m2_per_kelvin,
            ambient_temperature_in_celsius=ambient_temperature_in_celsius)

        # basis here: Q = m * cw * delta temperature, temperature loss is another term for delta temperature here
        c_p = PhysicsConfig.get_properties_for_energy_carrier(lt.LoadTypes.WATER
            ).specific_heat_capacity_in_joule_per_kg_per_kelvin
        t_loss_in_K_per_s = heat_loss_in_watt / (c_p * mass_in_storage_in_kg)

        return heat_loss_in_watt, t_loss_in_K_per_s

    def calculate_heat_loss_in_watt(
        self,
        storage_surface_in_m2: float,
        mean_temperature_in_storage_in_celsius: float,
        heat_transfer_coefficient_in_watt_per_m2_per_kelvin: float,
        ambient_temperature_in_celsius: float,
    ) -> float:
        """Calculate the current heat loss.

        It is dependent on storage surface area and current water temperature as well as heat transfer coefficient and ambient temperature.
        """

        # loss = heat coeff * surface * delta temperature
        heat_loss_in_watt = (
            heat_transfer_coefficient_in_watt_per_m2_per_kelvin
            * storage_surface_in_m2
            * (mean_temperature_in_storage_in_celsius - ambient_temperature_in_celsius)
        )
        return heat_loss_in_watt

    def calculate_surface_area_of_storage(self, storage_volume_in_liter: float) -> float:
        """Calculate the surface area of the storage which is assumed to be a cylinder."""

        storage_volume_in_m3 = storage_volume_in_liter * 1e-3
        # volume = r^2 * pi * h = r^2 * pi * 4r = 4 * r^3 * pi
        radius_of_storage_in_m = (storage_volume_in_m3 / (4 * np.pi)) ** (1 / 3)

        # lateral surface = 2 * pi * r * h (h=4*r here)
        lateral_surface_in_m2 = 2 * radius_of_storage_in_m * np.pi * (4 * radius_of_storage_in_m)
        # circle surface
        circle_surface_in_m2 = np.pi * radius_of_storage_in_m**2

        # total storage surface
        # cylinder surface area = lateral surface +  2 * circle surface
        storage_surface_in_m2 = lateral_surface_in_m2 + 2 * circle_surface_in_m2

        return float(storage_surface_in_m2)

    #########################################################################################################################################################

    def calc_thermal_energy(self, t_water: float, mass: float) -> float:
        """Calculate thermal energy in Wh of water. t_water has to be in °C or K, mass in kg.
        The thermal energy in storage is calculated with respect to 0 °C, so you can 
        simply give the mean temperature in °C. You can also give a temperature difference
        to calculate a thermal energy that was exchanged."""
        c_p = PhysicsConfig.get_properties_for_energy_carrier(lt.LoadTypes.WATER
            ).specific_heat_capacity_in_joule_per_kg_per_kelvin
        return c_p * mass * t_water / 3600 # 1 Wh = 3600 J

    def calc_thermal_power(self, mass_flow: float, t_diff: float) -> float:
        """Calculate thermal power in W of a water flow. 
        mass_flow has to be in kg per s. t_diff has to be in °C or K. """
        c_p = PhysicsConfig.get_properties_for_energy_carrier(lt.LoadTypes.WATER
            ).specific_heat_capacity_in_joule_per_kg_per_kelvin
        return c_p * mass_flow * t_diff


class SimpleHotWaterStorage(SimpleWaterStorage):
    """SimpleHotWaterStorage class."""

    # Input
    # A hot water storage can be used also with more than one heat generator. In this case you need to add a new input and output.
    WaterTemperatureToHeatDistribution = "WaterTemperatureToHeatDistribution"
    WaterTemperatureFromHeatDistribution = "WaterTemperatureFromHeatDistribution"
    WaterTemperatureFromHeatGenerator = "WaterTemperatureFromHeatGenerator"
    WaterTemperatureFromSecondaryHeatGenerator = "WaterTemperatureFromSecondaryHeatGenerator"
    WaterMassFlowRateFromSecondaryHeatGenerator = "WaterMassFlowRateFromSecondaryHeatGenerator"
    WaterMassFlowRateFromHeatGenerator = "WaterMassFlowRateFromHeatGenerator"
    WaterMassFlowRateFromHeatDistributionSystem = "WaterMassFlowRateFromHeatDistributionSystem"
    State = "State"

    # Output

    WaterTemperatureToHeatGenerator = "WaterTemperatureToHeatGenerator"
    WaterTemperatureToSecondaryHeatGenerator = "WaterTemperatureToSecondaryHeatGenerator"
    WaterMeanTemperatureInStorage = "WaterMeanTemperatureInStorage"

    # make some more outputs for testing simple storage

    ThermalEnergyInStorage = "ThermalEnergyInStorage"
    ThermalEnergyFromHeatGenerator = "ThermalEnergyFromHeatGenerator"
    ThermalEnergyFromSecondaryHeatGenerator = "ThermalEnergyFromSecondaryHeatGenerator"
    ThermalEnergyFromHeatDistribution = "ThermalEnergyFromHeatDistribution"
    ThermalEnergyIncreaseInStorage = "ThermalEnergyIncreaseInStorage"

    StandbyHeatLoss = "StandbyHeatLoss"
    ThermalPowerConsumptionHeatDistribution = "ThermalPowerConsumptionHeatDistribution"
    ThermalPowerFromHeatGenerator = "ThermalPowerFromHeatGenerator"
    ThermalPowerFromSecondaryHeatGenerator = "ThermalPowerFromSecondaryHeatGenerator"

    @utils.measure_execution_time
    def __init__(
        self,
        my_simulation_parameters: SimulationParameters,
        config: SimpleHotWaterStorageConfig,
        my_display_config: DisplayConfig = DisplayConfig(),
    ) -> None:
        """Construct all the neccessary attributes."""
        self.my_simulation_parameters = my_simulation_parameters
        self.config = config
        component_name = self.get_component_name()
        super().__init__(
            name=component_name,
            my_simulation_parameters=my_simulation_parameters,
            my_config=config,
            my_display_config=my_display_config,
        )
        # =================================================================================================================================
        # Initialization of variables
        self.seconds_per_timestep = my_simulation_parameters.seconds_per_timestep
        self.waterstorageconfig = config

        self.mean_water_temperature_in_water_storage_in_celsius: float = 35

        if SingletonSimRepository().exist_entry(key=SingletonDictKeyEnum.WATERMASSFLOWRATEOFHEATGENERATOR):
            self.water_mass_flow_rate_from_hg_in_kg_per_s_from_singleton_sim_repo = (
                SingletonSimRepository().get_entry(key=SingletonDictKeyEnum.WATERMASSFLOWRATEOFHEATGENERATOR)
            )
        else:
            self.water_mass_flow_rate_from_hg_in_kg_per_s_from_singleton_sim_repo = None

        self.position_hot_water_storage_in_system = self.waterstorageconfig.position_hot_water_storage_in_system
        self.build(heat_exchanger_is_present=self.waterstorageconfig.heat_exchanger_is_present)

        self.state: SimpleWaterStorageState = SimpleWaterStorageState(
            mean_water_temperature_in_celsius=self.mean_water_temperature_in_water_storage_in_celsius,
            temperature_loss_in_celsius_per_timestep=0,
        )
        self.previous_state = self.state.self_copy()

        # =================================================================================================================================
        # Input channels

        self.water_temperature_heat_distribution_system_input_channel: ComponentInput = self.add_input(
            self.component_name,
            self.WaterTemperatureFromHeatDistribution,
            lt.LoadTypes.TEMPERATURE,
            lt.Units.CELSIUS,
            True,
        )
        self.water_mass_flow_rate_heat_distribution_system_input_channel: ComponentInput = self.add_input(
            self.component_name,
            self.WaterMassFlowRateFromHeatDistributionSystem,
            lt.LoadTypes.WARM_WATER,
            lt.Units.KG_PER_SEC,
            False,
        )

        if self.position_hot_water_storage_in_system in [PositionHotWaterStorageInSystemSetup.PARALLEL_TO_HEAT_SOURCE]:
            self.water_temperature_heat_generator_input_channel: ComponentInput = self.add_input(
                self.component_name,
                self.WaterTemperatureFromHeatGenerator,
                lt.LoadTypes.TEMPERATURE,
                lt.Units.CELSIUS,
                True,
            )

            self.water_mass_flow_rate_heat_generator_input_channel: ComponentInput = self.add_input(
                self.component_name,
                self.WaterMassFlowRateFromHeatGenerator,
                lt.LoadTypes.WARM_WATER,
                lt.Units.KG_PER_SEC,
                False,
            )
            self.water_temperature_secondary_heat_generator_input_channel: ComponentInput = self.add_input(
                self.component_name,
                self.WaterTemperatureFromSecondaryHeatGenerator,
                lt.LoadTypes.TEMPERATURE,
                lt.Units.CELSIUS,
                False,
            )
            self.water_mass_flow_rate_secondary_heat_generator_input_channel: ComponentInput = self.add_input(
                self.component_name,
                self.WaterMassFlowRateFromSecondaryHeatGenerator,
                lt.LoadTypes.WARM_WATER,
                lt.Units.KG_PER_SEC,
                False,
            )

        self.state_channel: cp.ComponentInput = self.add_input(
            self.component_name, self.State, lt.LoadTypes.ANY, lt.Units.ANY, False
        )

        # Output channels

        self.water_temperature_heat_distribution_system_output_channel: ComponentOutput = self.add_output(
            self.component_name,
            self.WaterTemperatureToHeatDistribution,
            lt.LoadTypes.WATER,
            lt.Units.CELSIUS,
            output_description=f"here a description for {self.WaterTemperatureToHeatDistribution} will follow.",
        )

        self.water_temperature_heat_generator_output_channel: ComponentOutput = self.add_output(
            self.component_name,
            self.WaterTemperatureToHeatGenerator,
            lt.LoadTypes.WATER,
            lt.Units.CELSIUS,
            output_description=f"here a description for {self.WaterTemperatureToHeatGenerator} will follow.",
        )

        self.water_temperature_secondary_heat_generator_output_channel: ComponentOutput = self.add_output(
            self.component_name,
            self.WaterTemperatureToSecondaryHeatGenerator,
            lt.LoadTypes.WATER,
            lt.Units.CELSIUS,
            output_description=f"here a description for {self.WaterTemperatureToSecondaryHeatGenerator} will follow.",
        )

        self.water_temperature_mean_channel: ComponentOutput = self.add_output(
            self.component_name,
            self.WaterMeanTemperatureInStorage,
            lt.LoadTypes.WATER,
            lt.Units.CELSIUS,
            output_description=f"here a description for {self.WaterMeanTemperatureInStorage} will follow.",
        )

        self.thermal_energy_in_storage_channel: ComponentOutput = self.add_output(
            self.component_name,
            self.ThermalEnergyInStorage,
            lt.LoadTypes.HEATING,
            lt.Units.WATT_HOUR,
            output_description=f"here a description for {self.ThermalEnergyInStorage} will follow.",
        )
        self.thermal_energy_from_heat_generator_channel: ComponentOutput = self.add_output(
            self.component_name,
            self.ThermalEnergyFromHeatGenerator,
            lt.LoadTypes.HEATING,
            lt.Units.WATT_HOUR,
            output_description=f"here a description for {self.ThermalEnergyFromHeatGenerator} will follow.",
            postprocessing_flag=[lt.OutputPostprocessingRules.DISPLAY_IN_WEBTOOL],
        )
        self.thermal_energy_from_secondary_heat_generator_channel: ComponentOutput = self.add_output(
            self.component_name,
            self.ThermalEnergyFromSecondaryHeatGenerator,
            lt.LoadTypes.HEATING,
            lt.Units.WATT_HOUR,
            output_description=f"here a description for {self.ThermalEnergyFromSecondaryHeatGenerator} will follow.",
            postprocessing_flag=[lt.OutputPostprocessingRules.DISPLAY_IN_WEBTOOL],
        )
        self.thermal_energy_input_heat_distribution_system_channel: ComponentOutput = self.add_output(
            self.component_name,
            self.ThermalEnergyFromHeatDistribution,
            lt.LoadTypes.HEATING,
            lt.Units.WATT_HOUR,
            output_description=f"here a description for {self.ThermalEnergyFromHeatDistribution} will follow.",
            postprocessing_flag=[lt.OutputPostprocessingRules.DISPLAY_IN_WEBTOOL],
        )
        self.thermal_energy_increase_in_storage_channel: ComponentOutput = self.add_output(
            self.component_name,
            self.ThermalEnergyIncreaseInStorage,
            lt.LoadTypes.HEATING,
            lt.Units.WATT_HOUR,
            output_description=f"here a description for {self.ThermalEnergyIncreaseInStorage} will follow.",
        )
        self.stand_by_heat_loss_channel: ComponentOutput = self.add_output(
            self.component_name,
            self.StandbyHeatLoss,
            lt.LoadTypes.HEATING,
            lt.Units.WATT,
            output_description=f"here a description for {self.StandbyHeatLoss} will follow.",
        )
        self.thermal_power_heat_distribution_channel: ComponentOutput = self.add_output(
            self.component_name,
            self.ThermalPowerConsumptionHeatDistribution,
            lt.LoadTypes.HEATING,
            lt.Units.WATT,
            output_description=f"here a description for {self.ThermalPowerConsumptionHeatDistribution} will follow.",
        )

        self.thermal_power_from_heat_generator_channel: ComponentOutput = self.add_output(
            self.component_name,
            self.ThermalPowerFromHeatGenerator,
            lt.LoadTypes.HEATING,
            lt.Units.WATT,
            output_description=f"here a description for {self.ThermalPowerFromHeatGenerator} will follow.",
        )

        self.thermal_power_from_secondary_heat_generator_channel: ComponentOutput = self.add_output(
            self.component_name,
            self.ThermalPowerFromSecondaryHeatGenerator,
            lt.LoadTypes.HEATING,
            lt.Units.WATT,
            output_description=f"here a description for {self.ThermalPowerFromSecondaryHeatGenerator} will follow.",
        )

        self.add_default_connections(self.get_default_connections_from_heat_distribution_system())
        self.add_default_connections(self.get_default_connections_from_advanced_heat_pump())
        self.add_default_connections(self.get_default_connections_from_more_advanced_heat_pump())
        self.add_default_connections(self.get_default_connections_from_generic_boiler())

    def get_default_connections_from_heat_distribution_system(
        self,
    ) -> List[cp.ComponentConnection]:
        """Get heat distribution default connections."""

        # use importlib for importing the other component in order to avoid circular-import errors
        component_module_name = "hisim.components.heat_distribution_system"
        component_module = importlib.import_module(name=component_module_name)
        component_class = getattr(component_module, "HeatDistribution")
        connections = []
        hds_classname = component_class.get_classname()
        connections.append(
            cp.ComponentConnection(
                SimpleHotWaterStorage.WaterTemperatureFromHeatDistribution,
                hds_classname,
                component_class.WaterTemperatureOutput,
            )
        )
        connections.append(
            cp.ComponentConnection(
                SimpleHotWaterStorage.WaterMassFlowRateFromHeatDistributionSystem,
                hds_classname,
                component_class.WaterMassFlowHDS,
            )
        )
        return connections

    def get_default_connections_from_advanced_heat_pump(
        self,
    ) -> List[cp.ComponentConnection]:
        """Get advanced het pump default connections."""

        # use importlib for importing the other component in order to avoid circular-import errors
        component_module_name = "hisim.components.advanced_heat_pump_hplib"
        component_module = importlib.import_module(name=component_module_name)
        component_class = getattr(component_module, "HeatPumpHplib")
        connections = []
        hp_classname = component_class.get_classname()
        connections.append(
            cp.ComponentConnection(
                SimpleHotWaterStorage.WaterTemperatureFromHeatGenerator,
                hp_classname,
                component_class.TemperatureOutput,
            )
        )
        connections.append(
            cp.ComponentConnection(
                SimpleHotWaterStorage.WaterMassFlowRateFromHeatGenerator,
                hp_classname,
                component_class.MassFlowOutput,
            )
        )
        return connections

    def get_default_connections_from_more_advanced_heat_pump(
        self,
    ) -> List[cp.ComponentConnection]:
        """Get advanced het pump default connections."""

        # use importlib for importing the other component in order to avoid circular-import errors
        component_module_name = "hisim.components.more_advanced_heat_pump_hplib"
        component_module = importlib.import_module(name=component_module_name)
        component_class = getattr(component_module, "MoreAdvancedHeatPumpHPLib")
        connections = []
        hp_classname = component_class.get_classname()
        connections.append(
            cp.ComponentConnection(
                SimpleHotWaterStorage.WaterTemperatureFromHeatGenerator,
                hp_classname,
                component_class.TemperatureOutputSH,
            )
        )
        connections.append(
            cp.ComponentConnection(
                SimpleHotWaterStorage.WaterMassFlowRateFromHeatGenerator,
                hp_classname,
                component_class.MassFlowOutputSH,
            )
        )
        return connections

    def get_default_connections_from_generic_boiler(self) -> List[cp.ComponentConnection]:
        """Get gasheater default connections."""

        # use importlib for importing the other component in order to avoid circular-import errors
        component_module_name = "hisim.components.generic_boiler"
        component_module = importlib.import_module(name=component_module_name)
        component_class = getattr(component_module, "GenericBoiler")
        connections = []
        gasheater_classname = component_class.get_classname()
        connections.append(
            cp.ComponentConnection(
                SimpleHotWaterStorage.WaterTemperatureFromHeatGenerator,
                gasheater_classname,
                component_class.WaterOutputTemperatureSh,
            )
        )
        connections.append(
            cp.ComponentConnection(
                SimpleHotWaterStorage.WaterMassFlowRateFromHeatGenerator,
                gasheater_classname,
                component_class.WaterOutputMassFlowSh,
            )
        )
        return connections

    def i_prepare_simulation(self) -> None:
        """Prepare the simulation."""
        pass

    def write_to_report(self) -> List[str]:
        """Write a report."""
        return self.waterstorageconfig.get_string_dict()

    def i_save_state(self) -> None:
        """Save the current state."""
        self.previous_state = self.state.self_copy()

    def i_restore_state(self) -> None:
        """Restore the previous state."""
        self.state = self.previous_state.self_copy()

    def i_doublecheck(self, timestep: int, stsv: SingleTimeStepValues) -> None:
        """Doublecheck."""
        pass

    def i_simulate(self, timestep: int, stsv: SingleTimeStepValues, force_convergence: bool) -> None:
        """Simulate the heating water storage.
        Note: The energy flows are """        
        # get inputs
        state_controller = stsv.get_input_value(self.state_channel)
        t_water_from_hds_in_c = stsv.get_input_value(
            self.water_temperature_heat_distribution_system_input_channel)
        water_mass_flow_rate_from_hds_in_kg_per_s = stsv.get_input_value(
            self.water_mass_flow_rate_heat_distribution_system_input_channel)

        t_water_from_hg_in_c = 0
        water_flow_from_hg_in_kg_per_s = 0
        t_water_from_hg2_in_c = 0
        water_flow_from_hg2_in_kg_per_s = 0
        if self.is_parallel():
            t_water_from_hg_in_c = stsv.get_input_value(
                self.water_temperature_heat_generator_input_channel)
            t_water_from_hg2_in_c = stsv.get_input_value(
                self.water_temperature_secondary_heat_generator_input_channel)
            # get water mass flow rate of heat generator either from singleton sim repo or from input value
            if self.water_mass_flow_rate_from_hg_in_kg_per_s_from_singleton_sim_repo is not None:
                water_flow_from_hg_in_kg_per_s = (
                    self.water_mass_flow_rate_from_hg_in_kg_per_s_from_singleton_sim_repo)
            else:
                water_flow_from_hg_in_kg_per_s = stsv.get_input_value(
                    self.water_mass_flow_rate_heat_generator_input_channel)
                water_flow_from_hg2_in_kg_per_s = stsv.get_input_value(
                    self.water_mass_flow_rate_secondary_heat_generator_input_channel)

        # check water temperature limits
#        if not (0 < self.mean_water_temperature_in_water_storage_in_celsius < 90):
#            raise ValueError(f"""The water temperature in the water storage is with 
#                {self.mean_water_temperature_in_water_storage_in_celsius}°C way too high or too low.""")

        # get water masses from flow rates for simplicity later
        water_mass_from_hg_in_kg = water_flow_from_hg_in_kg_per_s * self.seconds_per_timestep
        water_mass_from_hg2_in_kg = water_flow_from_hg2_in_kg_per_s * self.seconds_per_timestep
        water_mass_from_hds_in_kg = water_mass_flow_rate_from_hds_in_kg_per_s * self.seconds_per_timestep

        # ----------------------------------------------------------------------------------------
        # ----- Calculations ---------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------

        # calc temperatures

        # with heat exchanger in water storage perfect heat exchange is possible
        if self.heat_exchanger_is_present:
            t_water_to_hds_in_c = self.state.mean_water_temperature_in_celsius
            t_water_to_hg_in_c = self.state.mean_water_temperature_in_celsius
            t_water_to_hg2_in_c = self.state.mean_water_temperature_in_celsius
        # otherwise the water in the water storage is more stratified, which demands some more calculations
        else:
            # state controller is 1 if the heat generator delivers a mass flow rate input
            if state_controller == 1:
                # hds gets water from heat generator (if heat generator is not off, mass flow is not zero)
                t_water_to_hds_in_c = self.calculate_water_output_temperature(
                    t_water_mean_in_water_storage_in_c=self.state.mean_water_temperature_in_celsius,
                    mixing_factor_water_input_portion=self.factor_for_water_input_portion,
                    mixing_factor_water_storage_portion=self.factor_for_water_storage_portion,
                    water_input_temperature_in_celsius=t_water_from_hg_in_c)
                # heat generators get water from hds (if heat generator is not off, mass flow is not zero)
                t_water_to_hg_in_c = self.calculate_water_output_temperature(
                    t_water_mean_in_water_storage_in_c=self.state.mean_water_temperature_in_celsius,
                    mixing_factor_water_input_portion=self.factor_for_water_input_portion,
                    mixing_factor_water_storage_portion=self.factor_for_water_storage_portion,
                    water_input_temperature_in_celsius=t_water_from_hds_in_c)
                t_water_to_hg2_in_c = self.calculate_water_output_temperature(
                    t_water_mean_in_water_storage_in_c=self.state.mean_water_temperature_in_celsius,
                    mixing_factor_water_input_portion=self.factor_for_water_input_portion,
                    mixing_factor_water_storage_portion=self.factor_for_water_storage_portion,
                    water_input_temperature_in_celsius=t_water_from_hds_in_c)
            # no water coming from heat generator, hds gets mean water and heat generator gets still water from hds
            elif state_controller == 0:
                t_water_to_hds_in_c = self.state.mean_water_temperature_in_celsius
                t_water_to_hg_in_c = self.calculate_water_output_temperature(
                    t_water_mean_in_water_storage_in_c=self.state.mean_water_temperature_in_celsius,
                    mixing_factor_water_input_portion=self.factor_for_water_input_portion,
                    mixing_factor_water_storage_portion=self.factor_for_water_storage_portion,
                    water_input_temperature_in_celsius=t_water_from_hds_in_c,)
                t_water_to_hg2_in_c = self.calculate_water_output_temperature(
                    t_water_mean_in_water_storage_in_c=self.state.mean_water_temperature_in_celsius,
                    mixing_factor_water_input_portion=self.factor_for_water_input_portion,
                    mixing_factor_water_storage_portion=self.factor_for_water_storage_portion,
                    water_input_temperature_in_celsius=t_water_from_hds_in_c,)
            else:
                raise ValueError("unknown storage controller state.")

        # new mean temperature in storage
#!        a1_vol = self.config.volume_heating_water_storage_in_liter
#!        a2_t_prev = self.state.mean_water_temperature_in_celsius # ! testing
#!        a3_e_prev = self.calc_thermal_energy(a2_t_prev, a1_vol)
        self.mean_water_temperature_in_water_storage_in_celsius = self.calculate_mean_water_temperature_in_water_storage(
            water_temperature_input_of_secondary_side_in_celsius=t_water_from_hds_in_c,
            water_temperature_from_heat_generator_in_celsius=t_water_from_hg_in_c,
            water_mass_in_storage_in_kg=self.water_mass_in_storage_in_kg,
            mass_of_input_water_flows_from_heat_generator_in_kg=water_mass_from_hg_in_kg,
            water_temperature_from_secondary_heat_generator_in_celsius=t_water_from_hg2_in_c,
            mass_of_input_water_flows_from_secondary_heat_generator_in_kg=water_mass_from_hg2_in_kg,
            mass_of_input_water_flows_of_secondary_side_in_kg=water_mass_from_hds_in_kg,
            previous_mean_water_temperature_in_water_storage_in_celsius=self.state.mean_water_temperature_in_celsius)
#!        a4_t_res = self.mean_water_temperature_in_water_storage_in_celsius # ! testing
#!        a5_e_res = self.calc_thermal_energy(a4_t_res, a1_vol)
#!        a6_t_diff = a4_t_res - a2_t_prev
#!        a7_e_diff = a5_e_res - a3_e_prev
#!        a8_e_diff_by_tdiff = self.calc_thermal_energy(a6_t_diff, a1_vol)

        # calc thermal power and energies
        p_therm_from_hg_in_W = self.calc_thermal_power(
            mass_flow=water_flow_from_hg_in_kg_per_s,
            t_diff=(t_water_from_hg_in_c-self.state.mean_water_temperature_in_celsius))
        p_therm_from_hg2_in_W = self.calc_thermal_power(
            mass_flow=water_flow_from_hg2_in_kg_per_s,
            t_diff=(t_water_from_hg2_in_c-self.state.mean_water_temperature_in_celsius))
        p_therm_to_hds_in_W = self.calc_thermal_power(
            mass_flow=water_mass_flow_rate_from_hds_in_kg_per_s,
            t_diff=(self.state.mean_water_temperature_in_celsius-t_water_from_hds_in_c))

        e_therm_in_storage_prev_in_Wh = self.calc_thermal_energy(
            t_water=self.state.mean_water_temperature_in_celsius, 
            mass=self.water_mass_in_storage_in_kg,)
        e_therm_in_storage_current_in_Wh = self.calc_thermal_energy(
            t_water=self.mean_water_temperature_in_water_storage_in_celsius,
            mass=self.water_mass_in_storage_in_kg,)

        e_therm_input_from_hg_in_Wh = p_therm_from_hg_in_W * self.seconds_per_timestep / 3600
        e_therm_input_from_hg2_in_Wh = p_therm_from_hg2_in_W * self.seconds_per_timestep / 3600
        e_therm_output_to_hds_in_Wh = p_therm_to_hds_in_W * self.seconds_per_timestep / 3600
        e_therm_increase_in_Wh = e_therm_in_storage_current_in_Wh - e_therm_in_storage_prev_in_Wh

        # ! check thermal energies and temperatures
#        e_increase_checker = -(e_therm_input_from_hg_in_Wh 
#                              + e_therm_input_from_hg2_in_Wh
#                              + e_therm_output_to_hds_in_Wh)
#        if not (0.99 < (e_therm_increase_in_Wh / e_increase_checker) < 1.01):
#            raise ValueError("Thermal energies do not match")
#        t_checker = (e_therm_in_storage_prev_in_Wh + e_increase_checker) / a1_vol / 4180 * 3600
#        if not (0.99 < (self.mean_water_temperature_in_water_storage_in_celsius / t_checker) < 1.01):
#            raise ValueError("Resulting mean temperature does not match")

        # ----------------------------------------------------------------------------------------
        # ----- Set outputs ----------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------

        if self.is_parallel():
            stsv.set_output_value(
                self.water_temperature_heat_distribution_system_output_channel,
                t_water_to_hds_in_c)

        stsv.set_output_value(
            self.water_temperature_heat_generator_output_channel,
            t_water_to_hg_in_c)
        stsv.set_output_value(
            self.water_temperature_secondary_heat_generator_output_channel,
            t_water_to_hg2_in_c)
        stsv.set_output_value(
            self.water_temperature_mean_channel,
            self.state.mean_water_temperature_in_celsius)
        stsv.set_output_value(
            self.thermal_energy_in_storage_channel,
            e_therm_in_storage_current_in_Wh)
        stsv.set_output_value(
            self.thermal_energy_from_heat_generator_channel,
            e_therm_input_from_hg_in_Wh)
        stsv.set_output_value(
            self.thermal_energy_from_secondary_heat_generator_channel,
            e_therm_input_from_hg2_in_Wh)
        stsv.set_output_value(
            self.thermal_energy_input_heat_distribution_system_channel,
            e_therm_output_to_hds_in_Wh)
        stsv.set_output_value(
            self.thermal_energy_increase_in_storage_channel,
            e_therm_increase_in_Wh)
        stsv.set_output_value(
            self.stand_by_heat_loss_channel,
            self.state.heat_loss_in_watt)
        stsv.set_output_value(
            self.thermal_power_heat_distribution_channel,
            p_therm_to_hds_in_W)
        stsv.set_output_value(
            self.thermal_power_from_heat_generator_channel,
            p_therm_from_hg_in_W)
        stsv.set_output_value(
            self.thermal_power_from_secondary_heat_generator_channel,
            p_therm_from_hg2_in_W)
        
        # ----------------------------------------------------------------------------------------
        # ----- Set new state --------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------

        heat_loss, temp_loss = self.calculate_heat_loss_and_temperature_loss(
            storage_surface_in_m2=self.storage_surface_in_m2,
            mean_water_temperature_in_water_storage_in_celsius=self.mean_water_temperature_in_water_storage_in_celsius,
            heat_transfer_coefficient_in_watt_per_m2_per_kelvin=self.heat_transfer_coefficient_in_watt_per_m2_per_kelvin,
            mass_in_storage_in_kg=self.water_mass_in_storage_in_kg,
            ambient_temperature_in_celsius=self.ambient_temperature_in_celsius)

        self.state.heat_loss_in_watt = heat_loss
        self.state.temperature_loss_in_celsius_per_timestep = temp_loss * self.seconds_per_timestep
        self.state.mean_water_temperature_in_celsius = (
            self.mean_water_temperature_in_water_storage_in_celsius - temp_loss)
        
# !      print(f"vol: {a1_vol: 4.2f}")
#        print(f"t_1: {a2_t_prev: 4.2f}")
#        print(f"e_1: {a3_e_prev: 4.2f}")
#        print(f"t_2: {a4_t_res: 4.2f}")
#        print(f"e_2: {a5_e_res: 4.2f}")
#        print(f"t_d: {a6_t_diff: 4.2f}")
#        print(f"e_d: {a7_e_diff: 4.2f}")
#        print(f"ed2: {a8_e_diff_by_tdiff: 4.2f}")
#        print("------------")

    def build(self, heat_exchanger_is_present: bool) -> None:
        """Build function.

        The function sets important constants an parameters for the calculations.
        """
        self.specific_heat_capacity_of_water_in_joule_per_kilogram_per_celsius = (
            PhysicsConfig.get_properties_for_energy_carrier(
                energy_carrier=lt.LoadTypes.WATER
            ).specific_heat_capacity_in_joule_per_kg_per_kelvin
        )
        self.specific_heat_capacity_of_water_in_watthour_per_kilogram_per_celsius = (
            PhysicsConfig.get_properties_for_energy_carrier(
                energy_carrier=lt.LoadTypes.WATER
            ).specific_heat_capacity_in_watthour_per_kg_per_kelvin
        )
        # https://www.internetchemie.info/chemie-lexikon/daten/w/wasser-dichtetabelle.php
        self.density_water_at_40_degree_celsius_in_kg_per_liter = 0.992

        # physical parameters of storage
        self.water_mass_in_storage_in_kg = (
            self.density_water_at_40_degree_celsius_in_kg_per_liter
            * self.waterstorageconfig.volume_heating_water_storage_in_liter
        )
        self.heat_transfer_coefficient_in_watt_per_m2_per_kelvin = (
            self.config.heat_transfer_coefficient_in_watt_per_m2_per_kelvin
        )
        self.storage_surface_in_m2 = self.calculate_surface_area_of_storage(
            storage_volume_in_liter=self.waterstorageconfig.volume_heating_water_storage_in_liter,
        )

        # the ambient temperature is here assumed as the basement temperature which is all year 17°C, this is where the water storage is located
        self.ambient_temperature_in_celsius = 20.0

        self.heat_exchanger_is_present = heat_exchanger_is_present
        # if heat exchanger is present, the heat is perfectly exchanged so the water output temperature corresponds to the mean temperature
        if self.heat_exchanger_is_present is True:
            (
                self.factor_for_water_storage_portion,
                self.factor_for_water_input_portion,
            ) = (1, 0)
        # if heat exchanger is not present, the water temperatures in the storage are more stratified
        # here a mixing factor is calcualted
        else:
            (
                self.factor_for_water_storage_portion,
                self.factor_for_water_input_portion,
            ) = self.calculate_mixing_factor_for_water_temperature_outputs()

    @staticmethod
    def get_cost_capex(
        config: SimpleHotWaterStorageConfig, simulation_parameters: SimulationParameters
    ) -> CapexCostDataClass:
        """Returns investment cost, CO2 emissions and lifetime."""
        kpi_tag = KpiTagEnumClass.STORAGE_HOT_WATER_SPACE_HEATING
        component_type = lt.ComponentType.THERMAL_ENERGY_STORAGE
        unit = lt.Units.LITER
        size_of_energy_system = config.volume_heating_water_storage_in_liter

        capex_cost_data_class = CapexComputationHelperFunctions.compute_capex_costs_and_emissions(
        simulation_parameters=simulation_parameters,
        component_type=component_type,
        unit=unit,
        size_of_energy_system=size_of_energy_system,
        config=config,
        kpi_tag=kpi_tag
        )
        config = CapexComputationHelperFunctions.overwrite_config_values_with_new_capex_values(config=config, capex_cost_data_class=capex_cost_data_class)

        return capex_cost_data_class

    def get_cost_opex(
        self,
        all_outputs: List,
        postprocessing_results: pd.DataFrame,
    ) -> OpexCostDataClass:
        # pylint: disable=unused-argument
        """Calculate OPEX costs, consisting of maintenance costs for hot water storage."""
        opex_cost_data_class = OpexCostDataClass(
            opex_energy_cost_in_euro=0,
            opex_maintenance_cost_in_euro=self.calc_maintenance_cost(),
            co2_footprint_in_kg=0,
            total_consumption_in_kwh=0,
            loadtype=lt.LoadTypes.ANY,
            kpi_tag=KpiTagEnumClass.STORAGE_HOT_WATER_SPACE_HEATING,
        )

        return opex_cost_data_class

    def get_component_kpi_entries(
        self,
        all_outputs: List,
        postprocessing_results: pd.DataFrame,
    ) -> List[KpiEntry]:
        """Calculates KPIs for the respective component and return all KPI entries as list."""
        list_of_kpi_entries: List[KpiEntry] = []
        for index, output in enumerate(all_outputs):
            if output.component_name == self.component_name:
                if output.field_name == self.StandbyHeatLoss and output.unit == lt.Units.WATT:
                    # calc heat loss
                    heat_loss_in_watt = postprocessing_results.iloc[:, index].loc[
                        postprocessing_results.iloc[:, index] > 0.0
                    ]
                    # get energy from power
                    heat_loss_in_kilowatt_hour = round(
                        KpiHelperClass.compute_total_energy_from_power_timeseries(
                            power_timeseries_in_watt=heat_loss_in_watt,
                            timeresolution=self.my_simulation_parameters.seconds_per_timestep,
                        ),
                        1,
                    )
                    heat_loss_entry = KpiEntry(
                        name="Standby heat loss of Hot water storage",
                        unit="kWh",
                        value=heat_loss_in_kilowatt_hour,
                        tag=KpiTagEnumClass.STORAGE_HOT_WATER_SPACE_HEATING,
                        description=self.component_name,
                    )
                    list_of_kpi_entries.append(heat_loss_entry)
        return list_of_kpi_entries

    # ----- helper functions -------------------------------------------------

    def is_parallel(self) -> bool:
        """Returns true if self.position_hot_water_storage_in_system is
        PARALLEL_TO_HEAT_SOURCE and false otherwise."""
        return (self.position_hot_water_storage_in_system == 
                PositionHotWaterStorageInSystemSetup.PARALLEL_TO_HEAT_SOURCE)


class SimpleHotWaterStorageController(cp.Component):
    """SimpleHotWaterStorageController Class."""

    # Inputs
    WaterMassFlowRateFromHeatGenerator = "WaterMassFlowRateFromHeatGenerator"

    # Outputs
    State = "State"

    def __init__(
        self,
        my_simulation_parameters: SimulationParameters,
        config: SimpleHotWaterStorageControllerConfig,
        my_display_config: DisplayConfig = DisplayConfig(),
    ) -> None:
        """Construct all the neccessary attributes."""

        self.my_simulation_parameters = my_simulation_parameters
        self.config = config
        component_name = self.get_component_name()
        super().__init__(
            name=component_name,
            my_simulation_parameters=my_simulation_parameters,
            my_config=config,
            my_display_config=my_display_config,
        )
        if SingletonSimRepository().exist_entry(key=SingletonDictKeyEnum.WATERMASSFLOWRATEOFHEATGENERATOR):
            self.water_mass_flow_rate_from_heat_generator_in_kg_per_second_from_singleton_sim_repo = (
                SingletonSimRepository().get_entry(key=SingletonDictKeyEnum.WATERMASSFLOWRATEOFHEATGENERATOR)
            )
        else:
            self.water_mass_flow_rate_from_heat_generator_in_kg_per_second_from_singleton_sim_repo = None

        self.controller_mode: str = "off"
        # Inputs
        self.water_mass_flow_rate_heat_generator_input_channel: ComponentInput = self.add_input(
            self.component_name,
            self.WaterMassFlowRateFromHeatGenerator,
            lt.LoadTypes.WARM_WATER,
            lt.Units.KG_PER_SEC,
            False,
        )
        # Outputs
        self.state_channel: ComponentOutput = self.add_output(
            self.component_name,
            self.State,
            lt.LoadTypes.ANY,
            lt.Units.ANY,
            output_description=f"here a description for {self.State} will follow.",
        )

    def build(self) -> None:
        """Build function.

        The function sets important constants and parameters for the calculations.
        """
        pass

    def i_prepare_simulation(self) -> None:
        """Prepare the simulation."""
        pass

    def i_save_state(self) -> None:
        """Save the current state."""
        pass

    def i_restore_state(self) -> None:
        """Restore the previous state."""
        pass

    def i_doublecheck(self, timestep: int, stsv: SingleTimeStepValues) -> None:
        """Doublecheck."""
        pass

    def write_to_report(self) -> None:
        """Write important variables to report."""
        pass

    def i_simulate(self, timestep: int, stsv: SingleTimeStepValues, force_convergence: bool) -> None:
        """Simulate the heat pump comtroller."""
        # prepare; skip if force convergence and get shorter names
        if force_convergence: return
        flow_val_from_sim_repo = (
            self.water_mass_flow_rate_from_heat_generator_in_kg_per_second_from_singleton_sim_repo)
        flow_val_channel = self.water_mass_flow_rate_heat_generator_input_channel
        # Retrieve inputs
        if flow_val_from_sim_repo is not None:
            flow_from_hg = flow_val_from_sim_repo
        else:
            flow_from_hg = stsv.get_input_value(flow_val_channel)
        # calc new state
        self.controller_mode = self.conditions_on_off(flow_from_hg)
        if self.controller_mode == "on": state = 1
        elif self.controller_mode == "off": state = 0
        else: raise ValueError("Controller State unknown.")
        # set output
        stsv.set_output_value(self.state_channel, state)

    def conditions_on_off(self, flow_from_hg_in_kg_per_s: float) -> str:
        """Set conditions for the simple hot water storage controller mode.
        Returns new state "on" or "off" depending on whether water from
        heat generator is flowing."""
        if self.controller_mode == "on":
            if flow_from_hg_in_kg_per_s == 0: 
                return "off" # turn off when no water flow
        elif self.controller_mode == "off":
            if flow_from_hg_in_kg_per_s != 0: 
                return "on" # turn on when water flow
        else:
            raise ValueError("unknown controller mode")
        return self.controller_mode


class SimpleDHWStorage(SimpleWaterStorage):
    """SimpleHotWaterStorage class."""

    # Input
    # A hot water storage can be used also with more than one heat generator. In this case you need to add a new input and output.
    WaterTemperatureFromHeatGenerator = "WaterTemperatureFromHeatGenerator"
    WaterMassFlowRateFromHeatGenerator = "WaterMassFlowRateFromHeatGenerator"
    WaterTemperatureFromSecondaryHeatGenerator = "WaterTemperatureFromSecondaryHeatGenerator"
    WaterMassFlowRateFromSecondaryHeatGenerator = "WaterMassFlowRateFromSecondaryHeatGenerator"
    WaterConsumption = "WaterConsumption"

    # Output
    WaterTemperatureToHeatGenerator = "WaterTemperatureToHeatGenerator"
    WaterTemperatureFromHeatGeneratorOutput = "WaterTemperatureFromHeatGenerator"
    WaterTemperatureFromSecondaryHeatGeneratorOutput = "WaterTemperatureFromSecondaryHeatGenerator"
    WaterMeanTemperatureInStorage = "WaterMeanTemperatureInStorage"
    StandbyTemperatureLoss = "StandbyTemperatureLoss"
    ThermalEnergyInStorage = "ThermalEnergyInStorage"
    ThermalEnergyFromHeatGenerator = "ThermalEnergyFromHeatGenerator"
    ThermalEnergyFromSecondaryHeatGenerator = "ThermalEnergyFromSecondaryHeatGenerator"
    ThermalEnergyConsumptionDHW = "ThermalEnergyConsumptionDHW"
    ThermalEnergyIncreaseInStorage = "ThermalEnergyIncreaseInStorage"
    ThermalPowerConsumptionDHW = "ThermalPowerConsumptionDHW"
    ThermalPowerFromHeatGenerator = "ThermalPowerFromHeatGenerator"
    ThermalPowerFromSecondaryHeatGenerator = "ThermalPowerFromSecondaryHeatGenerator"
    StandbyHeatLoss = "StandbyHeatLoss"
    WaterMassFlowRateOfDHW = "WaterMassFlowRateOfDHW"

    @utils.measure_execution_time
    def __init__(
        self,
        my_simulation_parameters: SimulationParameters,
        config: SimpleDHWStorageConfig,
        my_display_config: DisplayConfig = DisplayConfig(),
    ) -> None:
        """Construct all the neccessary attributes."""
        self.my_simulation_parameters = my_simulation_parameters
        self.config = config
        component_name = self.get_component_name()
        super().__init__(
            name=component_name,
            my_simulation_parameters=my_simulation_parameters,
            my_config=config,
            my_display_config=my_display_config,
        )
        # =================================================================================================================================
        # Initialization of variables
        self.seconds_per_timestep = my_simulation_parameters.seconds_per_timestep
        self.waterstorageconfig = config

        self.mean_water_temperature_in_water_storage_in_celsius: float = 60

        self.build()

        self.state: SimpleWaterStorageState = SimpleWaterStorageState(
            mean_water_temperature_in_celsius=self.mean_water_temperature_in_water_storage_in_celsius,
            temperature_loss_in_celsius_per_timestep=0,
            heat_loss_in_watt=0,
        )
        self.previous_state = self.state.self_copy()

        # =================================================================================================================================
        # Input channels

        self.water_consumption_channel: ComponentInput = self.add_input(
            self.component_name,
            self.WaterConsumption,
            lt.LoadTypes.WARM_WATER,
            lt.Units.LITER,
            True,
        )
        self.water_temperature_heat_generator_input_channel: ComponentInput = self.add_input(
            self.component_name,
            self.WaterTemperatureFromHeatGenerator,
            lt.LoadTypes.TEMPERATURE,
            lt.Units.CELSIUS,
            True,
        )
        self.water_mass_flow_rate_heat_generator_input_channel: ComponentInput = self.add_input(
            self.component_name,
            self.WaterMassFlowRateFromHeatGenerator,
            lt.LoadTypes.WARM_WATER,
            lt.Units.KG_PER_SEC,
            True,
        )

        self.water_temperature_secondary_heat_generator_input_channel: ComponentInput = self.add_input(
            self.component_name,
            self.WaterTemperatureFromSecondaryHeatGenerator,
            lt.LoadTypes.TEMPERATURE,
            lt.Units.CELSIUS,
            False,
        )
        self.water_mass_flow_rate_secondary_heat_generator_input_channel: ComponentInput = self.add_input(
            self.component_name,
            self.WaterMassFlowRateFromSecondaryHeatGenerator,
            lt.LoadTypes.WARM_WATER,
            lt.Units.KG_PER_SEC,
            False,
        )

        # Output channels

        self.water_temperature_to_heat_generator_channel: ComponentOutput = self.add_output(
            self.component_name,
            self.WaterTemperatureToHeatGenerator,
            lt.LoadTypes.WATER,
            lt.Units.CELSIUS,
            output_description=f"here a description for {self.WaterTemperatureToHeatGenerator} will follow.",
        )

        self.water_temperature_from_heat_generator_channel: ComponentOutput = self.add_output(
            self.component_name,
            self.WaterTemperatureFromHeatGeneratorOutput,
            lt.LoadTypes.WATER,
            lt.Units.CELSIUS,
            output_description=f"here a description for {self.WaterTemperatureFromHeatGeneratorOutput} will follow.",
        )

        self.water_temperature_from_secondary_heat_generator_channel: ComponentOutput = self.add_output(
            self.component_name,
            self.WaterTemperatureFromSecondaryHeatGeneratorOutput,
            lt.LoadTypes.WATER,
            lt.Units.CELSIUS,
            output_description="Water temperature [°C] from secondary DHW heat generator",
        )

        self.water_temperature_mean_channel: ComponentOutput = self.add_output(
            self.component_name,
            self.WaterMeanTemperatureInStorage,
            lt.LoadTypes.WATER,
            lt.Units.CELSIUS,
            output_description=f"here a description for {self.WaterMeanTemperatureInStorage} will follow.",
        )

        self.temperature_loss_channel: ComponentOutput = self.add_output(
            self.component_name,
            self.StandbyTemperatureLoss,
            lt.LoadTypes.WATER,
            lt.Units.CELSIUS,
            output_description=f"here a description for {self.StandbyTemperatureLoss} will follow.",
        )

        self.thermal_energy_in_storage_channel: ComponentOutput = self.add_output(
            self.component_name,
            self.ThermalEnergyInStorage,
            lt.LoadTypes.HEATING,
            lt.Units.WATT_HOUR,
            output_description=f"here a description for {self.ThermalEnergyInStorage} will follow.",
        )
        self.thermal_energy_from_heat_generator_channel: ComponentOutput = self.add_output(
            self.component_name,
            self.ThermalEnergyFromHeatGenerator,
            lt.LoadTypes.HEATING,
            lt.Units.WATT_HOUR,
            output_description=f"here a description for {self.ThermalEnergyFromHeatGenerator} will follow.",
            postprocessing_flag=[lt.OutputPostprocessingRules.DISPLAY_IN_WEBTOOL],
        )
        self.thermal_energy_from_secondary_heat_generator_channel: ComponentOutput = self.add_output(
            self.component_name,
            self.ThermalEnergyFromSecondaryHeatGenerator,
            lt.LoadTypes.HEATING,
            lt.Units.WATT_HOUR,
            output_description=f"here a description for {self.ThermalEnergyFromHeatGenerator} will follow.",
            postprocessing_flag=[lt.OutputPostprocessingRules.DISPLAY_IN_WEBTOOL],
        )
        self.thermal_energy_dhw_channel: ComponentOutput = self.add_output(
            self.component_name,
            self.ThermalEnergyConsumptionDHW,
            lt.LoadTypes.HEATING,
            lt.Units.WATT_HOUR,
            output_description=f"here a description for {self.ThermalEnergyConsumptionDHW} will follow.",
            postprocessing_flag=[lt.OutputPostprocessingRules.DISPLAY_IN_WEBTOOL],
        )

        self.thermal_energy_increase_in_storage_channel: ComponentOutput = self.add_output(
            self.component_name,
            self.ThermalEnergyIncreaseInStorage,
            lt.LoadTypes.HEATING,
            lt.Units.WATT_HOUR,
            output_description=f"here a description for {self.ThermalEnergyIncreaseInStorage} will follow.",
        )

        self.stand_by_heat_loss_channel: ComponentOutput = self.add_output(
            self.component_name,
            self.StandbyHeatLoss,
            lt.LoadTypes.HEATING,
            lt.Units.WATT,
            output_description=f"here a description for {self.StandbyHeatLoss} will follow.",
        )

        self.thermal_power_dhw_channel: ComponentOutput = self.add_output(
            self.component_name,
            self.ThermalPowerConsumptionDHW,
            lt.LoadTypes.HEATING,
            lt.Units.WATT,
            output_description=f"here a description for {self.ThermalPowerConsumptionDHW} will follow.",
        )

        self.thermal_power_from_heat_generator_channel: ComponentOutput = self.add_output(
            self.component_name,
            self.ThermalPowerFromHeatGenerator,
            lt.LoadTypes.HEATING,
            lt.Units.WATT,
            output_description=f"here a description for {self.ThermalPowerFromHeatGenerator} will follow.",
        )
        self.thermal_power_from_secondary_heat_generator_channel: ComponentOutput = self.add_output(
            self.component_name,
            self.ThermalPowerFromSecondaryHeatGenerator,
            lt.LoadTypes.HEATING,
            lt.Units.WATT,
            output_description=f"here a description for {self.ThermalPowerFromHeatGenerator} will follow.",
        )
        self.water_mass_flow_rate_dhw_output_channel: ComponentOutput = self.add_output(
            self.component_name,
            self.WaterMassFlowRateOfDHW,
            lt.LoadTypes.WARM_WATER,
            lt.Units.KG_PER_SEC,
            output_description=f"here a description for {self.WaterMassFlowRateOfDHW} will follow.",
        )

        self.add_default_connections(self.get_default_connections_from_more_advanced_heat_pump())
        self.add_default_connections(self.get_default_connections_from_generic_dhw_boiler())
        self.add_default_connections(self.get_default_connections_from_district_heating())
        self.add_default_connections(self.get_default_connections_from_utsp())
        self.add_default_connections(self.get_default_connections_from_solar_thermal_system())
        self.add_default_connections(self.get_default_connections_from_electric_heating())

    def get_default_connections_from_more_advanced_heat_pump(
        self,
    ) -> List[cp.ComponentConnection]:
        """Get advanced het pump default connections."""

        # use importlib for importing the other component in order to avoid circular-import errors
        component_module_name = "hisim.components.more_advanced_heat_pump_hplib"
        component_module = importlib.import_module(name=component_module_name)
        component_class = getattr(component_module, "MoreAdvancedHeatPumpHPLib")
        connections = []
        hp_classname = component_class.get_classname()
        connections.append(
            cp.ComponentConnection(
                SimpleDHWStorage.WaterTemperatureFromHeatGenerator,
                hp_classname,
                component_class.TemperatureOutputDHW,
            )
        )
        connections.append(
            cp.ComponentConnection(
                SimpleDHWStorage.WaterMassFlowRateFromHeatGenerator,
                hp_classname,
                component_class.MassFlowOutputDHW,
            )
        )
        return connections

    def get_default_connections_from_utsp(
        self,
    ) -> List[cp.ComponentConnection]:
        """Get advanced het pump default connections."""

        # use importlib for importing the other component in order to avoid circular-import errors
        component_module_name = "hisim.components.loadprofilegenerator_utsp_connector"
        component_module = importlib.import_module(name=component_module_name)
        component_class = getattr(component_module, "UtspLpgConnector")
        connections = []
        utsp_classname = component_class.get_classname()
        connections.append(
            cp.ComponentConnection(
                SimpleDHWStorage.WaterConsumption,
                utsp_classname,
                component_class.WaterConsumption,
            )
        )
        return connections

    def get_default_connections_from_generic_dhw_boiler(
        self,
    ) -> List[cp.ComponentConnection]:
        """Get generic dhw boiler default connections."""

        # use importlib for importing the other component in order to avoid circular-import errors
        component_module_name = "hisim.components.generic_boiler"
        component_module = importlib.import_module(name=component_module_name)
        component_class = getattr(component_module, "GenericBoiler")
        connections = []
        dhw_boiler_classname = component_class.get_classname()
        connections.append(
            cp.ComponentConnection(
                SimpleDHWStorage.WaterTemperatureFromHeatGenerator,
                dhw_boiler_classname,
                component_class.WaterOutputTemperatureDhw,
            )
        )
        connections.append(
            cp.ComponentConnection(
                SimpleDHWStorage.WaterMassFlowRateFromHeatGenerator,
                dhw_boiler_classname,
                component_class.WaterOutputMassFlowDhw,
            )
        )
        return connections

    def get_default_connections_from_solar_thermal_system(
        self,
    ) -> List[cp.ComponentConnection]:
        """Get solar thermal system default connections."""

        # use importlib for importing the other component in order to avoid circular-import errors
        component_module_name = "hisim.components.solar_thermal_system"
        component_module = importlib.import_module(name=component_module_name)
        component_class = getattr(component_module, "SolarThermalSystem")
        connections = []
        solar_thermal_system_classname = component_class.get_classname()
        connections.append(
            cp.ComponentConnection(
                SimpleDHWStorage.WaterTemperatureFromHeatGenerator,
                solar_thermal_system_classname,
                component_class.WaterTemperatureOutput,
            )
        )
        connections.append(
            cp.ComponentConnection(
                SimpleDHWStorage.WaterMassFlowRateFromHeatGenerator,
                solar_thermal_system_classname,
                component_class.WaterMassFlowOutput,
            )
        )
        return connections

    def get_default_connections_from_district_heating(
        self,
    ) -> List[cp.ComponentConnection]:
        """Get dhw district heating default connections."""

        # use importlib for importing the other component in order to avoid circular-import errors
        component_module_name = "hisim.components.generic_district_heating"
        component_module = importlib.import_module(name=component_module_name)
        component_class = getattr(component_module, "DistrictHeating")
        connections = []
        component_classname = component_class.get_classname()
        connections.append(
            cp.ComponentConnection(
                SimpleDHWStorage.WaterTemperatureFromHeatGenerator,
                component_classname,
                component_class.WaterOutputDhwTemperature,
            )
        )
        connections.append(
            cp.ComponentConnection(
                SimpleDHWStorage.WaterMassFlowRateFromHeatGenerator,
                component_classname,
                component_class.WaterOutputDhwMassFlowRate,
            )
        )
        return connections

    def get_default_connections_from_electric_heating(
        self,
    ) -> List[cp.ComponentConnection]:
        """Get dhw electric heating default connections."""

        # use importlib for importing the other component in order to avoid circular-import errors
        component_module_name = "hisim.components.generic_electric_heating"
        component_module = importlib.import_module(name=component_module_name)
        component_class = getattr(component_module, "ElectricHeating")
        connections = []
        component_classname = component_class.get_classname()
        connections.append(
            cp.ComponentConnection(
                SimpleDHWStorage.WaterTemperatureFromHeatGenerator,
                component_classname,
                component_class.WaterOutputDhwTemperature,
            )
        )
        connections.append(
            cp.ComponentConnection(
                SimpleDHWStorage.WaterMassFlowRateFromHeatGenerator,
                component_classname,
                component_class.WaterOutputDhwMassFlowRate,
            )
        )
        return connections

    def build(
        self,
    ) -> None:
        """Build function.

        The function sets important constants an parameters for the calculations.
        """
        self.drain_water_temperature = configuration.HouseholdWarmWaterDemandConfig.freshwater_temperature

        self.warm_water_temperature = (
            configuration.HouseholdWarmWaterDemandConfig.ww_temperature_demand
            - configuration.HouseholdWarmWaterDemandConfig.temperature_difference_hot
        )

        self.specific_heat_capacity_of_water_in_joule_per_kilogram_per_celsius = (
            PhysicsConfig.get_properties_for_energy_carrier(
                energy_carrier=lt.LoadTypes.WATER
            ).specific_heat_capacity_in_joule_per_kg_per_kelvin
        )
        self.specific_heat_capacity_of_water_in_watthour_per_kilogram_per_celsius = (
            PhysicsConfig.get_properties_for_energy_carrier(
                energy_carrier=lt.LoadTypes.WATER
            ).specific_heat_capacity_in_watthour_per_kg_per_kelvin
        )
        # https://www.internetchemie.info/chemie-lexikon/daten/w/wasser-dichtetabelle.php
        self.density_water_at_40_degree_celsius_in_kg_per_liter = 0.992

        # physical parameters of storage
        self.water_mass_in_storage_in_kg = (
            self.density_water_at_40_degree_celsius_in_kg_per_liter
            * self.waterstorageconfig.volume_heating_water_storage_in_liter
        )
        self.heat_transfer_coefficient_in_watt_per_m2_per_kelvin = (
            self.waterstorageconfig.heat_transfer_coefficient_in_watt_per_m2_per_kelvin
        )
        self.storage_surface_in_m2 = self.calculate_surface_area_of_storage(
            storage_volume_in_liter=self.waterstorageconfig.volume_heating_water_storage_in_liter,
        )

        self.ambient_temperature_in_celsius = 20.0

    def i_prepare_simulation(self) -> None:
        """Prepare the simulation."""
        pass

    def write_to_report(self) -> List[str]:
        """Write a report."""
        return self.waterstorageconfig.get_string_dict()

    def i_save_state(self) -> None:
        """Save the current state."""
        self.previous_state = self.state.self_copy()

    def i_restore_state(self) -> None:
        """Restore the previous state."""
        self.state = self.previous_state.self_copy()

    def i_doublecheck(self, timestep: int, stsv: SingleTimeStepValues) -> None:
        """Doublecheck."""
        pass

    def i_simulate(self, timestep: int, stsv: SingleTimeStepValues, force_convergence: bool) -> None:
        """Simulate the heating water storage."""

        # Get inputs --------------------------------------------------------------------------------------------------------

        water_temperature_input_of_dhw_in_celsius = self.drain_water_temperature
        water_temperature_output_of_dhw_in_celsius = self.warm_water_temperature
        water_mass_flow_rate_of_dhw_in_kg_per_s = (
            stsv.get_input_value(self.water_consumption_channel)
            * self.density_water_at_40_degree_celsius_in_kg_per_liter
            / self.seconds_per_timestep
        )

        water_temperature_from_heat_generator_in_celsius = stsv.get_input_value(
            self.water_temperature_heat_generator_input_channel
        )
        water_mass_flow_rate_from_hg_in_kg_per_s = stsv.get_input_value(
            self.water_mass_flow_rate_heat_generator_input_channel
        )

        # Optional secondary heat generator
        water_temperature_from_secondary_heat_generator_in_celsius = stsv.get_input_value(
            self.water_temperature_secondary_heat_generator_input_channel
        )
        water_mass_flow_rate_from_hg2_in_kg_per_s = stsv.get_input_value(
            self.water_mass_flow_rate_secondary_heat_generator_input_channel
        )

        # Water Temperature Limit Check  --------------------------------------------------------------------------------------------------------

#        if (
#            self.mean_water_temperature_in_water_storage_in_celsius > 90
#            or self.mean_water_temperature_in_water_storage_in_celsius < 0
#        ):
#            raise ValueError(
#                f"The water temperature in the DHW water storage is with {self.mean_water_temperature_in_water_storage_in_celsius}°C way too high or too low."
#            )

        # if (water_mass_flow_rate_of_dhw_in_kg_per_second > 0) and (self.mean_water_temperature_in_water_storage_in_celsius < self.warm_water_temperature):
        #     # if there is water consumption, the temperature must be high enough
        #     log.warning(f"The DHW water temperature is only {self.mean_water_temperature_in_water_storage_in_celsius}°C.")

        # Calculations ------------------------------------------------------------------------------------------------------

        # calc water masses
        water_mass_from_hg_in_kg = water_mass_flow_rate_from_hg_in_kg_per_s * self.seconds_per_timestep
        water_mass_of_dhw_in_kg = water_mass_flow_rate_of_dhw_in_kg_per_s * self.seconds_per_timestep
        water_mass_from_hg2_in_kg = water_mass_flow_rate_from_hg2_in_kg_per_s * self.seconds_per_timestep

        # calc water temperatures
        # ------------------------------

        # mean temperature in storage when all water flows are mixed with previous mean water storage temp
        self.mean_water_temperature_in_water_storage_in_celsius = self.calculate_mean_water_temperature_in_water_storage(
            water_temperature_input_of_secondary_side_in_celsius=water_temperature_input_of_dhw_in_celsius,
            water_mass_in_storage_in_kg=self.water_mass_in_storage_in_kg,
            water_temperature_from_heat_generator_in_celsius=water_temperature_from_heat_generator_in_celsius,
            mass_of_input_water_flows_from_heat_generator_in_kg=water_mass_from_hg_in_kg,
            water_temperature_from_secondary_heat_generator_in_celsius=water_temperature_from_secondary_heat_generator_in_celsius,
            mass_of_input_water_flows_from_secondary_heat_generator_in_kg=water_mass_from_hg2_in_kg,
            mass_of_input_water_flows_of_secondary_side_in_kg=water_mass_of_dhw_in_kg,
            previous_mean_water_temperature_in_water_storage_in_celsius=self.state.mean_water_temperature_in_celsius,
        )

        # calc thermal energies
        # ------------------------------

        previous_thermal_energy_in_storage_in_watt_hour = self.calc_thermal_energy(
            t_water=self.state.mean_water_temperature_in_celsius,
            mass=self.water_mass_in_storage_in_kg,
        )
        current_thermal_energy_in_storage_in_watt_hour = self.calc_thermal_energy(
            t_water=self.mean_water_temperature_in_water_storage_in_celsius,
            mass=self.water_mass_in_storage_in_kg,
        )
        thermal_energy_increase_current_vs_previous_mean_temperature_in_watt_hour = (
            current_thermal_energy_in_storage_in_watt_hour
            - previous_thermal_energy_in_storage_in_watt_hour
        )

        thermal_energy_input_from_heat_generator_in_watt_hour = self.calc_thermal_energy(
            mass=water_mass_from_hg_in_kg,
            t_water=water_temperature_from_heat_generator_in_celsius
                - self.state.mean_water_temperature_in_celsius,
        )

        # Secondary heat generator
        thermal_energy_input_from_secondary_heat_generator_in_watt_hour = self.calc_thermal_energy(
            mass=water_mass_from_hg2_in_kg,
            t_water=water_temperature_from_secondary_heat_generator_in_celsius
                - self.state.mean_water_temperature_in_celsius,
        )

        thermal_energy_consumption_of_dhw_in_watt_hour = self.calc_thermal_energy(
            mass=water_mass_of_dhw_in_kg,
            t_water=water_temperature_input_of_dhw_in_celsius
                - water_temperature_output_of_dhw_in_celsius,
        )

        # calc thermal power
        # ------------------------------
        thermal_power_from_heat_generator_in_watt = self.calc_thermal_power(
            mass_flow=water_mass_flow_rate_from_hg_in_kg_per_s,
            t_diff=water_temperature_from_heat_generator_in_celsius
                - self.state.mean_water_temperature_in_celsius
        )
        thermal_power_from_secondary_heat_generator_in_watt = self.calc_thermal_power(
            mass_flow=water_mass_flow_rate_from_hg2_in_kg_per_s,
            t_diff=water_temperature_from_secondary_heat_generator_in_celsius
                - self.state.mean_water_temperature_in_celsius
        )
        thermal_power_consumption_of_dhw_in_watt = self.calc_thermal_power(
            mass_flow=water_mass_flow_rate_of_dhw_in_kg_per_s,
            t_diff=water_temperature_output_of_dhw_in_celsius
                - water_temperature_input_of_dhw_in_celsius
        )

        water_temperature_to_heat_generator_in_celsius = self.state.mean_water_temperature_in_celsius

        stsv.set_output_value(
            self.water_temperature_to_heat_generator_channel,
            water_temperature_to_heat_generator_in_celsius,
        )

        stsv.set_output_value(
            self.water_temperature_from_heat_generator_channel,
            water_temperature_from_heat_generator_in_celsius,
        )

        stsv.set_output_value(
            self.water_temperature_from_secondary_heat_generator_channel,
            water_temperature_from_secondary_heat_generator_in_celsius,
        )

        stsv.set_output_value(
            self.water_temperature_mean_channel,
            self.state.mean_water_temperature_in_celsius,
        )

        stsv.set_output_value(
            self.temperature_loss_channel,
            self.state.temperature_loss_in_celsius_per_timestep,
        )

        stsv.set_output_value(
            self.thermal_energy_in_storage_channel,
            current_thermal_energy_in_storage_in_watt_hour,
        )

        stsv.set_output_value(
            self.thermal_energy_from_heat_generator_channel,
            thermal_energy_input_from_heat_generator_in_watt_hour,
        )

        stsv.set_output_value(
            self.thermal_energy_from_secondary_heat_generator_channel,
            thermal_energy_input_from_secondary_heat_generator_in_watt_hour,
        )

        stsv.set_output_value(
            self.thermal_energy_dhw_channel,
            thermal_energy_consumption_of_dhw_in_watt_hour,
        )

        stsv.set_output_value(
            self.thermal_energy_increase_in_storage_channel,
            thermal_energy_increase_current_vs_previous_mean_temperature_in_watt_hour,
        )

        stsv.set_output_value(
            self.stand_by_heat_loss_channel,
            self.state.heat_loss_in_watt,
        )

        stsv.set_output_value(
            self.thermal_power_dhw_channel,
            thermal_power_consumption_of_dhw_in_watt,
        )

        stsv.set_output_value(
            self.thermal_power_from_heat_generator_channel,
            thermal_power_from_heat_generator_in_watt,
        )

        stsv.set_output_value(
            self.thermal_power_from_secondary_heat_generator_channel,
            thermal_power_from_secondary_heat_generator_in_watt,
        )

        stsv.set_output_value(
            self.water_mass_flow_rate_dhw_output_channel,
            water_mass_flow_rate_of_dhw_in_kg_per_s,
        )
        # Set state -------------------------------------------------------------------------------------------------------
        # calc heat loss in W and the temperature loss

        (
            self.state.heat_loss_in_watt,
            t_loss_per_s,
        ) = self.calculate_heat_loss_and_temperature_loss(
            storage_surface_in_m2=self.storage_surface_in_m2,
            mean_water_temperature_in_water_storage_in_celsius=self.mean_water_temperature_in_water_storage_in_celsius,
            heat_transfer_coefficient_in_watt_per_m2_per_kelvin=self.heat_transfer_coefficient_in_watt_per_m2_per_kelvin,
            mass_in_storage_in_kg=self.water_mass_in_storage_in_kg,
            ambient_temperature_in_celsius=self.ambient_temperature_in_celsius,
        )

        self.state.temperature_loss_in_celsius_per_timestep = t_loss_per_s * self.seconds_per_timestep

        self.state.mean_water_temperature_in_celsius = (
            self.mean_water_temperature_in_water_storage_in_celsius
            - self.state.temperature_loss_in_celsius_per_timestep
        )

    @staticmethod
    def get_cost_capex(
        config: SimpleDHWStorageConfig, simulation_parameters: SimulationParameters
    ) -> CapexCostDataClass:
        """Returns investment cost, CO2 emissions and lifetime."""
        kpi_tag = KpiTagEnumClass.STORAGE_DOMESTIC_HOT_WATER
        component_type = lt.ComponentType.THERMAL_ENERGY_STORAGE
        unit = lt.Units.LITER
        size_of_energy_system = config.volume_heating_water_storage_in_liter

        capex_cost_data_class = CapexComputationHelperFunctions.compute_capex_costs_and_emissions(
        simulation_parameters=simulation_parameters,
        component_type=component_type,
        unit=unit,
        size_of_energy_system=size_of_energy_system,
        config=config,
        kpi_tag=kpi_tag
        )

        return capex_cost_data_class

    def get_cost_opex(
        self,
        all_outputs: List,
        postprocessing_results: pd.DataFrame,
    ) -> OpexCostDataClass:
        # pylint: disable=unused-argument
        """Calculate OPEX costs, consisting of maintenance costs for hot water storage."""
        opex_cost_data_class = OpexCostDataClass(
            opex_energy_cost_in_euro=0,
            opex_maintenance_cost_in_euro=self.calc_maintenance_cost(),
            co2_footprint_in_kg=0,
            total_consumption_in_kwh=0,
            loadtype=lt.LoadTypes.ANY,
            kpi_tag=KpiTagEnumClass.STORAGE_HOT_WATER_SPACE_HEATING,
        )

        return opex_cost_data_class

    def get_component_kpi_entries(
        self,
        all_outputs: List,
        postprocessing_results: pd.DataFrame,
    ) -> List[KpiEntry]:
        """Calculates KPIs for the respective component and return all KPI entries as list."""
        list_of_kpi_entries: List[KpiEntry] = []
        for index, output in enumerate(all_outputs):
            if output.component_name == self.component_name:
                if output.field_name == self.StandbyHeatLoss and output.unit == lt.Units.WATT:
                    # calc heat loss
                    heat_loss_in_watt = postprocessing_results.iloc[:, index].loc[
                        postprocessing_results.iloc[:, index] > 0.0
                    ]
                    # get energy from power
                    heat_loss_in_kilowatt_hour = round(
                        KpiHelperClass.compute_total_energy_from_power_timeseries(
                            power_timeseries_in_watt=heat_loss_in_watt,
                            timeresolution=self.my_simulation_parameters.seconds_per_timestep,
                        ),
                        1,
                    )
                    heat_loss_entry = KpiEntry(
                        name="Standby heat loss of DHW storage",
                        unit="kWh",
                        value=heat_loss_in_kilowatt_hour,
                        tag=KpiTagEnumClass.STORAGE_DOMESTIC_HOT_WATER,
                        description=self.component_name,
                    )
                    list_of_kpi_entries.append(heat_loss_entry)
        return list_of_kpi_entries
