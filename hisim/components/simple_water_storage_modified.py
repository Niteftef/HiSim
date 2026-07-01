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
from hisim import log
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

    building_name: str
    name: str
    volume_heating_water_storage_in_liter: float
    heat_transfer_coefficient_in_watt_per_m2_per_kelvin: float
    heat_exchanger_is_present: bool
    """Keep this on true, otherwise a weird mixing factor is applied to the outgoing water
    temperatures. Use the simulation_model attribute instead."""
    simulation_model: str
    """Governs which mathematical functions are used to simulate this component.
    Options include: 
    - "standard": The current HiSim implementation where the outgoing temperatures are calculated
        like explicit Euler and the new temperature in storage is calculated like implicit Euler,
        leading to discrepancies in the energy balance.
    - "hisim": Alias for "standard", does the same thing as "standard".
    - "explicit": Uses explicit Euler. Faster, but not numerically stable. If your simulation
        fails due to this component, try using smaller time steps. 
    - "implicit": Uses implicit Euler. Numerically stable but potentially requires many
        iterations to converge or may even completely fail to converge.
    - "analytical": NOT IMPLEMENTED. Should be possible for this. Has the same issues as implicit
        Euler, namely it introduces circular dependencies, but would be more accurate at least.
    - "explicit_with_bypass": Uses explicit Euler as long as the stability condition is met, but
        bypasses the tank entirely if the stability condition is exceeded. This takes the tank
        out of the system in those cases, which obviously is kinda wrong, but probably less wrong
        than the alternatives. Also introduces a direct circular dependency between the other
        components, but since they don't have thermal capacities (usually), that can probably
        be resolved much much faster.
    - "explicit_with_culling": Uses explicit Euler as long as the stability condition is met, but
        removes any overshoots due to numerical instability if the condition is not met. This
        culling introduces an error that is logged and that the model will try to feed back into
        the actual tank temperature in the next time step, but only to a degree that doesn't lead
        to numerical instability again. This is also kinda wrong of course, but hopefully less so
        that the standard method. It should work as long as the tank has a few time steps where it
        can deal with the error before the next big chunk of error comes in, which I think should
        be the case, most of the time, at least.
    """
    position_hot_water_storage_in_system: Union[PositionHotWaterStorageInSystemSetup, int]
    # it should be checked how much energy the storage lost during the simulated period (see guidelines below, p.2, accepted loss in kWh/days)
    # https://www.bdh-industrie.de/fileadmin/user_upload/ISH2019/Infoblaetter/Infoblatt_Nr_74_Energetische_Bewertung_Warmwasserspeicher.pdf
    device_co2_footprint_in_kg: Optional[float]
    """CO2 footprint of investment in kg"""
    investment_costs_in_euro: Optional[float]
    """cost for investment in Euro"""
    lifetime_in_years: Optional[float]
    """lifetime in years"""
    maintenance_costs_in_euro_per_year: Optional[float]
    """maintenance cost in euro per year"""
    subsidy_as_percentage_of_investment_costs: Optional[float]
    """subsidies as percentage of investment costs"""
    
    @classmethod
    def get_main_classname(cls):
        """Return the full class name of the base class."""
        return SimpleHotWaterStorage.get_full_classname()

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
            simulation_model="standard",
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
        sizing_option: HotWaterStorageSizingEnum = HotWaterStorageSizingEnum.SIZE_ACCORDING_TO_GENERAL_HEATING_SYSTEM,
    ) -> "SimpleHotWaterStorageConfig":
        """Gets a default storage with scaling according to heating load of the building_name.

        The information for scaling the buffer storage is taken from the heating system guidelines from Buderus:
        https://www.baunetzwissen.de/heizung/fachwissen/speicher/dimensionierung-von-pufferspeichern-161296
        Or from here:
        https://www.flexiheatuk.com/buffer-vessel-sizing-for-hydronic-heating-systems/#:~:text=20%2D25%20litres%20per%20kW,kW%20for%20heat%20pump%20systems

        """
        # if the used heating system is a heat pump use formular
        if sizing_option == HotWaterStorageSizingEnum.SIZE_ACCORDING_TO_HEAT_PUMP:

            volume_heating_water_storage_in_liter = max_thermal_power_in_watt_of_heating_system / 1e3 * 50
            # https://www.flexiheatuk.com/buffer-vessel-sizing-for-hydronic-heating-systems/#:~:text=20%2D25%20litres%20per%20kW,kW%20for%20heat%20pump%20systems

        # otherwise use approximation: 60l per kw thermal power
        elif sizing_option == HotWaterStorageSizingEnum.SIZE_ACCORDING_TO_GENERAL_HEATING_SYSTEM:
            volume_heating_water_storage_in_liter = max_thermal_power_in_watt_of_heating_system / 1e3 * 20

        # large storage for pellet heating to avoid frequent on-off
        elif sizing_option == HotWaterStorageSizingEnum.SIZE_ACCORDING_TO_PELLET_HEATING:
            volume_heating_water_storage_in_liter = max_thermal_power_in_watt_of_heating_system / 1e3 * 40

        # large storage even more important than for pellets, as on-off behavior should be avoided
        elif sizing_option == HotWaterStorageSizingEnum.SIZE_ACCORDING_TO_WOOD_CHIP_HEATING:
            volume_heating_water_storage_in_liter = max_thermal_power_in_watt_of_heating_system / 1e3 * 50

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
            simulation_model="standard",
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

    building_name: str
    name: str

    @classmethod
    def get_main_classname(cls):
        """Return the full class name of the base class."""
        return SimpleHotWaterStorageController.get_full_classname()

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

    building_name: str
    name: str
    volume_heating_water_storage_in_liter: float
    heat_transfer_coefficient_in_watt_per_m2_per_kelvin: float
    simulation_model: str
    """For an explanation, see the equally named attribute in SimpleHotWaterStorageConfig"""
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
    def get_main_classname(cls):
        """Return the full class name of the base class."""
        return SimpleDHWStorage.get_full_classname()

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
            simulation_model="standard",
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
            simulation_model="standard",
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
    temperature_mismatch: float = 0.0
    """For the culling model, this is the difference between the theoretical temperature of the water
    in the buffer tank and the actual (after culling). t_mismatch = t_theoretical - t_actual"""

    def self_copy(self):
        """Copy the Simple Hot Water Storage State."""
        return SimpleWaterStorageState(
            self.mean_water_temperature_in_celsius,
            self.temperature_loss_in_celsius_per_timestep,
            self.heat_loss_in_watt,
            self.temperature_mismatch
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
        explicit_or_implicit: str = "explicit",
    ) -> float:
        """Calculate the mean temperature of the water in the water tank."""
        # prepare
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
        # now t_intermediate is the weighted average of the previous temperature plus all inflows
        # depending in the method, the outflows now have to be subtracted or not
        if explicit_or_implicit == "explicit": # potentially numerically unstable
            t_prev = previous_mean_water_temperature_in_water_storage_in_celsius
            result = (t_intermediate * mass_sum - t_prev * mass_sum_inputs) / water_mass_in_storage_in_kg
        elif explicit_or_implicit == "implicit":  # in this method, the outflows are the same temperature,
            result = t_intermediate               # so it doesn't matter whether we actively subtract them
        else:
            raise KeyError(f"Unrecognized method, should be explizit or implicit but was: {explicit_or_implicit}")
        return result

    def calc_culling_model(self,
        some_kwargs: dict,
        previous_mismatch: float,
        t_in_storage: float,
        weighted_average_inflow_temperature: float
    ) -> tuple[float, float]:
        """t_mismatch is defined by: t_actual_in_storage + t_mismatch = t_theoretical_in_storage"""
        # calc theoretical temperature without culling and add back previous mismatch
        t_new_in_storage_theoretical = self.calculate_mean_water_temperature_in_water_storage(
            **some_kwargs, explicit_or_implicit="explicit")
        t_new_in_storage_theoretical = t_new_in_storage_theoretical + previous_mismatch
        # cull if necessary
        lower_limit, upper_limit = sorted([t_in_storage, weighted_average_inflow_temperature])
        if t_new_in_storage_theoretical < lower_limit:
            t_mismatch = t_new_in_storage_theoretical - lower_limit
            t_new_in_storage = t_new_in_storage_theoretical - t_mismatch
            assert t_new_in_storage == lower_limit, (f"t_new_in_storage ({t_new_in_storage}) "
                                                    f"should be equal to lower_limit ({lower_limit})")
        elif t_new_in_storage_theoretical > upper_limit:
            t_mismatch = t_new_in_storage_theoretical - upper_limit
            t_new_in_storage = t_new_in_storage_theoretical - t_mismatch
            assert t_new_in_storage == upper_limit, (f"t_new_in_storage ({t_new_in_storage}) "
                                                    f"should be equal to upper_limit ({upper_limit})")
        else:
            t_mismatch = 0
            t_new_in_storage = t_new_in_storage_theoretical
        # return new temperature and the mismatch
        return t_new_in_storage, t_mismatch

    def calculate_mixing_factor_for_water_temperature_outputs(self) -> Any:
        """Calculate mixing factor for water outputs."""
        # mixing factor depends on seconds per timestep
        # if one timestep = 1h (3600s) or more, the factor for the water storage portion is one
        factor_for_water_storage_portion = min(self.seconds_per_timestep / 3600, 1)
        factor_for_water_input_portion = 1 - factor_for_water_storage_portion
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
        """Calculate heat energy loss in W and temperature loss in K/s.

        Calculate the heat energy loss in W and the temperature loss in K/s of the water storage
        based on surface area, heat transfer coefficient, inner and outer temperature and water
        mass in storage.
        """
        heat_loss_in_watt = self.calculate_heat_loss_in_watt(
            mean_temperature_in_storage_in_celsius=mean_water_temperature_in_water_storage_in_celsius,
            storage_surface_in_m2=storage_surface_in_m2,
            heat_transfer_coefficient_in_watt_per_m2_per_kelvin=heat_transfer_coefficient_in_watt_per_m2_per_kelvin,
            ambient_temperature_in_celsius=ambient_temperature_in_celsius)
        # basis here: Q = m * cw * delta temperature, temperature loss is another term for delta temperature here
        c_p = PhysicsConfig.get_properties_for_energy_carrier(
                energy_carrier=lt.LoadTypes.WATER
            ).specific_heat_capacity_in_joule_per_kg_per_kelvin
        temperature_loss_of_water_in_kelvin_per_s = heat_loss_in_watt / (c_p * mass_in_storage_in_kg)
        # return result
        return heat_loss_in_watt, temperature_loss_of_water_in_kelvin_per_s

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
        # return result
        return float(storage_surface_in_m2)

    ##############################################################################################

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
    State = "State"
    WaterTemperatureFromHeatDistribution = "WaterTemperatureFromHeatDistribution"
    WaterTemperatureFromHeatGenerator = "WaterTemperatureFromHeatGenerator"
    WaterTemperatureFromSecondaryHeatGenerator = "WaterTemperatureFromSecondaryHeatGenerator"
    WaterMassFlowRateFromHeatDistributionSystem = "WaterMassFlowRateFromHeatDistributionSystem"
    WaterMassFlowRateFromHeatGenerator = "WaterMassFlowRateFromHeatGenerator"
    WaterMassFlowRateFromSecondaryHeatGenerator = "WaterMassFlowRateFromSecondaryHeatGenerator"

    # Output
    WaterTemperatureToHeatDistribution = "WaterTemperatureToHeatDistribution"
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

    TemperatureMismatch = "TemperatureMismatch"

    # ============================================================================================
    # ===== The __init__ function and its helpers ================================================
    # ============================================================================================

    @utils.measure_execution_time
    def __init__(
        self,
        my_simulation_parameters: SimulationParameters,
        config: SimpleHotWaterStorageConfig,
        my_display_config: DisplayConfig = DisplayConfig(),
    ) -> None:
        """Construct all the neccessary attributes."""
        # Basic attributes
        self.my_simulation_parameters = my_simulation_parameters
        self.config = config
        component_name = self.get_component_name()
        super().__init__(
            name=component_name,
            my_simulation_parameters=my_simulation_parameters,
            my_config=config,
            my_display_config=my_display_config,
        )
        # Initialization of variables
        self.seconds_per_timestep = my_simulation_parameters.seconds_per_timestep
        self.waterstorageconfig = config
        self.mean_water_temperature_in_water_storage_in_celsius: float = 35
        self.no_explicit_warning_yet = True
        self.position_hot_water_storage_in_system = self.waterstorageconfig.position_hot_water_storage_in_system
        # ! I'm like pretty sure this is deprecated. I already deleted it in i_simulate
        if SingletonSimRepository().exist_entry(key=SingletonDictKeyEnum.WATERMASSFLOWRATEOFHEATGENERATOR):
            self.water_mass_flow_rate_from_hg_in_kg_per_s_from_singleton_sim_repo = (
                SingletonSimRepository().get_entry(key=SingletonDictKeyEnum.WATERMASSFLOWRATEOFHEATGENERATOR)
            )
        else:
            self.water_mass_flow_rate_from_hg_in_kg_per_s_from_singleton_sim_repo = None
        # build function call
        self.build(heat_exchanger_is_present=self.waterstorageconfig.heat_exchanger_is_present)
        # set state
        self.state: SimpleWaterStorageState = SimpleWaterStorageState(
            mean_water_temperature_in_celsius=self.mean_water_temperature_in_water_storage_in_celsius,
            temperature_loss_in_celsius_per_timestep=0,
        )
        self.previous_state = self.state.self_copy()
        # add inputs, outputs and default connections
        self.add_all_inputs()
        self.add_all_outputs()
        self.add_default_connections(self.get_default_connections_from_heat_distribution_system())
        self.add_default_connections(self.get_default_connections_from_advanced_heat_pump())
        self.add_default_connections(self.get_default_connections_from_more_advanced_heat_pump())
        self.add_default_connections(self.get_default_connections_from_generic_boiler())

    def add_all_inputs(self):
        """Adds all the inputs of the building as attributes. To be used in __init__."""
        self.state_channel: cp.ComponentInput = self.add_input(
            self.component_name,
            self.State,
            lt.LoadTypes.ANY,
            lt.Units.ANY,
            False
        )
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
        # only for parallel buffers
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

    def add_all_outputs(self):
        """Adds all the outputs as attributes. To be used in __init__."""
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
        self.temperature_mismatch_channel: ComponentOutput = self.add_output(
            self.component_name,
            self.TemperatureMismatch,
            lt.LoadTypes.TEMPERATURE,
            lt.Units.KELVIN,
            output_description=f"For the culling model, this is the mismatch between the theoretical and actual "
                "(after culling) temperature in the tank. t_mismatch = t_theoretical - t_actual."
        )

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

    def build(self, heat_exchanger_is_present: bool) -> None:
        """Build function.

        The function sets important constants an parameters for the calculations.
        """
        # specific heat capacities
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
        if self.heat_exchanger_is_present:
            self.factor_for_water_storage_portion = 1
            self.factor_for_water_input_portion = 0
        # if heat exchanger is not present, the water temperatures in the storage are more stratified
        # here a mixing factor is calcualted
        else:
            (
                self.factor_for_water_storage_portion,
                self.factor_for_water_input_portion,
            ) = self.calculate_mixing_factor_for_water_temperature_outputs()

    # ============================================================================================
    # ===== Simulation functions =================================================================
    # ============================================================================================

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
        t_water_from_hds_in_c = stsv.get_input_value(self.water_temperature_heat_distribution_system_input_channel)
        water_mass_flow_rate_from_hds_in_kg_per_s = stsv.get_input_value(self.water_mass_flow_rate_heat_distribution_system_input_channel)
        if self.is_parallel():
            t_water_from_hg_in_c = stsv.get_input_value(self.water_temperature_heat_generator_input_channel)
            t_water_from_hg2_in_c = stsv.get_input_value(self.water_temperature_secondary_heat_generator_input_channel)
            water_flow_from_hg_in_kg_per_s = stsv.get_input_value(self.water_mass_flow_rate_heat_generator_input_channel)
            water_flow_from_hg2_in_kg_per_s = stsv.get_input_value(self.water_mass_flow_rate_secondary_heat_generator_input_channel)
        else:  # buffer tank is in series, therefore it only gets water input from the hds
            t_water_from_hg_in_c = 0
            water_flow_from_hg_in_kg_per_s = 0
            t_water_from_hg2_in_c = 0
            water_flow_from_hg2_in_kg_per_s = 0

        # check water temperature limits
        if not (0 < self.mean_water_temperature_in_water_storage_in_celsius < 90):
            raise ValueError(f"""The water temperature in the water storage is with 
                {self.mean_water_temperature_in_water_storage_in_celsius}°C way too high or too low.""")

        # get water masses from flow rates for simplicity later
        water_mass_from_hg_in_kg = water_flow_from_hg_in_kg_per_s * self.seconds_per_timestep
        water_mass_from_hg2_in_kg = water_flow_from_hg2_in_kg_per_s * self.seconds_per_timestep
        water_mass_from_hds_in_kg = water_mass_flow_rate_from_hds_in_kg_per_s * self.seconds_per_timestep
        water_mass_sum = water_mass_from_hg_in_kg + water_mass_from_hg2_in_kg + water_mass_from_hds_in_kg

        # I need to pass this shit at least three times, may as well prepare that here
        some_kwargs = {
            "water_temperature_input_of_secondary_side_in_celsius": t_water_from_hds_in_c,
            "water_temperature_from_heat_generator_in_celsius": t_water_from_hg_in_c,
            "water_mass_in_storage_in_kg": self.water_mass_in_storage_in_kg,
            "mass_of_input_water_flows_from_heat_generator_in_kg": water_mass_from_hg_in_kg,
            "water_temperature_from_secondary_heat_generator_in_celsius": t_water_from_hg2_in_c,
            "mass_of_input_water_flows_from_secondary_heat_generator_in_kg": water_mass_from_hg2_in_kg,
            "mass_of_input_water_flows_of_secondary_side_in_kg": water_mass_from_hds_in_kg,
            "previous_mean_water_temperature_in_water_storage_in_celsius": self.state.mean_water_temperature_in_celsius
        }

        # also useful
        if water_mass_sum > 0:
            weighted_average_inflow_temperature = (
                t_water_from_hg_in_c * water_mass_from_hg_in_kg
                + t_water_from_hg2_in_c * water_mass_from_hg2_in_kg
                + t_water_from_hds_in_c * water_mass_from_hds_in_kg
            ) / water_mass_sum
        else: #! Is this good?
            weighted_average_inflow_temperature = (
                t_water_from_hg_in_c + t_water_from_hg2_in_c + t_water_from_hds_in_c
            ) / 3


        # ----------------------------------------------------------------------------------------
        # ----- Calculations ---------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------

        # note: I removed the heat_exchanger_present condition, we always assume perfect mixing now.

        # Standard model
        if self.config.simulation_model in ["standard", "hisim"]:
            t_out_1 = self.mean_water_temperature_in_water_storage_in_celsius
            t_out = self.state.mean_water_temperature_in_celsius
            t_new_in_storage = self.calculate_mean_water_temperature_in_water_storage(
                **some_kwargs, explicit_or_implicit="implicit")
        # Explicit Euler calculation
        elif self.config.simulation_model == "explicit":
            t_out = self.state.mean_water_temperature_in_celsius
            t_new_in_storage = self.calculate_mean_water_temperature_in_water_storage(
                **some_kwargs, explicit_or_implicit="explicit")
            if water_mass_sum > self.water_mass_in_storage_in_kg and self.no_explicit_warning_yet:  # stability condition not met
                log.warning("Using explicit Euler in buffer tank even though the stability criterion "
                    "is not met! If this simulation fails, consider reducing seconds_per_timestep! "
                    "The stability criterion is that the water flow must be smaller than the tank volume."
                    f"\n  Timestep: {timestep}"
                    f"\n  Water flow from heat generator: {water_flow_from_hg_in_kg_per_s} kg/s"
                    f"\n  Water flow from 2nd heat generator: {water_flow_from_hg2_in_kg_per_s} kg/s"
                    f"\n  Water flow from heat distribution: {water_mass_flow_rate_from_hds_in_kg_per_s} kg/s"
                    f"\n  Total resulting water mass flow in this time step: {water_mass_sum} kg"
                    f"\n  Volume of the buffer tank: {self.water_mass_in_storage_in_kg} kg")
                self.no_explicit_warning_yet = False
        # Implicit Euler calculation
        elif self.config.simulation_model == "implicit":
            t_new_in_storage = self.calculate_mean_water_temperature_in_water_storage(
                **some_kwargs, explicit_or_implicit="implicit")
            t_out = t_new_in_storage
        # Analytical solving (not implemented)
        elif self.config.simulation_model == "analytical":
            raise NotImplementedError("Analytical model not implemented yet! Choose a different one.")
        # Explicit Euler with bypass solution for unstable regime
        elif self.config.simulation_model == "explicit_with_bypass":
            if water_mass_sum <= self.water_mass_in_storage_in_kg:  # stability condition met
                t_out = self.state.mean_water_temperature_in_celsius
                t_new_in_storage = self.calculate_mean_water_temperature_in_water_storage(
                    **some_kwargs, explicit_or_implicit="explicit")
            else:
                log.information("Stability criterion in buffer tank not met. Using bypass to compensate.")
                # t_out is simply weighted average of the inflow temperatures
                t_out = weighted_average_inflow_temperature
                # buffer temperature stays as it is because the tank gets bypassed entirely
                t_new_in_storage = self.mean_water_temperature_in_water_storage_in_celsius
        # Explicit Euler with culling solution for unstable regime
        elif self.config.simulation_model == "explicit_with_culling":
            t_out = self.state.mean_water_temperature_in_celsius
            t_new_in_storage, t_mismatch = self.calc_culling_model(
                some_kwargs = some_kwargs,
                previous_mismatch = self.state.temperature_mismatch,
                t_in_storage = self.state.mean_water_temperature_in_celsius,
                weighted_average_inflow_temperature = weighted_average_inflow_temperature
            )
            self.state.temperature_mismatch = t_mismatch
        # wtf did you put in lol
        else:
            raise KeyError(f"Simulation model not recognized: {self.config.simulation_model}")

        # new temperatures (here, I interface with old code, so that's the reason for the redundancies)
        # Todo: Remove redundancies and extract the code that is copied between dhw and sh buffer tank
        t_water_to_hds_in_c = t_out
        t_water_to_hg_in_c = t_out
        t_water_to_hg2_in_c = t_out
        self.mean_water_temperature_in_water_storage_in_celsius = t_new_in_storage

        # calc thermal power and energies
        p_therm_from_hg_in_W = self.calc_thermal_power(
            mass_flow = water_flow_from_hg_in_kg_per_s,
            t_diff = t_water_from_hg_in_c-self.state.mean_water_temperature_in_celsius)
        p_therm_from_hg2_in_W = self.calc_thermal_power(
            mass_flow = water_flow_from_hg2_in_kg_per_s,
            t_diff = t_water_from_hg2_in_c-self.state.mean_water_temperature_in_celsius)
        p_therm_to_hds_in_W = self.calc_thermal_power(
            mass_flow = water_mass_flow_rate_from_hds_in_kg_per_s,
            t_diff = self.state.mean_water_temperature_in_celsius-t_water_from_hds_in_c)

        e_therm_in_storage_prev_in_Wh = self.calc_thermal_energy(
            t_water = self.state.mean_water_temperature_in_celsius, 
            mass = self.water_mass_in_storage_in_kg)
        e_therm_in_storage_current_in_Wh = self.calc_thermal_energy(
            t_water = self.mean_water_temperature_in_water_storage_in_celsius,
            mass = self.water_mass_in_storage_in_kg)

        e_therm_input_from_hg_in_Wh = p_therm_from_hg_in_W * self.seconds_per_timestep / 3600
        e_therm_input_from_hg2_in_Wh = p_therm_from_hg2_in_W * self.seconds_per_timestep / 3600
        e_therm_output_to_hds_in_Wh = p_therm_to_hds_in_W * self.seconds_per_timestep / 3600
        
        e_therm_increase_in_Wh = e_therm_in_storage_current_in_Wh - e_therm_in_storage_prev_in_Wh

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
        stsv.set_output_value(
            self.temperature_mismatch_channel,
            self.state.temperature_mismatch)

        # Set state. Except mismatch, which was set earlier inside the if condition.
        self.state.heat_loss_in_watt, t_loss = self.calculate_heat_loss_and_temperature_loss(
            storage_surface_in_m2=self.storage_surface_in_m2,
            mean_water_temperature_in_water_storage_in_celsius=self.mean_water_temperature_in_water_storage_in_celsius,
            heat_transfer_coefficient_in_watt_per_m2_per_kelvin=self.heat_transfer_coefficient_in_watt_per_m2_per_kelvin,
            mass_in_storage_in_kg=self.water_mass_in_storage_in_kg,
            ambient_temperature_in_celsius=self.ambient_temperature_in_celsius,
        )
        self.state.temperature_loss_in_celsius_per_timestep = t_loss * self.seconds_per_timestep
        self.state.mean_water_temperature_in_celsius = (
            self.mean_water_temperature_in_water_storage_in_celsius - t_loss * self.seconds_per_timestep)

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
        # return result
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
            # filters
            if not output.component_name == self.component_name: continue
            if not output.field_name == self.StandbyHeatLoss: continue
            if not output.unit == lt.Units.WATT: continue
            # calc heat loss
            temp = postprocessing_results.iloc[:, index]
            heat_loss_in_watt = temp.loc[temp > 0.0]
            # get energy from power
            heat_loss_in_kilowatt_hour = KpiHelperClass.compute_total_energy_from_power_timeseries(
                power_timeseries_in_watt=heat_loss_in_watt,
                timeresolution=self.my_simulation_parameters.seconds_per_timestep,
            )
            heat_loss_in_kilowatt_hour = round(heat_loss_in_kilowatt_hour, 1)
            heat_loss_entry = KpiEntry(
                name="Standby heat loss of Hot water storage",
                unit="kWh",
                value=heat_loss_in_kilowatt_hour,
                tag=KpiTagEnumClass.STORAGE_HOT_WATER_SPACE_HEATING,
                description=self.component_name,
            )
            list_of_kpi_entries.append(heat_loss_entry)
        # return result
        return list_of_kpi_entries

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
        # Basic attributes
        self.my_simulation_parameters = my_simulation_parameters
        self.config = config
        component_name = self.get_component_name()
        super().__init__(
            name=component_name,
            my_simulation_parameters=my_simulation_parameters,
            my_config=config,
            my_display_config=my_display_config,
        )
        # flow rate and mode
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
        if self.controller_mode == "on" and flow_from_hg_in_kg_per_s == 0:
            return "off" # turn off when no water flow
        elif self.controller_mode == "off" and flow_from_hg_in_kg_per_s != 0:
            return "on" # turn on when water flow
        else:
            return self.controller_mode  # return previous mode.


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
    WaterTemperatureToSecondaryHeatGenerator = "WaterTemperatureToSecondaryHeatGenerator"
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

    TemperatureMismatch = "TemperatureMismatch"

    # ============================================================================================
    # ===== The __init__ function and its helpers, including build() =============================
    # ============================================================================================

    @utils.measure_execution_time
    def __init__(
        self,
        my_simulation_parameters: SimulationParameters,
        config: SimpleDHWStorageConfig,
        my_display_config: DisplayConfig = DisplayConfig(),
    ) -> None:
        """Construct all the neccessary attributes."""
        # Basic __init__() stuff and super().__init__()
        self.my_simulation_parameters = my_simulation_parameters
        self.config = config
        component_name = self.get_component_name()
        super().__init__(
            name=component_name,
            my_simulation_parameters=my_simulation_parameters,
            my_config=config,
            my_display_config=my_display_config,
        )
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
        # add inputs, outputs and default connections
        self.add_all_inputs()
        self.add_all_outputs()
        self.add_default_connections(self.get_default_connections_from_more_advanced_heat_pump())
        self.add_default_connections(self.get_default_connections_from_generic_dhw_boiler())
        self.add_default_connections(self.get_default_connections_from_district_heating())
        self.add_default_connections(self.get_default_connections_from_utsp())
        self.add_default_connections(self.get_default_connections_from_solar_thermal_system())
        self.add_default_connections(self.get_default_connections_from_electric_heating())

    def add_all_inputs(self):
        """Adds all the inputs of the building as attributes. To be used in __init__."""
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

    def add_all_outputs(self):
        """Adds all the outputs of the building as attributes. To be used in __init__."""
        self.water_temperature_to_heat_generator_channel: ComponentOutput = self.add_output(
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
        self.temperature_mismatch_channel: ComponentOutput = self.add_output(
            self.component_name,
            self.TemperatureMismatch,
            lt.LoadTypes.TEMPERATURE,
            lt.Units.KELVIN,
            output_description=f"For the culling model, this is the mismatch between the theoretical and actual "
                "(after culling) temperature in the tank. t_mismatch = t_theoretical - t_actual."
        )

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

    def build(self) -> None:
        """Build function.

        The function sets important constants an parameters for the calculations.
        """
        # fresh and warm drinking water temperatures
        self.drain_water_temperature = configuration.HouseholdWarmWaterDemandConfig.freshwater_temperature
        self.warm_water_temperature = (
            configuration.HouseholdWarmWaterDemandConfig.ww_temperature_demand
            - configuration.HouseholdWarmWaterDemandConfig.temperature_difference_hot
        )
        # specific heat capacities
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

    # ============================================================================================
    # ===== Simulation functions =================================================================
    # ============================================================================================

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
        # get inputs from dhw
        t_water_input_of_dhw_in_c = self.drain_water_temperature
        t_water_output_to_dhw_in_c = self.warm_water_temperature
        water_flow_of_dhw_in_kg_per_s = (
            stsv.get_input_value(self.water_consumption_channel)
            * self.density_water_at_40_degree_celsius_in_kg_per_liter
            / self.seconds_per_timestep
        )
        t_water_from_hg_in_c = stsv.get_input_value(self.water_temperature_heat_generator_input_channel)
        water_flow_from_hg_in_kg_per_s = stsv.get_input_value(self.water_mass_flow_rate_heat_generator_input_channel)
        t_water_from_hg2_in_c = stsv.get_input_value(self.water_temperature_secondary_heat_generator_input_channel)
        water_flow_from_hg2_in_kg_per_s = stsv.get_input_value(self.water_mass_flow_rate_secondary_heat_generator_input_channel)

        # Water Temperature Limit Check
        msg = "The water temperature in the DHW water storage is with {}°C way too {}."
        if self.mean_water_temperature_in_water_storage_in_celsius > 90:
            raise ValueError(msg.format(self.mean_water_temperature_in_water_storage_in_celsius, "high"))
        if self.mean_water_temperature_in_water_storage_in_celsius < 0:
            raise ValueError(msg.format(self.mean_water_temperature_in_water_storage_in_celsius, "low"))

        # calc water masses for simplicity later
        water_mass_from_hg_in_kg = water_flow_from_hg_in_kg_per_s * self.seconds_per_timestep
        water_mass_from_hg2_in_kg = water_flow_from_hg2_in_kg_per_s * self.seconds_per_timestep
        water_mass_of_dhw_in_kg = water_flow_of_dhw_in_kg_per_s * self.seconds_per_timestep
        water_mass_sum = water_mass_from_hg_in_kg + water_mass_from_hg2_in_kg + water_mass_of_dhw_in_kg

        # I need to pass this shit at least three times, may as well prepare that here
        some_kwargs = {
            "water_temperature_input_of_secondary_side_in_celsius": t_water_input_of_dhw_in_c,
            "water_temperature_from_heat_generator_in_celsius": t_water_from_hg_in_c,
            "water_mass_in_storage_in_kg": self.water_mass_in_storage_in_kg,
            "mass_of_input_water_flows_from_heat_generator_in_kg": water_mass_from_hg_in_kg,
            "water_temperature_from_secondary_heat_generator_in_celsius": t_water_from_hg2_in_c,
            "mass_of_input_water_flows_from_secondary_heat_generator_in_kg": water_mass_from_hg2_in_kg,
            "mass_of_input_water_flows_of_secondary_side_in_kg": water_mass_of_dhw_in_kg,
            "previous_mean_water_temperature_in_water_storage_in_celsius": self.state.mean_water_temperature_in_celsius
        }

        # also useful
        if water_mass_sum > 0:
            weighted_average_inflow_temperature = (
                t_water_from_hg_in_c * water_mass_from_hg_in_kg
                + t_water_from_hg2_in_c * water_mass_from_hg2_in_kg
                + t_water_input_of_dhw_in_c * water_mass_of_dhw_in_kg
            ) / water_mass_sum
        else:
            weighted_average_inflow_temperature = (
                t_water_from_hg_in_c + t_water_from_hg2_in_c + t_water_input_of_dhw_in_c
            ) / 3

        # ----------------------------------------------------------------------------------------
        # ----- Calculations ---------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------

        # Standard model
        if self.config.simulation_model in ["standard", "hisim"]:
            t_out = self.state.mean_water_temperature_in_celsius
            t_new_in_storage = self.calculate_mean_water_temperature_in_water_storage(
                **some_kwargs, explicit_or_implicit="implicit")
        # Explicit Euler calculation
        elif self.config.simulation_model == "explicit":
            t_out = self.state.mean_water_temperature_in_celsius
            t_new_in_storage = self.calculate_mean_water_temperature_in_water_storage(
                **some_kwargs, explicit_or_implicit="explicit")
            if water_mass_sum > self.water_mass_in_storage_in_kg:  # stability condition not met
                log.warning("Using explicit Euler in buffer tank even though the stability criterion "
                    "is not met! If this simulation fails, consider reducing seconds_per_timestep! "
                    "The stability criterion is that the water flow must be smaller than the tank volume."
                    f"\n  Water flow from heat generator: {water_flow_from_hg_in_kg_per_s} kg/s"
                    f"\n  Water flow from 2nd heat generator: {water_flow_from_hg2_in_kg_per_s} kg/s"
                    f"\n  Water flow from domestic hot water: {water_flow_of_dhw_in_kg_per_s} kg/s"
                    f"\n  Total resulting water mass flow in this time step: {water_mass_sum} kg"
                    f"\n  Volume of the buffer tank: {self.water_mass_in_storage_in_kg} kg")
        # Implicit Euler calculation
        elif self.config.simulation_model == "implicit":
            t_new_in_storage = self.calculate_mean_water_temperature_in_water_storage(
                **some_kwargs, explicit_or_implicit="implicit")
            t_out = t_new_in_storage
        # Analytical solving (not implemented)
        elif self.config.simulation_model == "analytical":
            raise NotImplementedError("Analytical model not implemented yet! Choose a different one.")
        # Explicit Euler with bypass solution for unstable regime
        elif self.config.simulation_model == "explicit_with_bypass":
            if water_mass_sum <= self.water_mass_in_storage_in_kg:  # stability condition met
                t_out = self.state.mean_water_temperature_in_celsius
                t_new_in_storage = self.calculate_mean_water_temperature_in_water_storage(
                    **some_kwargs, explicit_or_implicit="explicit")
            else:
                log.information("Stability criterion in buffer tank not met. Using bypass to compensate.")
                # t_out is simply weighted average of the inflow temperatures
                t_out = weighted_average_inflow_temperature
                # buffer temperature stays as it is because the tank gets bypassed entirely
                t_new_in_storage = self.mean_water_temperature_in_water_storage_in_celsius
        # Explicit Euler with culling solution for unstable regime
        elif self.config.simulation_model == "explicit_with_culling":
            t_out = self.state.mean_water_temperature_in_celsius
            t_new_in_storage, t_mismatch = self.calc_culling_model(
                some_kwargs = some_kwargs,
                previous_mismatch = self.state.temperature_mismatch,
                t_in_storage = self.state.mean_water_temperature_in_celsius,
                weighted_average_inflow_temperature = weighted_average_inflow_temperature
            )
            self.state.temperature_mismatch = t_mismatch
        # wtf did you put in lol
        else:
            raise KeyError(f"Simulation model not recognized: {self.config.simulation_model}")
        
        # new temperatures (here, I interface with old code, so that's the reason for the redundancies)
        # Todo: Remove redundancies and extract the code that is copied between dhw and sh buffer tank
        t_water_to_hg_in_c = t_out
        t_water_to_hg2_in_c = t_out
        self.mean_water_temperature_in_water_storage_in_celsius = t_new_in_storage

        # calc thermal power and energies
        P_th_from_hg_in_W = self.calc_thermal_power(
            mass_flow = water_flow_from_hg_in_kg_per_s,
            t_diff = t_water_from_hg_in_c - self.state.mean_water_temperature_in_celsius)
        P_th_from_hg2_in_W = self.calc_thermal_power(
            mass_flow = water_flow_from_hg2_in_kg_per_s,
            t_diff = t_water_from_hg2_in_c - self.state.mean_water_temperature_in_celsius)
        P_th_to_dhw_in_W = self.calc_thermal_power(
            mass_flow = water_flow_of_dhw_in_kg_per_s,
            t_diff = t_water_output_to_dhw_in_c - t_water_input_of_dhw_in_c)

        E_th_in_storage_previous_in_Wh = self.calc_thermal_energy(
            t_water = self.state.mean_water_temperature_in_celsius,
            mass = self.water_mass_in_storage_in_kg)
        E_th_in_storage_current_in_Wh = self.calc_thermal_energy(
            t_water = self.mean_water_temperature_in_water_storage_in_celsius,
            mass = self.water_mass_in_storage_in_kg)

        E_th_input_from_hg_in_Wh = P_th_from_hg_in_W * self.seconds_per_timestep / 3600
        E_th_input_from_hg2_in_Wh = P_th_from_hg2_in_W * self.seconds_per_timestep / 3600
        E_th_output_to_dhw_in_Wh = P_th_to_dhw_in_W * self.seconds_per_timestep / 3600

        E_th_in_storage_increase_in_Wh = E_th_in_storage_current_in_Wh - E_th_in_storage_previous_in_Wh

        # ----------------------------------------------------------------------------------------
        # ----- Set outputs ----------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------

        stsv.set_output_value(
            self.water_temperature_to_heat_generator_channel,
            t_water_to_hg_in_c)
        stsv.set_output_value(
            self.water_temperature_secondary_heat_generator_output_channel,
            t_water_to_hg2_in_c)
        stsv.set_output_value(
            self.water_temperature_from_heat_generator_channel,
            t_water_from_hg_in_c)
        stsv.set_output_value(
            self.water_temperature_from_secondary_heat_generator_channel,
            t_water_from_hg2_in_c)
        stsv.set_output_value(
            self.water_temperature_mean_channel,
            self.state.mean_water_temperature_in_celsius)
        stsv.set_output_value(
            self.temperature_loss_channel,
            self.state.temperature_loss_in_celsius_per_timestep)
        stsv.set_output_value(
            self.thermal_energy_in_storage_channel,
            E_th_in_storage_current_in_Wh)
        stsv.set_output_value(
            self.thermal_energy_from_heat_generator_channel,
            E_th_input_from_hg_in_Wh)
        stsv.set_output_value(
            self.thermal_energy_from_secondary_heat_generator_channel,
            E_th_input_from_hg2_in_Wh)
        stsv.set_output_value(
            self.thermal_energy_dhw_channel,
            E_th_output_to_dhw_in_Wh)
        stsv.set_output_value(
            self.thermal_energy_increase_in_storage_channel,
            E_th_in_storage_increase_in_Wh)
        stsv.set_output_value(
            self.stand_by_heat_loss_channel,
            self.state.heat_loss_in_watt)
        stsv.set_output_value(
            self.thermal_power_dhw_channel,
            P_th_to_dhw_in_W)
        stsv.set_output_value(
            self.thermal_power_from_heat_generator_channel,
            P_th_from_hg_in_W)
        stsv.set_output_value(
            self.thermal_power_from_secondary_heat_generator_channel,
            P_th_from_hg2_in_W)
        stsv.set_output_value(
            self.water_mass_flow_rate_dhw_output_channel,
            water_flow_of_dhw_in_kg_per_s)
        stsv.set_output_value(
            self.temperature_mismatch_channel,
            self.state.temperature_mismatch)

        # Set state. Except mismatch, which was set earlier inside the if condition.
        self.state.heat_loss_in_watt, t_loss = self.calculate_heat_loss_and_temperature_loss(
            storage_surface_in_m2=self.storage_surface_in_m2,
            mean_water_temperature_in_water_storage_in_celsius=self.mean_water_temperature_in_water_storage_in_celsius,
            heat_transfer_coefficient_in_watt_per_m2_per_kelvin=self.heat_transfer_coefficient_in_watt_per_m2_per_kelvin,
            mass_in_storage_in_kg=self.water_mass_in_storage_in_kg,
            ambient_temperature_in_celsius=self.ambient_temperature_in_celsius,
        )
        self.state.temperature_loss_in_celsius_per_timestep = t_loss * self.seconds_per_timestep
        self.state.mean_water_temperature_in_celsius = (
            self.mean_water_temperature_in_water_storage_in_celsius - self.state.temperature_loss_in_celsius_per_timestep
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
        # return
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
            # filters
            if not output.component_name == self.component_name: continue
            if not output.field_name == self.StandbyHeatLoss: continue
            if not output.unit == lt.Units.WATT: continue
            # calc heat loss
            temp = postprocessing_results.iloc[:, index]
            heat_loss_in_watt = temp.loc[temp > 0.0]
            # get energy from power
            heat_loss_in_kilowatt_hour = KpiHelperClass.compute_total_energy_from_power_timeseries(
                power_timeseries_in_watt=heat_loss_in_watt,
                timeresolution=self.my_simulation_parameters.seconds_per_timestep,
            )
            heat_loss_in_kilowatt_hour = round(heat_loss_in_kilowatt_hour, 1)
            heat_loss_entry = KpiEntry(
                name="Standby heat loss of DHW storage",
                unit="kWh",
                value=heat_loss_in_kilowatt_hour,
                tag=KpiTagEnumClass.STORAGE_DOMESTIC_HOT_WATER,
                description=self.component_name,
            )
            list_of_kpi_entries.append(heat_loss_entry)
        # return result
        return list_of_kpi_entries
