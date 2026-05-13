# This module contains a simplified buildings component that follows the 1R1C approach.
# This means simplest possible modeling, one thermal zone with a thermal capacity and
# only one conducting connection to the outside.

# ! author Felix

# Imports

# Generic/Built-in
from typing import List
from dataclasses import dataclass
from dataclasses_json import dataclass_json

# Owned
from hisim.component import Component, SingleTimeStepValues, ConfigBase, DisplayConfig, ComponentInput
from hisim import loadtypes as lt
from hisim.simulationparameters import SimulationParameters










@dataclass_json
@dataclass
class Building1R1CConfig(ConfigBase):
    """Configuration of the Random Numbers."""

    @classmethod
    def get_main_classname(cls):
        """Returns the full class name of the base class."""
        return Building1R1C.get_full_classname()

    building_name: str = "BUI1"
    name: str = "Building1R1C"
    u_value: float = 0.5
    area: float = 100.0
    thermal_capacity: float = 100000.0
    initial_internal_temperature: float = 20.0









class Building1R1C(Component):
    """This component models a building with one thermal zone and one thermal connection to the outside (1R1C approach).\\
    It has a single thermal capacity and a single area&u-value for heat loss to the outside.\\
    It can receive thermal power from a number of different sources and uses the ambient temperature as a boundary condition
    for the heat loss to the outside.\\
    The outputs are the internal temperature and the solar gain through windows (if solar radiation inputs are used)."""

    # Inputs -> Heating
    ThermalPowerInput = "ThermalPowerInput"
    HeatingByResidents = "HeatingByResidents"
    HeatingByDevices = "HeatingByDevices"

    # Inputs -> Weather influences
    AmbientTemperature = "AmbientTemperature"
    Altitude = "Altitude" # also called elevation
    Azimuth = "Azimuth"
    ApparentZenith = "ApparentZenith"
    DirectNormalIrradiance = "DirectNormalIrradiance"
    DirectNormalIrradianceExtra = "DirectNormalIrradianceExtra"
    DiffuseHorizontalIrradiance = "DiffuseHorizontalIrradiance"
    GlobalHorizontalIrradiance = "GlobalHorizontalIrradiance"
    TemperatureOutside = "TemperatureOutside"

    # Outputs
    InternalTemperature = "InternalTemperature"
    SolarGainThroughWindows = "SolarGainThroughWindows"


    def __init__(
        self, 
        my_simulation_parameters: SimulationParameters,
        my_config: Building1R1CConfig=Building1R1CConfig(),
        my_display_config=DisplayConfig()
    ) -> None:
        self.my_config = my_config
        self.my_simulation_parameters = my_simulation_parameters
        super().__init__(name=my_config.name, 
                         my_simulation_parameters=my_simulation_parameters,
                         my_config=my_config,
                         my_display_config=my_display_config)
        
        self.state = 20.0 # initial internal temperature in degree Celsius

        self.thermal_power_input_channel: ComponentInput = self.add_input(
            self.config.name,
            self.ThermalPowerInput,
            lt.LoadTypes.HEATING,
            lt.Units.WATT,
            True,
        )

        self.ambient_temperature_channel: ComponentInput = self.add_input(
            self.config.name,
            self.AmbientTemperature,
            lt.LoadTypes.TEMPERATURE,
            lt.Units.CELSIUS,
            True,
        )

        self.internal_temperature_output = self.add_output(
            self.config.name,
            Building1R1C.InternalTemperature,
            lt.LoadTypes.TEMPERATURE,
            lt.Units.CELSIUS,
            output_description="Internal Temperature Output",
        )

    def i_save_state(
        self,
    ) -> None:
        """Save the current state."""
        self.previous_state = self.state

    def i_restore_state(
        self,
    ) -> None:
        """Restore the previous state."""
        self.state = self.previous_state

    def i_doublecheck(self, timestep: int, stsv: SingleTimeStepValues) -> None:
        """Doublechecks."""
        pass

    def i_prepare_simulation(self) -> None:
        """Prepares the simulation."""
        pass

    def write_to_report(self) -> List[str]:
        """Write to report."""
        lines = []
        lines.append(f"Building 1R1C model: {self.config.name}")
        lines.append(f"U-value: {self.my_config.u_value}")
        lines.append(f"Area: {self.my_config.area}")
        lines.append(f"Thermal Capacity: {self.my_config.thermal_capacity}")
        return lines





    def i_simulate(self, timestep: int, stsv: SingleTimeStepValues, force_convergence: bool) -> None:
        """Simulates the component."""
        # get inputs & state
        thermal_power_input = stsv.get_input_value(self.thermal_power_input_channel)
        ambient_temperature = stsv.get_input_value(self.ambient_temperature_channel)
        previous_internal_temperature = self.state

        # calculate internal temperature
        temperature_gain_input = thermal_power_input / self.my_config.thermal_capacity
        heat_loss_ambient = self.my_config.u_value * self.my_config.area * (previous_internal_temperature - ambient_temperature) 
        temperature_loss_ambient = heat_loss_ambient / self.my_config.thermal_capacity
        new_internal_temperature = previous_internal_temperature + temperature_gain_input - temperature_loss_ambient

        # set output and new state
        self.state = new_internal_temperature
        stsv.set_output_value(self.internal_temperature_output, new_internal_temperature)