# This module contains an incredibly simple heat generator component that
# provides heat based on a temperature input. Connect the building temperature
# to the input and set a threshold temperature.

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
class SimpleHeatGeneratorConfig(ConfigBase):
    """Configuration of the Simple Heat Generator."""

    @classmethod
    def get_main_classname(cls):
        """Returns the full class name of the base class."""
        return SimpleHeatGenerator.get_full_classname()

    building_name: str = "BUI1"
    name: str = "SimpleHeatGenerator"
    threshold_temperature: float = 20
    thermal_power: float = 5000.0








class SimpleHeatGenerator(Component):
    # Inputs -> Control temperature
    ControlTemperature = "ControlTemperature"

    # Outputs -> Thermal power
    ThermalPower = "ThermalPower"

    def __init__(
        self, 
        my_simulation_parameters: SimulationParameters,
        my_config: SimpleHeatGeneratorConfig=SimpleHeatGeneratorConfig(),
        my_display_config=DisplayConfig()
    ) -> None:
        self.my_config = my_config
        self.my_simulation_parameters = my_simulation_parameters
        super().__init__(name=my_config.name, 
                         my_simulation_parameters=my_simulation_parameters,
                         my_config=my_config,
                         my_display_config=my_display_config)
        
        # state variable is used to store control temperature to avoid nonconverging begavior
        self.state = self.my_config.threshold_temperature

        self.control_temperature_input_channel: ComponentInput = self.add_input(
            self.config.name,
            self.ControlTemperature,
            lt.LoadTypes.TEMPERATURE,
            lt.Units.CELSIUS,
            True,
        )

        self.thermal_power_output = self.add_output(
            self.config.name,
            SimpleHeatGenerator.ThermalPower,
            lt.LoadTypes.HEATING,
            lt.Units.WATT,
            output_description="Thermal Power Output",
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
        # get input and state
        control_temperature_input = stsv.get_input_value(self.control_temperature_input_channel)

        # simple control logic: if control temperature is below threshold, provide thermal power, otherwise not
        if self.state < self.my_config.threshold_temperature:
            thermal_output = self.my_config.thermal_power
        else:
            thermal_output = 0.0

        # set output and new state
        self.state = control_temperature_input
        stsv.set_output_value(self.thermal_power_output, thermal_output)