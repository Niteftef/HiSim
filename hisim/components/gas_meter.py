"""Gas meter module. """
# clean
from dataclasses import dataclass
from typing import List

from dataclasses_json import dataclass_json
import pandas as pd

from hisim import component as cp
from hisim import loadtypes as lt
from hisim.component import ComponentInput, OpexCostDataClass
from hisim.dynamic_component import (
    DynamicComponent,
    DynamicConnectionInput,
    DynamicConnectionOutput,
)
from hisim.components.configuration import EmissionFactorsAndCostsForFuelsConfig
from hisim.simulationparameters import SimulationParameters


@dataclass_json
@dataclass
class GasMeterConfig(cp.ConfigBase):

    """Gas Meter Config."""

    @classmethod
    def get_main_classname(cls):
        """Returns the full class name of the base class."""
        return GasMeter.get_full_classname()

    name: str
    consumption_in_kwh: float

    @classmethod
    def get_Gas_meter_default_config(cls):
        """Gets a default GasMeter."""
        return GasMeterConfig(
            name="GasMeter",
            consumption_in_kwh= 0.0
        )


class GasMeter(DynamicComponent):

    """Gas meter class.

    It calculates the Gas consumption dynamically for all components.
    """

    # Outputs
    GasConsumption = "GasConsumption"
    CumulativeConsumption = "CumulativeConsumption"

    def __init__(
        self,
        my_simulation_parameters: SimulationParameters,
        config: GasMeterConfig,
    ):
        """Initialize the component."""
        self.gas_meter_config = config
        self.name = self.gas_meter_config.name
        self.my_component_inputs: List[DynamicConnectionInput] = []
        self.my_component_outputs: List[DynamicConnectionOutput] = []
        super().__init__(
            self.my_component_inputs,
            self.my_component_outputs,
            self.name,
            my_simulation_parameters,
            my_config=config,
        )

        self.production_inputs: List[ComponentInput] = []
        self.consumption_inputs: List[ComponentInput] = []

        self.seconds_per_timestep = self.my_simulation_parameters.seconds_per_timestep
        # Component has states
        self.state = GasMeterState(
            cumulative_production_in_watt_hour=0, cumulative_consumption_in_watt_hour=0
        )
        self.previous_state = self.state.self_copy()

        # Outputs


        self.gas_consumption_channel: cp.ComponentOutput = self.add_output(
            object_name=self.component_name,
            field_name=self.GasConsumption,
            load_type=lt.LoadTypes.GAS,
            unit=lt.Units.WATT_HOUR,
            sankey_flow_direction=False,
            output_description=f"here a description for {self.GasConsumption} will follow.",
        )

        self.cumulative_gas_consumption_channel: cp.ComponentOutput = self.add_output(
            object_name=self.component_name,
            field_name=self.CumulativeConsumption,
            load_type=lt.LoadTypes.GAS,
            unit=lt.Units.WATT_HOUR,
            sankey_flow_direction=False,
            output_description=f"here a description for {self.CumulativeConsumption} will follow.",
        )


    def write_to_report(self):
        """Writes relevant information to report."""
        return self.gas_meter_config.get_string_dict()

    def i_save_state(self) -> None:
        """Saves the state."""
        self.previous_state = self.state.self_copy()

    def i_restore_state(self) -> None:
        """Restores the state."""
        self.state = self.previous_state.self_copy()

    def i_prepare_simulation(self) -> None:
        """Prepares the simulation."""
        pass

    def i_doublecheck(self, timestep: int, stsv: cp.SingleTimeStepValues) -> None:
        """Doublechecks values."""
        pass

    def i_simulate(
        self, timestep: int, stsv: cp.SingleTimeStepValues, force_convergence: bool
    ) -> None:
        """Simulate the grid energy balancer."""

        if timestep == 0:

            self.consumption_inputs = self.get_dynamic_inputs(
                tags=[
                    lt.InandOutputType.FUEL_CONSUMPTION,
                    lt.ComponentType.GAS_HEATER,
                ]
            )

        # Gas #

        # get sum of consumption for all inputs for each iteration

        consumption_in_watt = sum(
            [
                stsv.get_input_value(component_input=elem)
                for elem in self.consumption_inputs
            ]
        )

        # transform watt to watthour
        consumption_in_watt_hour = (
            consumption_in_watt * self.seconds_per_timestep / 3600
        )

        # calculate cumulative consumption
        cumulative_consumption_in_watt_hour = (
            self.state.cumulative_consumption_in_watt_hour
            + consumption_in_watt_hour
        )


        stsv.set_output_value(
            self.gas_consumption_channel, consumption_in_watt_hour,
        )

        stsv.set_output_value(
            self.cumulative_gas_consumption_channel,
            cumulative_consumption_in_watt_hour,
        )

        self.state.cumulative_consumption_in_watt_hour = (
            cumulative_consumption_in_watt_hour
        )

    def get_cost_opex(
        self,
        all_outputs: List,
        postprocessing_results: pd.DataFrame,
    ) -> OpexCostDataClass:
        """Calculate OPEX costs, consisting of Gas costs and revenues."""
        for index, output in enumerate(all_outputs):
            if (
                output.component_name == self.config.name
                and output.field_name == self.GasConsumption
            ):  # Todo: check component name from examples: find another way of using the correct outputs
                self.config.consumption_in_kwh = round(
                    sum(postprocessing_results.iloc[:, index]), 1
                )
        emissions_and_cost_factors = EmissionFactorsAndCostsForFuelsConfig.get_values_for_year(self.my_simulation_parameters.year)
        co2_per_unit = (
            emissions_and_cost_factors.gas_footprint_in_kg_per_kwh
        )
        euro_per_unit = (
            emissions_and_cost_factors.gas_costs_in_euro_per_kwh
        )

        opex_cost_per_simulated_period_in_euro = self.config.consumption_in_kwh * euro_per_unit
        co2_per_simulated_period_in_kg = self.config.consumption_in_kwh * co2_per_unit


        # TODO: hier consumption 0 oder config.consumption_in_kwh eintragen?
        opex_cost_data_class = OpexCostDataClass(
            opex_cost=opex_cost_per_simulated_period_in_euro,
            co2_footprint=co2_per_simulated_period_in_kg,
            consumption=0,
        )


        return opex_cost_data_class


@dataclass
class GasMeterState:

    """GasMeterState class."""

    cumulative_consumption_in_watt_hour: float

    def self_copy(self,):
        """Copy the GasMeterState."""
        return GasMeterState(
            self.cumulative_consumption_in_watt_hour,
        )
