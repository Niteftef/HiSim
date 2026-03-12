"""This module contains an alternative water storage component and its supporting classes.
The other options are: simple_water_storage.py, example_storage.py, generic_heat_water_storage.py,
and generic_hot_water_storage.py."""

from typing import List
from hisim.component import Component, ConfigBase, DisplayConfig, SingleTimeStepValues
from hisim.loadtypes import LoadTypes, Units
from hisim.simulationparameters import SimulationParameters



class WaterStorageConfig(ConfigBase):
    """Configuration of the Water Storage."""

    @classmethod
    def get_main_classname(cls):
        """Returns the full class name of the base class."""
        return WaterStorage.get_full_classname()

    name: str # this thing has to be set manually to ensure uniqueness for each instance.
    building_name: str = "BUI1"
    volume: float = 500 # liters or kg, those are treated interchangebly here # TODO: reasonable standard value
    surface_area: float = 4 # m² # todo automatic surface area calculation based on volume and maybe shape
    hull_u_value: float = 2.0 # W/(m²K) # todo reasonable standard value
    # todo implement heat transfer through bad connections
    simulation_model: str = "simple_mixed"

    # If you want to have constant values for any of the inputs, you can set them here.
    # If you set a connection to the channel in the main component, that will override this value.
    connection1_flow_input: float = None # kg/s # type: ignore
    connection1_temperature_input: float = None # °C # type: ignore
    connection2_flow_input: float = None # kg/s # type: ignore
    connection2_temperature_input: float = None # °C # type: ignore
    connection3_flow_input: float = None # kg/s # type: ignore
    connection3_temperature_input: float = None # °C # type: ignore
    connection4_flow_input: float = None # kg/s # type: ignore
    connection4_temperature_input: float = None # °C # type: ignore
    ambient_temperature_input: float = None # °C # type: ignore

    def __init__(self, name: str, **kwargs) -> None:
        """Only the name is required, everything else has default values.
        However, you can overwrite any parameter you want by providing its name and a value.
        Example: config = WaterStorageConfig(name="MyStorage", volume=1000, hull_u_value=0.5).
        Read the class definition for details on the parameters and their default values."""
        self.name = name
        for key, value in kwargs.items(): setattr(self, key, value)

    def set_sim_model(self, model_name_or_id: str | int):
        """Sets the simulation model to be used for this water storage. The simulation model
        defines how the behavior of the storage, temperature, outputs, energy balances, etc.
        are calculated. The default model is simple_mixed.
        
        Models:
            - simple_mixed (1): A simple model that calculates the energy balance of inflows
                and outflows for a given time step and updates the temperature in storage
                accordingly. Outflow temperatures are based on the temperature at the beginning of
                the time step, temperature in storage assumes perfect mixing, i.e., at the end of
                each time step, the entire water in the storage has the same temperature.\\
                This model is simple and fast, but inaccurate for large time steps. 
                It even becomes unstable if the water flow in a time step is larger than the
                storage volume, since it needs to put out more water at the previous temperature
                than there is, leading to an overcooling effect where the temperature drops below 
                the temperature of the inflows.
            - simple_stratified (2): to be added # Todo
            - implicit_mixed (3): A conceptually simple but computationally expensive model.
                Works like simple_mixed, but the outflow temperatures are based on the average of
                the temperature at the beginning and at the end of the time step. This is
                essentially an implicit Euler implementation of the simple_mixed model, which 
                makes it stable over long time steps but requires an iterative solution for each
                time step, leading to possibly many iterations and thus high computational cost.
            - blended_mixed (4): A model that blends the simple_mixed and implicit_mixed models.
                It uses the previous temperature for the outflow temperature, but essentially
                assumes the final temperature for the outflows when calculating the final
                temperature. This may sound weird but makes it possible to calculate the final
                temperature directly since there is no feedback into the outflow temperatures and
                therefore no feedback from the other connected components that might respond
                differently to different outflow temperatures.\\
                This makes this model stable for any time steps, but at the cost of an energy
                imbalance. This energy imbalance is stored in the state and corrected in the next
                time step.

        Args:
            model_name_or_id: The name or id of the simulation model to be used.
        """
        pass # !ahsdlfas df


class WaterStorage(Component):
    """This class represents a flexible water storage component that can be used for any water
    storage. Is has up to 5 generic connections, all of which require a flow and a temperature
    input, while the storage provides the temperature of the output.
    Basically nothing has to be set manually, but almost everything can. See the config class
    for details on which configuration options exist and what the default values are."""

    # None of these connections are mandatory. You can use as many as you like.
    # You can also use a connection but have any of the channels be constant values.
    # This option would be set in the config, see the config.
     
    # Connection 1 is typically for the heat generator
    Connection1FlowInput = "Connection1FlowInput"
    Connection1TemperatureInput = "Connection1TemperatureInput"
    Connection1TemperatureOutput = "Connection1TemperatureOutput"
    # Connection 2 is typically for a secondary heat generator
    Connection2FlowInput = "Connection2FlowInput"
    Connection2TemperatureInput = "Connection2TemperatureInput"
    Connection2TemperatureOutput = "Connection2TemperatureOutput"
    # Connection 3 is typically for the consumer
    Connection3FlowInput = "Connection3FlowInput"
    Connection3TemperatureInput = "Connection3TemperatureInput"
    Connection3TemperatureOutput = "Connection3TemperatureOutput"
    # Connection 4 is left for any additional connection
    Connection4FlowInput = "Connection4FlowInput"
    Connection4TemperatureInput = "Connection4TemperatureInput"
    Connection4TemperatureOutput = "Connection4TemperatureOutput"

    # Additional inputs for additional behaviors
    AmbientTemperatureInput = "AmbientTemperatureInput" # °C
    # Additional outputs for the simulation results
    MeanTemperatureInStorage = "MeanTemperatureInStorage" # °C
    StandbyHeatLoss = "StandbyHeatLoss" # Watts

    # Additional member variables that this class has:
    # self.component_name: str # The name of the component, declared in super init
    # self.inputs: List[ComponentInput] # List of inputs, declared in super init
    # self.outputs: List[ComponentOutput] # List of outputs, declared in super init
    # self.outputs_initialized: bool # Whether the outputs are initialized, declared in super init
    # self.inputs_initialized: bool # Whether the inputs are initialized, declared in super init
    # self.my_simulation_parameters: SimulationParameters # the sim papam, declared in super init
    # self.simulation_repository: SimRepository # idfk, it is also declared in super init
    # self.default_connections: Dict[str, List[ComponentConnection]] # declared in super init
    # self.config = my_config # the config, declared in super init
    # self.my_display_config: DisplayConfig # the display config, declared in super init

    # self.state_temperature: float # °C, the current temperature in the storage, declared in init
    # self.previous_temperature: float # °C, used in save_state & restore_state, declared in init
    # self.state_energy_imbalance: float # J, used for the blended_mixed model, declared in init
    # self.previous_energy_imbalance :float # used in save_state & restore_state, declared in init
    # self.cp: float # J/(kg*K), specific heat capacity of water, declared in init
    # input and output channels for each of the inputs and outputs declared at the top

    def __init__(
        self,
        my_simulation_parameters: SimulationParameters,
        name_or_config: str | WaterStorageConfig,
        my_display_config: DisplayConfig = DisplayConfig(),
        **kwargs
    ) -> None:
        """This constructor is different from the usual hisim components in that it does not require
        you to provide a config object. It is entirely possible to take the default config.
        However, if you take the default config, you have to provide a custom name.
        This is because this component may be used multiple times in the same simulation, and the name
        has to be unique for each instance. It has no further meaning, so don't stress about it.
    
        Args:
            my_simulation_parameters: The simulation parameters of the current simulation.
            name_or_config: Either a custom name for the component, or a custom config object.
                This one is important because the name has to be unique and therefore no default
                value exists. Just choose any name you like, it has no further meaning.
            my_display_config: The display config of the component. Optional, defaults to a
                default display config.
            **kwargs: You can provide additional keyword arguments. These will be passed to the config 
                object. See the config class and its constructor / init function for details.
        """
        # figure out name or config
        if isinstance(name_or_config, str):
            name = name_or_config
            config = WaterStorageConfig(name=name, **kwargs)
        elif isinstance(name_or_config, WaterStorageConfig):
            config = name_or_config
            name = config.name
        else:
            raise TypeError(f"""Type error in WaterStorage: The init argument 'name_or_config' must
                be a string or a WaterStorageConfig. Provided type was: {type(name_or_config)}""")
        # super init, sets self.component_name, self.my_simulation_parameters, self.config and more
        super().__init__(
            name = name,
            my_simulation_parameters = my_simulation_parameters,
            my_config = config,
            my_display_config = my_display_config
        )
        self.config: WaterStorageConfig # this declares the type for coding assistance tools
        # -------------------------------------
        # --- set state and other variables ---
        # -------------------------------------
        self.state_temperature = 60 # °C # Todo: reasonable value!
        self.previous_temperature = self.state_temperature # used in save_state and restore_state
        self.state_energy_imbalance = 0 # J, used for the blended_mixed model
        self.previous_energy_imbalance = self.state_energy_imbalance # I assume you can guess
        self.cp = 4180 # J/(kg*K) specific heat capacity of water
        # this is redundant, but won't change, so I can save some computations by doing it here:
        self.total_heat_capacity = self.config.volume * self.cp # J/K, total cap of the storage
        # ---------------------------------------
        # --- Build input and output channels ---
        # ---------------------------------------
        # Connection in and out, these are boilerplate, so a for loop works
        for i in range(1, 5):
            setattr(self, f"connection{i}_flow_channel", self.add_input(
                object_name=self.component_name,
                field_name=getattr(self, f"Connection{i}FlowInput"),
                load_type=LoadTypes.WATER,
                unit=Units.KG_PER_SEC,
                mandatory=False
            ))
            setattr(self, f"connection{i}_t_in_channel", self.add_input(
                object_name=self.component_name,
                field_name=getattr(self, f"Connection{i}TemperatureInput"),
                load_type=LoadTypes.TEMPERATURE,
                unit=Units.CELSIUS,
                mandatory=False
            ))
            setattr(self, f"connection{i}_t_out_channel", self.add_output(
                object_name=self.component_name,
                field_name=getattr(self, f"Connection{i}TemperatureOutput"),
                load_type=LoadTypes.TEMPERATURE,
                unit=Units.CELSIUS,
                output_description=f"Water temperature output to connection {i} in °C"
            ))
        # Additional inputs for additional behaviors
        self.ambient_temperature_channel = self.add_input(
            object_name=self.component_name,
            field_name=self.AmbientTemperatureInput,
            load_type=LoadTypes.TEMPERATURE,
            unit=Units.CELSIUS,
            mandatory=False
        )
        # Additional outputs for the simulation results
        self.mean_t_in_storage_channel = self.add_output(
            object_name=self.component_name,
            field_name=self.MeanTemperatureInStorage,
            load_type=LoadTypes.TEMPERATURE,
            unit=Units.CELSIUS,
            output_description="Mean temperature in the storage in °C"
        )
        self.standby_heat_loss_channel = self.add_output(
            object_name=self.component_name,
            field_name=self.StandbyHeatLoss,
            load_type=LoadTypes.HEATING,
            unit=Units.WATT,
            output_description="Heat loss to the environment in Watts"
        )
        # Check if the setup of this component is valid and will work
        self.check_self()


    def check_self(self):
        """Checks if the entire setup and definition of this component are valid and will actually
        work in simulations. If not, raises an error. It checks if every flow has a temperature 
        and if the ambient temperature is set (either as an input or a constant)."""
        # for each flow input, there needs to be a corresponding temperature input
        for i in range(1, 5):
            flow_channel = getattr(self, f"connection{i}_flow_channel")
            flow_constant = getattr(self.config, f"connection{i}_flow_input")
            t_in_channel = getattr(self, f"connection{i}_t_in_channel")
            t_in_constant = getattr(self.config, f"connection{i}_temperature_input")
            # if no flow exists, it's fine. Just a temperature is probably a mistake but whatever
            if (flow_channel.src_field_name is None and flow_constant is None): continue
            # else if flow exists, we need a temperature as well
            if (t_in_channel.src_field_name is None and t_in_constant is None):
                raise ValueError(f"""Invalid setup in WaterStorage: Connection {i} has a flow 
                    input but no temperature input. Every flow needs a temperature.""")
        # we need some ambient temperature
        if (self.ambient_temperature_channel.src_field_name is None 
            and self.config.ambient_temperature_input is None):
            raise ValueError("""Invalid setup in WaterStorage: Ambient temperature input missing. 
                Connect the ambient temperature input channel or set a constant in the config.""")

    def write_to_report(self) -> list[str]:
        """Write a report."""
        return self.config.get_string_dict()

    def i_prepare_simulation(self) -> None:
        """Prepare the simulation."""
        pass

    def i_save_state(self) -> None:
        """Save the current state."""
        self.previous_temperature = self.state_temperature
        self.previous_energy_imbalance = self.state_energy_imbalance

    def i_restore_state(self) -> None:
        """Restore the previous state."""
        self.state_temperature = self.previous_temperature
        self.state_energy_imbalance = self.previous_energy_imbalance

    def i_doublecheck(self, timestep: int, stsv: SingleTimeStepValues) -> None:
        """Doublecheck."""
        pass

    def i_simulate(self, timestep: int, stsv: SingleTimeStepValues, force_convergence: bool) -> None:
        """Simulate the heating water storage."""
        if force_convergence: pass
        elif self.config.simulation_model == "simple_mixed":
            self.simulate_simple_mixed(stsv)
        elif self.config.simulation_model == "simple_stratified":
            self.simulate_simple_stratified(timestep, stsv)
        elif self.config.simulation_model == "implicit_mixed":
            self.simulate_implicit_mixed(timestep, stsv)
        elif self.config.simulation_model == "blended_mixed":
            self.simulate_blended_mixed(timestep, stsv)
        else:
            raise ValueError(f"""Invalid simulation model in WaterStorage: 
                {self.config.simulation_model}. Valid options are: simple_mixed,
                simple_stratified, implicit_mixed, and blended_mixed""")

    def simulate_simple_mixed(self, stsv: SingleTimeStepValues) -> None:
        # load inputs ! important: if an input is not connected, stsv.get_input_value returns 0
        # You can check if an input channel is connected to anything by checking if that inputs
        # src_object_name or src_field_name is not None.
        # f.ex.: if self.ambient_temperature_channel.src_field_name is not None: ...
        # get ambient temperature, either from input or config
        if self.ambient_temperature_channel.src_field_name is not None:
            ambient_temperature = stsv.get_input_value(self.ambient_temperature_channel)
        else: ambient_temperature = self.config.ambient_temperature_input
        # use self.state_temperature as output temperature and for standby heat loss
        t_out = self.state_temperature
        heat_loss_power = 0
        for i in range(1, 5):
            values = []
            if self.get_connection_inputs(i, values, stsv):
                flow, t_in = values
                heat_loss_power += flow * (t_out - t_in) * self.cp
        standby_heat_loss = (self.config.hull_u_value * self.config.surface_area * 
                             (self.state_temperature - ambient_temperature))
        heat_loss_power += standby_heat_loss
        # set outputs and update state
        stsv.set_output_value(self.standby_heat_loss_channel, heat_loss_power)
        for i in range(1, 5):
            if self.get_connection_inputs(i, [], stsv): # if the connection inputs are valid
                t_out_channel = getattr(self, f"connection{i}_t_out_channel")
                stsv.set_output_value(t_out_channel, t_out)
        self.state_temperature += (heat_loss_power / self.total_heat_capacity
                                   * self.my_simulation_parameters.seconds_per_timestep)
        stsv.set_output_value(self.mean_t_in_storage_channel, self.state_temperature)

    def simulate_simple_stratified(self, timestep: int, stsv: SingleTimeStepValues) -> None:
        # Todo: Implement
        pass

    def simulate_implicit_mixed(self, timestep: int, stsv: SingleTimeStepValues) -> None:
        # Todo: Implement
        pass

    def simulate_blended_mixed(self, timestep: int, stsv: SingleTimeStepValues) -> None:
        # Todo: Implement
        pass

    # ---------------------------------------------
    # --- Helper amd utility functions ------------
    # ---------------------------------------------

    def get_connection_inputs(self, i: int, values: list, stsv: SingleTimeStepValues) -> bool:
        """Helper function to get the flow and temperature inputs for a given connection 'i'.
        It checks first if a input is connected and otherwise if a constant value is set in the config.
        If values exist for both, they are stored in 'values'and the function returns True.
        Otherwise, 'values' remains unchanged and the function returns False."""
        flow_channel = getattr(self, f"connection{i}_flow_channel")
        t_in_channel = getattr(self, f"connection{i}_t_in_channel")
        flow_val = None
        t_in_val = None
        if flow_channel.src_field_name is not None:
            flow_val = stsv.get_input_value(flow_channel)
        else: flow_val = getattr(self.config, f"connection{i}_flow_input") # if not set, this is None anyways
        if t_in_channel.src_field_name is not None:
            t_in_val = stsv.get_input_value(t_in_channel)
        else: t_in_val = getattr(self.config, f"connection{i}_temperature_input") # same
        if flow_val is not None and t_in_val is not None:
            values = [flow_val, t_in_val]
            return True
        else: return False