import numpy as np
import yaml
from dataclasses import asdict, dataclass, field, is_dataclass, fields
from copy import deepcopy
from typing import Union
import jax, jaxlib
import jax.numpy as jnp

from typing import Dict, List

from openscvx.dynamics import Dynamics


# Define a custom representer for NumPy arrays to convert them to lists
def numpy_representer(dumper, data):
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data.tolist(), flow_style=False
    )


# Define a custom representer for PyJit types to convert them to strings
def pjitfunction_representer(dumper, data):
    return dumper.represent_str(str(data))


def compiled_representer(dumper, data):
    return dumper.represent_str(str(data))


def dataclass_to_dict(instance):
    if not is_dataclass(instance):
        return instance
    return {k: dataclass_to_dict(v) for k, v in asdict(instance).items()}


def generate_schema(dataclass_type):
    schema = {}
    for field in fields(dataclass_type):
        field_name = field.name
        field_type = field.type
        if is_dataclass(field_type):
            schema[field_name] = generate_schema(field_type)
        else:
            schema[field_name] = field_name
    return schema


def flat_to_nested_dict(flat_dict, schema):
    nested_dict = {}
    for key, sub_keys in schema.items():
        nested_dict[key] = {
            key: flat_dict[key] for key in sub_keys.keys() if key in flat_dict
        }
    return nested_dict


def numpy_to_list(d):
    if isinstance(d, np.ndarray):
        return d.tolist()
    elif isinstance(d, dict):
        return {k: numpy_to_list(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [numpy_to_list(v) for v in d]
    else:
        return d


def list_to_numpy(d):
    if isinstance(d, list):
        try:
            return np.array(d)
        except:
            return [list_to_numpy(v) for v in d]
    elif isinstance(d, dict):
        return {k: list_to_numpy(v) for k, v in d.items()}
    else:
        return d


def yaml_to_dict(path):
    yaml.add_representer(np.ndarray, numpy_representer)
    yaml.add_representer(jaxlib.xla_extension.PjitFunction, pjitfunction_representer)
    yaml.add_representer(jax._src.stages.Compiled, compiled_representer)
    with open(path, "r") as file:
        params_loaded = yaml.safe_load(file)
    params = list_to_numpy(params_loaded)
    return params


def dict_to_yaml(params, path):
    yaml.add_representer(np.ndarray, numpy_representer)
    yaml.add_representer(jaxlib.xla_extension.PjitFunction, pjitfunction_representer)
    yaml.add_representer(jax._src.stages.Compiled, compiled_representer)
    params_converted = numpy_to_list(params)
    with open(path, "w") as file:
        yaml.dump(params_converted, file, default_flow_style=None, sort_keys=False)


def get_affine_scaling_matrices(n, minimum, maximum):
    S = np.diag(np.maximum(np.ones(n), abs(minimum - maximum) / 2))
    c = (maximum + minimum) / 2
    return S, c


@dataclass
class SimConfig:
    x_bar: np.ndarray
    u_bar: np.ndarray
    initial_state: np.ndarray
    initial_control: np.ndarray
    final_state: np.ndarray
    max_state: np.ndarray
    min_state: np.ndarray
    max_control: np.ndarray
    min_control: np.ndarray
    total_time: float
    n_states: int = None
    n_controls: int = None
    max_dt: float = 1e2
    min_dt: float = 1e-2
    inter_sample: int = 30
    dt: float = 0.1
    profiling: bool = False
    debug: bool = False
    solver: str = "QOCO"
    cvxpygen: bool = False
    S_x: np.ndarray = None
    c_x: np.ndarray = None
    S_u: np.ndarray = None
    c_u: np.ndarray = None

    def __post_init__(self):
        self.n_states = len(self.max_state)
        self.n_controls = len(self.max_control)

        assert (
            len(self.initial_state.value) == self.n_states - 1
        ), f"Initial state must have {self.n_states - 1} elements"
        assert (
            len(self.final_state.value) == self.n_states - 1
        ), f"Final state must have {self.n_states - 1} elements"
        assert (
            self.max_state.shape[0] == self.n_states
        ), f"Max state must have {self.n_states} elements"
        assert (
            self.min_state.shape[0] == self.n_states
        ), f"Min state must have {self.n_states} elements"
        assert (
            self.initial_control.shape[0] == self.n_controls
        ), f"Initial control must have {self.n_controls} elements"
        assert (
            self.max_control.shape[0] == self.n_controls
        ), f"Max control must have {self.n_controls} elements"
        assert (
            self.min_control.shape[0] == self.n_controls
        ), f"Min control must have {self.n_controls} elements"

        if self.S_x is None or self.c_x is None:
            self.S_x, self.c_x = get_affine_scaling_matrices(
                self.n_states, self.min_state, self.max_state
            )
        if self.S_u is None or self.c_u is None:
            self.S_u, self.c_u = get_affine_scaling_matrices(
                self.n_controls, self.min_control, self.max_control
            )


@dataclass
class ScpConfig:
    w_tr: float
    lam_vc: float
    ep_tr: float = 1e-4
    ep_vb: float = 1e-4
    ep_vc: float = 1e-8
    lam_cost: float = 0.0
    lam_vb: float = 0.0
    k_max: int = 200
    n: int = None
    dis_type: str = "FOH"
    uniform_time_grid: bool = False
    fixed_final_time: bool = False
    cost_drop: int = -1
    cost_relax: float = 1.0
    w_tr_adapt: float = 1.0
    w_tr_max: float = None
    w_tr_max_scaling_factor: float = None
    fixed_final_vel: bool = False
    fixed_initial_att: bool = False
    fixed_final_att: bool = False

    def __post_init__(self):
        keys_to_scale = ["w_tr", "lam_vc", "lam_cost", "lam_vb"]
        scale = max(getattr(self, key) for key in keys_to_scale)
        for key in keys_to_scale:
            setattr(self, key, getattr(self, key) / scale)

        if self.w_tr_max_scaling_factor is not None and self.w_tr_max is None:
            self.w_tr_max = self.w_tr_max_scaling_factor * self.w_tr


# Make a new class call VehConfig which takes in a functionhandle
# The function handle will be passed the config object


@dataclass
class Config:
    sim: SimConfig
    scp: ScpConfig
    veh: Dynamics

    def __post_init__(self):
        pass

    @classmethod
    def from_config(cls, config_instance, savedir=None, savefile="config.yaml"):
        return config_instance

    @classmethod
    def from_dict(cls, params_in, savedir=None, savefile="config.yaml"):
        # Generate schema for the Config class automatically based on member dataclasses
        schema = generate_schema(cls)
        # Check if input dictionary is flat (no subdictionaries)
        if not any(key in params_in for key in schema.keys()):
            params_in = flat_to_nested_dict(params_in, schema)

        for key in schema.keys():
            if key not in params_in:
                params_in[key] = {}

        # Converting params_in to class instances
        sim_config = SimConfig(**params_in["sim"])
        scp_config = ScpConfig(**params_in["scp"])

        config_instance = cls(
            sim=sim_config,
            scp=scp_config,
        )

        config_instance = cls.from_config(
            config_instance, savedir=savedir, savefile=savefile
        )

        return config_instance

    @classmethod
    def from_yaml(cls, path, savedir=None, savefile="config.yaml"):
        params_loaded = yaml_to_dict(path)
        return cls.from_dict(params_loaded, savedir=savedir, savefile=savefile)

    def to_yaml(self, path):
        params = asdict(self)
        dict_to_yaml(params, path)
