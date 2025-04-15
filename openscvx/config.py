import numpy as np
from dataclasses import asdict, dataclass, field
from typing import Dict

from openscvx.dynamics import Dynamics


def get_affine_scaling_matrices(n, minimum, maximum):
    S = np.diag(np.maximum(np.ones(n), abs(minimum - maximum) / 2))
    c = (maximum + minimum) / 2
    return S, c


@dataclass
class SimConfig:
    x_bar: np.ndarray
    u_bar: np.ndarray
    initial_state: np.ndarray
    final_state: np.ndarray
    max_state: np.ndarray
    min_state: np.ndarray
    max_control: np.ndarray
    min_control: np.ndarray
    total_time: float
    n_states: int = None
    n_controls: int = None
    inter_sample: int = 30
    dt: float = 0.1
    profiling: bool = False
    debug: bool = False
    solver: str = "QOCO"
    solver_args: dict = field(default_factory=lambda: {'abstol': 1E-6, 'reltol': 1E-9})
    cvxpygen: bool = False
    custom_integrator: bool = True
    S_x: np.ndarray = None
    inv_S_x: np.ndarray = None
    c_x: np.ndarray = None
    S_u: np.ndarray = None
    inv_S_u: np.ndarray = None
    c_u: np.ndarray = None
    diffrax: bool = False
    diffrax_solver: str = 'Tsit5'
    diffrax_args: Dict = field(default_factory=dict)

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
            self.max_control.shape[0] == self.n_controls
        ), f"Max control must have {self.n_controls} elements"
        assert (
            self.min_control.shape[0] == self.n_controls
        ), f"Min control must have {self.n_controls} elements"

        if self.S_x is None or self.c_x is None:
            self.S_x, self.c_x = get_affine_scaling_matrices(
                self.n_states, self.min_state, self.max_state
            )
            # Use the fact that S_x is diagonal to compute the inverse
            self.inv_S_x = np.diag(1 / np.diag(self.S_x))
        if self.S_u is None or self.c_u is None:
            self.S_u, self.c_u = get_affine_scaling_matrices(
                self.n_controls, self.min_control, self.max_control
            )
            self.inv_S_u = np.diag(1 / np.diag(self.S_u))


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
    cost_drop: int = -1
    cost_relax: float = 1.0
    w_tr_adapt: float = 1.0
    w_tr_max: float = None
    w_tr_max_scaling_factor: float = None

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
