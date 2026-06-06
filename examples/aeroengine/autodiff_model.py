from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.io import loadmat

from pyhb import AutodiffSecondOrderTimeModel, ForcingTerm, LinearOperatorTerm


DEFAULT_MATRIX_PATH = Path(__file__).resolve().parent / "data" / "aero_engine_system_parameter_matrix.mat"


@dataclass(frozen=True)
class AeroEngineAutodiffParameters:
    speed_ratio: float = 1.2
    lp_disk_e: tuple[float, ...] = (20e-6, 20e-6, 30e-6)
    hp_disk_e: tuple[float, ...] = (0e-6, 100e-6, 0e-6, 0e-6, 0e-6, 0e-6, 134e-6, 10e-6)
    lp_disk_m: tuple[float, ...] = (50.292, 51.1704, 73.115)
    hp_disk_m: tuple[float, ...] = (38.2962, 12.1848, 9.1364, 8.6804, 8.2476, 8.2278, 9.4462, 78.436)
    lp_disk_loc: tuple[int, ...] = (2, 3, 19)
    hp_disk_loc: tuple[int, ...] = (23, 24, 25, 26, 27, 28, 29, 33)
    bearing_node_i: int = 18
    bearing_node_o: int = 35
    bearing_di: float = 118.94e-3
    bearing_do: float = 164.064e-3
    bearing_nb: int = 28
    bearing_clearance: float = 2e-6
    bearing_kb: float = 2.5e8


class AeroEngineAutodiffRotorModel(AutodiffSecondOrderTimeModel):
    def __init__(
        self,
        matrix_path: str | Path = DEFAULT_MATRIX_PATH,
        parameters: AeroEngineAutodiffParameters | None = None,
    ) -> None:
        self.parameters = parameters or AeroEngineAutodiffParameters()
        matrix_data = loadmat(matrix_path)
        self.mass = np.asarray(matrix_data["M"], dtype=np.float64)
        self.stiffness = np.asarray(matrix_data["K"], dtype=np.float64)
        self.damping = np.asarray(matrix_data["C"], dtype=np.float64)
        self.gyroscopic = np.asarray(matrix_data["J"], dtype=np.float64)
        self._n_dof = int(self.mass.shape[0])
        if (
            self.mass.shape != self.stiffness.shape
            or self.mass.shape != self.damping.shape
            or self.mass.shape != self.gyroscopic.shape
        ):
            raise ValueError("M, C, J, and K must have the same shape")
        if self.mass.shape[0] != self.mass.shape[1]:
            raise ValueError("system matrices must be square")

        parameters = self.parameters
        self.speed_ratio = float(parameters.speed_ratio)
        self.omega_c = (parameters.bearing_di + self.speed_ratio * parameters.bearing_do) / (
            parameters.bearing_di + parameters.bearing_do
        )
        self.lp_disk_e = np.asarray(parameters.lp_disk_e, dtype=np.float64)
        self.hp_disk_e = np.asarray(parameters.hp_disk_e, dtype=np.float64)
        self.lp_disk_m = np.asarray(parameters.lp_disk_m, dtype=np.float64)
        self.hp_disk_m = np.asarray(parameters.hp_disk_m, dtype=np.float64)
        self.lp_disk_x = self._node_x_indices(parameters.lp_disk_loc)
        self.hp_disk_x = self._node_x_indices(parameters.hp_disk_loc)
        self.lp_disk_y = self.lp_disk_x + self._n_dof // 2
        self.hp_disk_y = self.hp_disk_x + self._n_dof // 2
        self.bearing_ix = self._node_x_indices((parameters.bearing_node_i,))[0]
        self.bearing_ox = self._node_x_indices((parameters.bearing_node_o,))[0]
        self.bearing_iy = self.bearing_ix + self._n_dof // 2
        self.bearing_oy = self.bearing_ox + self._n_dof // 2
        self.roller_phase = 2.0 * np.pi / parameters.bearing_nb * np.arange(
            parameters.bearing_nb,
            dtype=np.float64,
        )

    @property
    def n_dof(self) -> int:
        return self._n_dof

    @property
    def nonlinear_force_dofs(self) -> tuple[int, int, int, int]:
        return (self.bearing_ix, self.bearing_ox, self.bearing_iy, self.bearing_oy)

    @property
    def nonlinear_coordinate_dofs(self) -> tuple[int, int, int, int]:
        return (self.bearing_ix, self.bearing_ox, self.bearing_iy, self.bearing_oy)

    @property
    def bearing_nonlinear_dofs(self) -> tuple[int, int, int, int]:
        return self.nonlinear_coordinate_dofs

    @property
    def autodiff_variables(self) -> tuple[str, ...]:
        return ("x",)

    @property
    def autodiff_parameter_dependent(self) -> bool:
        return False

    def _node_x_indices(self, node_locations: tuple[int, ...]) -> NDArray[np.int64]:
        return np.asarray([2 * location - 2 for location in node_locations], dtype=np.int64)

    def linear_operator_terms(self) -> tuple[LinearOperatorTerm, ...]:
        return (
            LinearOperatorTerm(self.mass, "ddx", 2.0),
            LinearOperatorTerm(self.damping, "dx", 1.0),
            LinearOperatorTerm(self.gyroscopic, "dx", 2.0),
            LinearOperatorTerm(self.stiffness, "x", 0.0),
        )

    def forcing_terms(self, t: NDArray[np.float64]) -> tuple[ForcingTerm, ...]:
        return (ForcingTerm(self._base_unbalance_force(t), 2.0),)

    def _base_unbalance_force(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        force = np.zeros((t.size, self.n_dof), dtype=np.float64)
        lp_me = self.lp_disk_e * self.lp_disk_m
        hp_me = self.hp_disk_e * self.hp_disk_m

        force[:, self.lp_disk_x] = np.cos(t)[:, None] * lp_me[None, :]
        force[:, self.lp_disk_y] = np.sin(t)[:, None] * lp_me[None, :]

        hp_scale = self.speed_ratio**2
        hp_angle = self.speed_ratio * t
        force[:, self.hp_disk_x] = hp_scale * np.cos(hp_angle)[:, None] * hp_me[None, :]
        force[:, self.hp_disk_y] = hp_scale * np.sin(hp_angle)[:, None] * hp_me[None, :]
        return force

    def local_nonlinear_force_torch(
        self,
        t: torch.Tensor,
        local_x: torch.Tensor,
        local_dx: torch.Tensor,
        local_ddx: torch.Tensor,
        parameter: torch.Tensor,
    ) -> torch.Tensor:
        roller_phase = torch.as_tensor(self.roller_phase, dtype=t.dtype, device=t.device)
        theta = roller_phase[:, None] + float(self.omega_c) * t[None, :]
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        dx_io = local_x[:, 0] - local_x[:, 1]
        dy_io = local_x[:, 2] - local_x[:, 3]
        delta = dx_io[None, :] * cos_theta + dy_io[None, :] * sin_theta - float(self.parameters.bearing_clearance)
        active_delta = torch.clamp(delta, min=0.0)

        kb = float(self.parameters.bearing_kb)
        fx = (kb * active_delta.pow(1.5) * cos_theta).sum(dim=0)
        fy = (kb * active_delta.pow(1.5) * sin_theta).sum(dim=0)
        return torch.stack((fx, -fx, fy, -fy), dim=1)
