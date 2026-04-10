from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


SUPPORTED_SIGMA_W_MODES = {"constant", "linear"}


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for the two-stage laser rangefinder simulation.

    The same parameter names are used in `configs/default_config.json`.
    """

    # Distance grid, m.
    L_min: float = 0.0
    L_max: float = 800.0
    dL: float = 10.0

    # Monte Carlo settings:
    # N is the number of trials for one valid distance,
    # M is the number of pulses inside one trial.
    N: int = 2000
    M: int = 8

    # Geometric stage parameters:
    # theta_0 is the base divergence in radians,
    # d_target is the target diameter in meters.
    theta_0: float = 1.0e-4
    d_target: float = 0.5

    # Energy and signal decision thresholds.
    eta_min: float = 0.15
    A0: float = 1.0
    b: float = 0.0
    sigma_A: float = 0.05
    T: float = 0.25
    alpha: float = 0.35

    # Beam wander model:
    # "constant" uses sigma_w_value,
    # "linear" uses sigma_w_value + sigma_w_slope * L.
    sigma_w_mode: str = "linear"
    sigma_w_value: float = 0.01
    sigma_w_slope: float = 8.0e-5

    # Final operating-range requirement and reproducibility seed.
    p_required: float = 0.95
    random_seed: int = 20260409

    # Optional advanced spot-diameter model:
    # d(L) = sqrt(d0^2 + (theta(L) * L)^2).
    use_initial_diameter: bool = False
    d0: float = 0.01

    @property
    def target_radius(self) -> float:
        """Return target radius in meters."""
        return self.d_target / 2.0

    @classmethod
    def from_json(cls, path: Path) -> "SimulationConfig":
        """Load configuration from a JSON file."""
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SimulationConfig":
        """Create a validated configuration from a dictionary-like object."""
        normalized: dict[str, Any] = {key: value for key, value in data.items() if value is not None}
        if "sigma_w_mode" in normalized:
            normalized["sigma_w_mode"] = str(normalized["sigma_w_mode"]).lower()
        config = cls(**normalized)
        config.validate()
        return config

    def merge(self, overrides: Mapping[str, Any]) -> "SimulationConfig":
        """Return a new config with selected values overridden."""
        merged = self.to_dict()
        for key, value in overrides.items():
            if value is not None:
                merged[key] = value
        return self.from_mapping(merged)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration to a plain dictionary."""
        return asdict(self)

    def validate(self) -> None:
        """Validate parameter ranges before the simulation starts."""
        if self.L_max < self.L_min:
            raise ValueError("L_max must be greater than or equal to L_min.")
        if self.dL <= 0:
            raise ValueError("dL must be positive.")
        if self.N <= 0:
            raise ValueError("N must be a positive integer.")
        if self.M <= 0:
            raise ValueError("M must be a positive integer.")
        if self.theta_0 < 0:
            raise ValueError("theta_0 must be non-negative.")
        if self.d_target <= 0:
            raise ValueError("d_target must be positive.")
        if not 0.0 <= self.eta_min <= 1.0:
            raise ValueError("eta_min must be between 0 and 1.")
        if self.A0 < 0:
            raise ValueError("A0 must be non-negative.")
        if self.sigma_A < 0:
            raise ValueError("sigma_A must be non-negative.")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in the interval (0, 1].")
        if self.sigma_w_mode not in SUPPORTED_SIGMA_W_MODES:
            supported = ", ".join(sorted(SUPPORTED_SIGMA_W_MODES))
            raise ValueError(f"sigma_w_mode must be one of: {supported}.")
        if self.sigma_w_value < 0:
            raise ValueError("sigma_w_value must be non-negative.")
        if not 0.0 <= self.p_required <= 1.0:
            raise ValueError("p_required must be between 0 and 1.")
        if self.d0 < 0:
            raise ValueError("d0 must be non-negative.")
