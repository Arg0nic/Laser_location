from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SimulationConfig:
    L_min: float = 0.0
    L_max: float = 800.0
    dL: float = 10.0
    N: int = 3000
    M: int = 8
    theta_0: float = 1.0e-4
    d_target: float = 0.5
    eta_min: float = 0.25
    A0: float = 1.0
    b: float = 0.0
    sigma_A: float = 0.10
    T: float = 0.42
    alpha: float = 0.35
    sigma_w_value: float = 0.03
    sigma_w_slope: float = 1.8e-4
    p_required: float = 0.95
    random_seed: int | None = 20260411

    @property
    def target_radius(self) -> float:
        return self.d_target / 2.0

    @classmethod
    def from_json(cls, path: Path) -> "SimulationConfig":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SimulationConfig":
        normalized: dict[str, Any] = {}
        for key, value in data.items():
            if value is not None or key == "random_seed":
                normalized[key] = value
        sigma_mode = normalized.pop("sigma_w_mode", None)
        if sigma_mode is not None and str(sigma_mode).lower() != "linear":
            raise ValueError("Only linear sigma_w_mode is supported in this project.")
        normalized.pop("use_initial_diameter", None)
        normalized.pop("d0", None)
        config = cls(**normalized)
        config.validate()
        return config

    def merge(self, overrides: Mapping[str, Any]) -> "SimulationConfig":
        merged = self.to_dict()
        for key, value in overrides.items():
            if value is not None:
                merged[key] = value
        return self.from_mapping(merged)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
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
        if self.sigma_w_value < 0:
            raise ValueError("sigma_w_value must be non-negative.")
        if not 0.0 <= self.p_required <= 1.0:
            raise ValueError("p_required must be between 0 and 1.")
