from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from simulation import SimulationConfig, SimulationResults, export_results, run_simulation
from simulation.utils import create_run_output_dir, format_optional_distance


DEFAULT_CONFIG_PATH = Path("configs/default_config.json")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Simulate the maximum operating range of a UAV laser rangefinder "
            "in a turbulent near-ground atmosphere."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to a JSON configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Root directory for run outputs. A timestamped subdirectory is created automatically.",
    )
    parser.add_argument("--L-min", dest="L_min", type=float, help="Minimum distance in meters.")
    parser.add_argument("--L-max", dest="L_max", type=float, help="Maximum distance in meters.")
    parser.add_argument("--dL", type=float, help="Distance step in meters.")
    parser.add_argument("--N", type=int, help="Monte Carlo trials per valid distance.")
    parser.add_argument("--M", type=int, help="Number of pulses per Monte Carlo trial.")
    parser.add_argument(
        "--random-seed",
        dest="random_seed",
        type=int,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--sigma-w-mode",
        dest="sigma_w_mode",
        choices=["constant", "linear"],
        help="Beam wander mode.",
    )
    parser.add_argument(
        "--sigma-w-value",
        dest="sigma_w_value",
        type=float,
        help="Constant beam wander sigma or linear intercept in meters.",
    )
    parser.add_argument(
        "--sigma-w-slope",
        dest="sigma_w_slope",
        type=float,
        help="Slope for linear beam wander mode in meters per meter.",
    )
    parser.add_argument(
        "--eta-min",
        dest="eta_min",
        type=float,
        help="Minimum energy fraction required to continue a trial.",
    )
    parser.add_argument(
        "--threshold",
        dest="T",
        type=float,
        help="Detection threshold applied to the final EWMA amplitude.",
    )
    parser.add_argument(
        "--p-required",
        dest="p_required",
        type=float,
        help="Required success probability for the operating range criterion.",
    )
    parser.add_argument(
        "--use-initial-diameter",
        dest="use_initial_diameter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable the optional advanced spot-diameter model with d0.",
    )
    parser.add_argument(
        "--d0",
        type=float,
        help="Initial beam diameter in meters for the optional advanced model.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation and only save CSV plus used_config.json.",
    )
    return parser


def load_config(config_path: Path, overrides: dict[str, Any]) -> SimulationConfig:
    """Load config from JSON and apply CLI overrides."""
    config = SimulationConfig.from_json(config_path)
    return config.merge(overrides)


def build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Collect CLI overrides while leaving unspecified values untouched."""
    keys = [
        "L_min",
        "L_max",
        "dL",
        "N",
        "M",
        "random_seed",
        "sigma_w_mode",
        "sigma_w_value",
        "sigma_w_slope",
        "eta_min",
        "T",
        "p_required",
        "use_initial_diameter",
        "d0",
    ]
    return {key: getattr(args, key) for key in keys}


def print_summary(
    config: SimulationConfig,
    config_path: Path,
    output_dir: Path,
    artifacts: dict[str, Path],
    result: SimulationResults,
) -> None:
    """Print a concise simulation summary."""
    spot_model = "advanced sqrt(d0^2 + (theta*L)^2)" if config.use_initial_diameter else "basic theta(L)*L"
    sigma_w_text = (
        f"constant sigma_w = {config.sigma_w_value:.6f} m"
        if config.sigma_w_mode == "constant"
        else (
            "linear sigma_w(L) = "
            f"{config.sigma_w_value:.6f} + {config.sigma_w_slope:.6f} * L m"
        )
    )

    print("Laser rangefinder simulation summary")
    print(f"  config file: {config_path}")
    print(f"  distance grid: {config.L_min:.3f} .. {config.L_max:.3f} m with step {config.dL:.3f} m")
    print(f"  Monte Carlo: N = {config.N}, M = {config.M}, random_seed = {config.random_seed}")
    print(f"  beam model: theta_0 = {config.theta_0:.6e} rad, spot model = {spot_model}")
    print(f"  target: d_target = {config.d_target:.3f} m, R_t = {config.target_radius:.3f} m")
    print(
        "  signal model: "
        f"eta_min = {config.eta_min:.3f}, A0 = {config.A0:.3f}, b = {config.b:.3f}, "
        f"sigma_A = {config.sigma_A:.3f}, T = {config.T:.3f}, alpha = {config.alpha:.3f}"
    )
    print(f"  beam wander: {sigma_w_text}")
    print(f"  maximum geometric distance: {format_optional_distance(result.max_geometric_distance)}")
    print(
        "  maximum operating distance "
        f"(p >= {config.p_required:.3f}): {format_optional_distance(result.operating_distance)}"
    )
    print(f"  output directory: {output_dir}")
    print("  files produced:")
    for name, path in artifacts.items():
        print(f"    {name}: {path}")


def main() -> int:
    """Run the simulation from the command line."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        config = load_config(args.config, build_overrides(args))
        result = run_simulation(config)
        output_dir = create_run_output_dir(args.output_dir)
        artifacts = export_results(result, config, output_dir, save_plots=not args.no_plots)
        print_summary(config, args.config, output_dir, artifacts, result)
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
