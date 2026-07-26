#!/usr/bin/env python3
"""Generate a preliminary ride-dynamics screening package.

This module extends the nominal 2D point-mass simulation with:

* rider-frame longitudinal, lateral, and normal specific force;
* filtered acceleration onset rate (jerk) time histories;
* threshold-event detection with exposure durations;
* Monte Carlo sensitivity analysis for vehicle and environmental losses; and
* traceable CSV, Markdown, and PNG outputs.

It is intentionally *not* an ASTM compliance engine. Licensed criteria,
ride-specific data, a validated 3D multi-body model, structural calculations,
physical testing, and approval by qualified engineers/AHJ are still required.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from roller_coaster_dynamics import G, Vehicle, simulate, track_profile

Comparison = Literal["above", "below", "abs_above"]


@dataclass(frozen=True)
class ScreeningRule:
    """User-configurable preliminary threshold, never an ASTM limit."""

    name: str
    signal: str
    comparison: Comparison
    threshold: float
    units: str


@dataclass(frozen=True)
class ScreeningEvent:
    """One contiguous interval that meets a screening-rule condition."""

    rule: str
    signal: str
    comparison: str
    threshold: float
    units: str
    start_time_s: float
    end_time_s: float
    duration_s: float
    start_distance_m: float
    end_distance_m: float
    peak_value: float


# These deliberately conservative-looking values are examples for exercising
# the event detector. They are not copied from and must not be represented as
# ASTM F2291 criteria.
DEFAULT_RULES = (
    ScreeningRule(
        "Illustrative positive-normal review",
        "normal_g",
        "above",
        4.0,
        "g",
    ),
    ScreeningRule(
        "Illustrative low-normal review",
        "normal_g",
        "below",
        0.25,
        "g",
    ),
    ScreeningRule(
        "Illustrative longitudinal review",
        "longitudinal_g",
        "abs_above",
        0.50,
        "g",
    ),
    ScreeningRule(
        "Illustrative onset-rate review",
        "vector_jerk_gps",
        "above",
        15.0,
        "g/s",
    ),
)


def rider_frame_signals(
    results: dict[str, np.ndarray],
    *,
    filter_window_s: float = 0.15,
    polynomial_order: int = 3,
) -> dict[str, np.ndarray]:
    """Calculate rider-frame specific force and filtered onset rates.

    Axes used by this planar model:

    * +X: tangent in the direction of travel;
    * +Y: rider's lateral direction (zero in a strictly planar model);
    * +Z: left-hand track normal, toward the loop center in the loop.

    Specific force is the non-gravitational acceleration an ideal
    accelerometer attached to the vehicle would measure. The X component is
    ``a_t - g_t`` and the Z component is the track normal support load.

    The Savitzky-Golay filter is a numerical screening choice only. A formal
    analysis must use the filtering, coordinate, and event definitions required
    by the applicable licensed standard and test procedure.
    """

    time = results["time"]
    theta = results["theta"]
    tangential_acceleration = results["tangential_acceleration"]

    longitudinal_g = (tangential_acceleration + G * np.sin(theta)) / G
    lateral_g = np.zeros_like(longitudinal_g)
    normal_g = results["g_net"].copy()

    dt = float(np.median(np.diff(time)))
    desired_window = max(polynomial_order + 2, int(round(filter_window_s / dt)))
    if desired_window % 2 == 0:
        desired_window += 1
    maximum_odd_window = len(time) if len(time) % 2 == 1 else len(time) - 1
    window = min(desired_window, maximum_odd_window)
    if window <= polynomial_order:
        raise ValueError("time history is too short for the requested filter")

    components = np.column_stack((longitudinal_g, lateral_g, normal_g))
    filtered = savgol_filter(
        components,
        window_length=window,
        polyorder=polynomial_order,
        axis=0,
        mode="interp",
    )
    jerk = np.gradient(filtered, time, axis=0)  # [g/s]

    return {
        "longitudinal_g": longitudinal_g,
        "lateral_g": lateral_g,
        "normal_g": normal_g,
        "resultant_g": np.linalg.norm(components, axis=1),
        "longitudinal_g_filtered": filtered[:, 0],
        "lateral_g_filtered": filtered[:, 1],
        "normal_g_filtered": filtered[:, 2],
        "longitudinal_jerk_gps": jerk[:, 0],
        "lateral_jerk_gps": jerk[:, 1],
        "normal_jerk_gps": jerk[:, 2],
        "vector_jerk_gps": np.linalg.norm(jerk, axis=1),
        "filter_window_s": np.full_like(time, window * dt),
    }


def _condition(values: np.ndarray, rule: ScreeningRule) -> np.ndarray:
    if rule.comparison == "above":
        return values > rule.threshold
    if rule.comparison == "below":
        return values < rule.threshold
    if rule.comparison == "abs_above":
        return np.abs(values) > rule.threshold
    raise ValueError(f"Unsupported comparison: {rule.comparison}")


def detect_events(
    time: np.ndarray,
    distance: np.ndarray,
    signals: dict[str, np.ndarray],
    rules: tuple[ScreeningRule, ...] | list[ScreeningRule],
) -> list[ScreeningEvent]:
    """Return contiguous threshold intervals and their measured durations."""

    events: list[ScreeningEvent] = []
    for rule in rules:
        if rule.signal not in signals:
            raise KeyError(f"Unknown screening signal: {rule.signal}")
        values = signals[rule.signal]
        mask = _condition(values, rule)
        transitions = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
        starts = np.flatnonzero(transitions == 1)
        stops = np.flatnonzero(transitions == -1) - 1

        for start, stop in zip(starts, stops):
            segment = values[start : stop + 1]
            if rule.comparison == "below":
                peak = float(np.min(segment))
            elif rule.comparison == "abs_above":
                peak = float(segment[np.argmax(np.abs(segment))])
            else:
                peak = float(np.max(segment))

            events.append(
                ScreeningEvent(
                    rule=rule.name,
                    signal=rule.signal,
                    comparison=rule.comparison,
                    threshold=rule.threshold,
                    units=rule.units,
                    start_time_s=float(time[start]),
                    end_time_s=float(time[stop]),
                    duration_s=float(time[stop] - time[start]),
                    start_distance_m=float(distance[start]),
                    end_distance_m=float(distance[stop]),
                    peak_value=peak,
                )
            )
    return events


def run_uncertainty_sweep(
    cases: int,
    seed: int,
) -> list[dict[str, float | int | bool]]:
    """Run bounded random variations of the simplified loss model.

    These ranges are demonstrative assumptions, not manufacturing tolerances:

    * mass: 400--600 kg;
    * rolling coefficient: 0.003--0.009;
    * Cd*A: 0.55--0.95 m^2; and
    * air density: 1.10--1.30 kg/m^3.
    """

    if cases < 1:
        raise ValueError("cases must be at least 1")

    rng = np.random.default_rng(seed)
    track = track_profile()
    rows: list[dict[str, float | int | bool]] = []

    for case_id in range(cases):
        vehicle = Vehicle(
            mass=float(rng.uniform(400.0, 600.0)),
            rolling_mu=float(rng.uniform(0.003, 0.009)),
            drag_area=float(rng.uniform(0.55, 0.95)),
            air_density=float(rng.uniform(1.10, 1.30)),
        )
        row: dict[str, float | int | bool] = {
            "case": case_id,
            **asdict(vehicle),
        }
        try:
            results = simulate(
                track,
                vehicle,
                samples=800,
                max_step=0.08,
            )
            signals = rider_frame_signals(results)
            row.update(
                {
                    "complete": True,
                    "ride_time_s": float(results["time"][-1]),
                    "maximum_speed_mps": float(np.max(results["velocity"])),
                    "minimum_normal_g": float(np.min(signals["normal_g"])),
                    "maximum_normal_g": float(np.max(signals["normal_g"])),
                    "maximum_abs_longitudinal_g": float(
                        np.max(np.abs(signals["longitudinal_g"]))
                    ),
                    "maximum_vector_jerk_gps": float(
                        np.max(signals["vector_jerk_gps"])
                    ),
                }
            )
        except RuntimeError:
            row.update(
                {
                    "complete": False,
                    "ride_time_s": np.nan,
                    "maximum_speed_mps": np.nan,
                    "minimum_normal_g": np.nan,
                    "maximum_normal_g": np.nan,
                    "maximum_abs_longitudinal_g": np.nan,
                    "maximum_vector_jerk_gps": np.nan,
                }
            )
        rows.append(row)
    return rows


def _write_dict_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_time_history(
    path: Path,
    results: dict[str, np.ndarray],
    signals: dict[str, np.ndarray],
) -> None:
    """Write the nominal trace with units embedded in stable column names."""

    columns = {
        "time_s": results["time"],
        "distance_m": results["distance"],
        "horizontal_x_m": results["x"],
        "elevation_z_m": results["z"],
        "speed_mps": results["velocity"],
        "track_angle_rad": results["theta"],
        "curvature_per_m": results["curvature"],
        "radius_m": results["radius"],
        "tangential_acceleration_mps2": results["tangential_acceleration"],
        "longitudinal_g": signals["longitudinal_g"],
        "lateral_g": signals["lateral_g"],
        "normal_g": signals["normal_g"],
        "resultant_g": signals["resultant_g"],
        "longitudinal_jerk_gps": signals["longitudinal_jerk_gps"],
        "lateral_jerk_gps": signals["lateral_jerk_gps"],
        "normal_jerk_gps": signals["normal_jerk_gps"],
        "vector_jerk_gps": signals["vector_jerk_gps"],
    }
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))


def plot_assessment(
    output: Path,
    results: dict[str, np.ndarray],
    signals: dict[str, np.ndarray],
    sweep: list[dict[str, float | int | bool]],
) -> None:
    """Plot directional specific force, onset rate, and uncertainty results."""

    figure, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)
    distance = results["distance"]

    axes[0].plot(distance, signals["longitudinal_g"], label="Longitudinal X")
    axes[0].plot(distance, signals["lateral_g"], label="Lateral Y")
    axes[0].plot(distance, signals["normal_g"], label="Normal Z")
    axes[0].set(
        title="Nominal Rider-Frame Specific Force (Planar Point-Mass Model)",
        xlabel="Distance [m]",
        ylabel="Specific force [g]",
    )
    axes[0].legend(ncol=3)
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        results["time"],
        signals["longitudinal_jerk_gps"],
        label="Longitudinal",
    )
    axes[1].plot(
        results["time"],
        signals["normal_jerk_gps"],
        label="Normal",
    )
    axes[1].plot(
        results["time"],
        signals["vector_jerk_gps"],
        color="black",
        alpha=0.65,
        label="Vector magnitude",
    )
    axes[1].set(
        title="Filtered Acceleration Onset Rate — Screening Filter, Not ASTM Filter",
        xlabel="Time [s]",
        ylabel="Onset rate [g/s]",
    )
    axes[1].legend(ncol=3)
    axes[1].grid(alpha=0.25)

    completed = [row for row in sweep if bool(row["complete"])]
    speed = np.array([float(row["maximum_speed_mps"]) for row in completed])
    normal = np.array([float(row["maximum_normal_g"]) for row in completed])
    rolling = np.array([float(row["rolling_mu"]) for row in completed])
    scatter = axes[2].scatter(
        speed,
        normal,
        c=rolling,
        cmap="viridis",
        edgecolors="black",
        linewidths=0.3,
    )
    colorbar = figure.colorbar(scatter, ax=axes[2], pad=0.02)
    colorbar.set_label("Rolling coefficient, μ")
    axes[2].set(
        title="Bounded Loss-Model Sensitivity Study",
        xlabel="Maximum speed [m/s]",
        ylabel="Maximum normal specific force [g]",
    )
    axes[2].grid(alpha=0.25)

    figure.suptitle(
        "PRELIMINARY ENGINEERING SCREENING — NOT ASTM COMPLIANCE",
        color="#b91c1c",
        fontweight="bold",
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_report(
    path: Path,
    results: dict[str, np.ndarray],
    signals: dict[str, np.ndarray],
    events: list[ScreeningEvent],
    sweep: list[dict[str, float | int | bool]],
    rules: tuple[ScreeningRule, ...] | list[ScreeningRule],
    seed: int,
) -> None:
    """Create a traceable Markdown summary without a compliance verdict."""

    complete = [row for row in sweep if bool(row["complete"])]
    max_normal = np.array([float(row["maximum_normal_g"]) for row in complete])
    min_normal = np.array([float(row["minimum_normal_g"]) for row in complete])
    max_speed = np.array([float(row["maximum_speed_mps"]) for row in complete])

    lines = [
        "# Preliminary Ride-Dynamics Engineering Assessment",
        "",
        "> **NOT AN ASTM COMPLIANCE DETERMINATION.** The thresholds in this",
        "> package are illustrative software-screening inputs. A qualified ride",
        "> engineer must replace them with licensed, applicable requirements and",
        "> validate the model against the physical ride and governing jurisdiction.",
        "",
        "## Nominal run",
        "",
        f"- Ride time: {results['time'][-1]:.3f} s",
        f"- Distance traveled: {results['distance'][-1]:.3f} m",
        f"- Maximum speed: {np.max(results['velocity']):.3f} m/s",
        f"- Normal specific-force range: "
        f"{np.min(signals['normal_g']):.3f} to "
        f"{np.max(signals['normal_g']):.3f} g",
        f"- Maximum absolute longitudinal specific force: "
        f"{np.max(np.abs(signals['longitudinal_g'])):.3f} g",
        f"- Maximum screening-filter vector onset rate: "
        f"{np.max(signals['vector_jerk_gps']):.3f} g/s",
        f"- Applied screening filter window: "
        f"{signals['filter_window_s'][0]:.4f} s",
        "",
        "## Illustrative screening rules",
        "",
        "| Rule | Signal | Comparison | Threshold |",
        "|---|---|---:|---:|",
    ]
    for rule in rules:
        lines.append(
            f"| {rule.name} | `{rule.signal}` | {rule.comparison} | "
            f"{rule.threshold:g} {rule.units} |"
        )

    lines.extend(
        [
            "",
            "## Detected review intervals",
            "",
            "These are review flags, not passes or failures.",
            "",
        ]
    )
    if events:
        lines.extend(
            [
                "| Rule | Start [s] | Duration [s] | Peak | Distance [m] |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for event in events:
            lines.append(
                f"| {event.rule} | {event.start_time_s:.3f} | "
                f"{event.duration_s:.3f} | {event.peak_value:.3f} "
                f"{event.units} | {event.start_distance_m:.2f}–"
                f"{event.end_distance_m:.2f} |"
            )
    else:
        lines.append("No intervals were detected for the configured example rules.")

    lines.extend(
        [
            "",
            "## Bounded sensitivity study",
            "",
            f"- Seed: {seed}",
            f"- Completed cases: {len(complete)} of {len(sweep)}",
        ]
    )
    if complete:
        lines.extend(
            [
                f"- Maximum-speed range: {np.min(max_speed):.3f} to "
                f"{np.max(max_speed):.3f} m/s",
                f"- Minimum normal-g range: {np.min(min_normal):.3f} to "
                f"{np.max(min_normal):.3f} g",
                f"- Maximum normal-g range: {np.min(max_normal):.3f} to "
                f"{np.max(max_normal):.3f} g",
            ]
        )

    lines.extend(
        [
            "",
            "## Required before a formal compliance claim",
            "",
            "- Licensed current ASTM requirements and a clause-by-clause matrix",
            "- Actual 3D track, banking, train, bogie, wheel, restraint, and rider data",
            "- Validated multi-body and structural/fatigue load models",
            "- Normal, emergency, degraded, and reasonably foreseeable fault cases",
            "- Safety-related control-system and braking analyses",
            "- Manufacturing QA, inspection, maintenance, and operations documentation",
            "- Calibrated physical testing and model correlation",
            "- Review and acceptance by qualified engineers and the authority having jurisdiction",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def load_rules(path: Path | None) -> tuple[ScreeningRule, ...]:
    """Load user-supplied screening rules or return documented examples."""

    if path is None:
        return DEFAULT_RULES
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(ScreeningRule(**item) for item in payload["rules"])


def run_assessment(
    output_directory: Path,
    cases: int = 40,
    seed: int = 2291,
    rules_path: Path | None = None,
) -> None:
    """Run the complete preliminary assessment and write its artifacts."""

    output_directory.mkdir(parents=True, exist_ok=True)
    rules = load_rules(rules_path)
    results = simulate(track_profile(), Vehicle())
    signals = rider_frame_signals(results)
    events = detect_events(
        results["time"],
        results["distance"],
        signals,
        rules,
    )
    sweep = run_uncertainty_sweep(cases, seed)

    export_time_history(
        output_directory / "nominal_time_history.csv",
        results,
        signals,
    )
    _write_dict_rows(
        output_directory / "screening_events.csv",
        [asdict(event) for event in events],
    )
    _write_dict_rows(
        output_directory / "uncertainty_sweep.csv",
        sweep,
    )
    plot_assessment(
        output_directory / "preliminary_assessment.png",
        results,
        signals,
        sweep,
    )
    write_report(
        output_directory / "PRELIMINARY_REPORT.md",
        results,
        signals,
        events,
        sweep,
        rules,
        seed,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("assessment_output"),
    )
    parser.add_argument(
        "--cases",
        type=int,
        default=40,
        help="number of bounded uncertainty cases (default: 40)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2291,
        help="random seed for repeatability (default: 2291)",
    )
    parser.add_argument(
        "--rules",
        type=Path,
        help="optional JSON file containing project-specific screening rules",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_assessment(
        args.output_directory,
        cases=args.cases,
        seed=args.seed,
        rules_path=args.rules,
    )
    print(
        "Preliminary engineering package written to "
        f"{args.output_directory.resolve()}"
    )
    print("Status: NOT AN ASTM COMPLIANCE DETERMINATION")


if __name__ == "__main__":
    main()
