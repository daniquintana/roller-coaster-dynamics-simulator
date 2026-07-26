#!/usr/bin/env python3
"""Point-mass roller-coaster dynamics on a smooth 2D parametric track.

The track is represented by x(q), z(q), rather than z(x), because a true
vertical loop is not a single-valued function of horizontal position.

Run:
    python roller_coaster_dynamics.py
    python roller_coaster_dynamics.py --no-show --output coaster_results.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.interpolate import CubicSpline

G = 9.81  # gravitational acceleration [m/s^2]


@dataclass(frozen=True)
class Vehicle:
    """Vehicle and loss-model properties in SI units."""

    mass: float = 500.0  # vehicle plus riders [kg]
    rolling_mu: float = 0.006  # dimensionless rolling-resistance coefficient
    drag_area: float = 0.75  # Cd*A, drag coefficient times frontal area [m^2]
    air_density: float = 1.225  # air density at sea level [kg/m^3]


@dataclass(frozen=True)
class Track:
    """Parametric cubic-spline track x(q), z(q)."""

    q: np.ndarray
    x_spline: CubicSpline
    z_spline: CubicSpline

    @property
    def q_end(self) -> float:
        return float(self.q[-1])

    def geometry(
        self, q: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return x, z, slope angle, signed curvature, and metric dq->ds.

        Positive signed curvature bends toward the left-hand normal
        n = (-sin(theta), cos(theta)). At the loop and valley this is the
        rider-support direction; at a hill crest the curvature is negative.
        """

        xp = self.x_spline(q, 1)
        zp = self.z_spline(q, 1)
        xpp = self.x_spline(q, 2)
        zpp = self.z_spline(q, 2)
        metric = np.hypot(xp, zp)  # ds/dq [m per q-unit]
        theta = np.arctan2(zp, xp)  # tangent angle above horizontal [rad]
        curvature = (xp * zpp - zp * xpp) / metric**3  # signed kappa [1/m]
        return self.x_spline(q), self.z_spline(q), theta, curvature, metric


def _clothoid_loop(
    x0: float,
    z0: float,
    bottom_curvature: float = 0.018,
    top_curvature: float = 0.090,
    points: int = 900,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a symmetric vertical loop with linear curvature variation.

    Curvature rises linearly from a low value at the fast bottom to a high
    value at the slow top, then falls symmetrically. Each half turns through
    pi radians, so the full loop turns through exactly 2*pi. This gives a
    larger radius at the bottom and smaller radius at the crown, moderating
    both peak positive g and crown negative g.
    """

    mean_curvature = 0.5 * (bottom_curvature + top_curvature)
    half_length = np.pi / mean_curvature
    total_length = 2.0 * half_length
    s = np.linspace(0.0, total_length, points)
    curvature = np.where(
        s <= half_length,
        bottom_curvature
        + (top_curvature - bottom_curvature) * s / half_length,
        top_curvature
        - (top_curvature - bottom_curvature)
        * (s - half_length)
        / half_length,
    )
    theta = cumulative_trapezoid(curvature, s, initial=0.0)
    x = x0 + cumulative_trapezoid(np.cos(theta), s, initial=0.0)
    z = z0 + cumulative_trapezoid(np.sin(theta), s, initial=0.0)
    return x, z


def track_profile() -> Track:
    """Create a smooth drop, clothoid loop, and camelback hill.

    Returns a parametric cubic spline. The spline parameter q is cumulative
    chord length and is therefore close to, but not assumed to be, arc length.
    """

    # First drop: 62 m to 5 m, with a nonzero initial downhill slope so a
    # vehicle starting exactly from rest accelerates without a launch impulse.
    u_drop = np.linspace(0.0, 1.0, 420)
    x_drop = 70.0 * u_drop
    z_drop = 62.0 - 57.0 * (2.0 * u_drop - u_drop**2)

    # Vertical loop begins and ends horizontally at approximately z = 5 m.
    x_loop, z_loop = _clothoid_loop(x_drop[-1], z_drop[-1])

    # Smooth camelback hill: sin^2 gives horizontal entry and exit tangents.
    u_hill = np.linspace(0.0, 1.0, 500)
    x_hill = x_loop[-1] + 180.0 * u_hill
    z_hill = z_loop[-1] + 20.0 * np.sin(np.pi * u_hill) ** 2

    # Avoid duplicate join points before fitting a C2 cubic spline.
    x_nodes = np.concatenate((x_drop, x_loop[1:], x_hill[1:]))
    z_nodes = np.concatenate((z_drop, z_loop[1:], z_hill[1:]))
    dq = np.hypot(np.diff(x_nodes), np.diff(z_nodes))
    q_nodes = np.concatenate(([0.0], np.cumsum(dq)))

    return Track(
        q=q_nodes,
        x_spline=CubicSpline(q_nodes, x_nodes),
        z_spline=CubicSpline(q_nodes, z_nodes),
    )


def derivatives(
    _time: float, state: np.ndarray, track: Track, vehicle: Vehicle
) -> np.ndarray:
    """ODE right-hand side for state = [q, v].

    Tangential force balance:
        m*dv/dt = -m*g*sin(theta) - mu*m*g*|cos(theta)|
                  - 0.5*rho*(Cd*A)*v^2

    The rolling term is a basic engineering approximation. A detailed wheel
    model would use actual wheel/rail normal load and bearing losses.
    """

    q, velocity = state
    _, _, theta, _, metric = track.geometry(q)

    gravity_tangent = -G * np.sin(theta)  # [m/s^2]
    rolling = vehicle.rolling_mu * G * abs(np.cos(theta))  # [m/s^2]
    aero_drag = (
        0.5
        * vehicle.air_density
        * vehicle.drag_area
        * velocity
        * abs(velocity)
        / vehicle.mass
    )  # [m/s^2], signed with velocity

    acceleration_tangent = gravity_tangent - rolling - aero_drag
    dq_dt = velocity / metric
    return np.array([dq_dt, acceleration_tangent])


def simulate(
    track: Track,
    vehicle: Vehicle,
    max_time: float = 120.0,
    samples: int = 2400,
    max_step: float = 0.03,
) -> dict[str, np.ndarray]:
    """Integrate the coaster motion until it reaches the end of the track.

    ``samples`` and ``max_step`` may be reduced for parameter sweeps. The
    tighter defaults are retained for the nominal engineering time history.
    """

    if samples < 2:
        raise ValueError("samples must be at least 2")
    if max_step <= 0.0:
        raise ValueError("max_step must be positive")

    def reached_end(_t: float, y: np.ndarray, *_args: object) -> float:
        return y[0] - track.q_end

    reached_end.terminal = True  # type: ignore[attr-defined]
    reached_end.direction = 1  # type: ignore[attr-defined]

    # A truly zero initial speed is valid because the initial tangent slopes
    # downhill, producing positive gravitational tangential acceleration.
    solution = solve_ivp(
        derivatives,
        (0.0, max_time),
        y0=np.array([0.0, 0.0]),
        args=(track, vehicle),
        events=reached_end,
        rtol=1.0e-8,
        atol=1.0e-10,
        max_step=max_step,
        dense_output=True,
    )
    if not solution.success:
        raise RuntimeError(f"Integration failed: {solution.message}")
    if not solution.t_events[0].size:
        raise RuntimeError(
            "Vehicle did not reach the track end. Increase max_time or reduce losses."
        )

    time = np.linspace(0.0, solution.t_events[0][0], samples)
    q, velocity = solution.sol(time)
    x, z, theta, curvature, metric = track.geometry(q)

    # Numerically integrate true distance traveled: ds/dt = v.
    distance = cumulative_trapezoid(velocity, time, initial=0.0)
    radius = np.divide(
        1.0,
        np.abs(curvature),
        out=np.full_like(curvature, np.inf),
        where=np.abs(curvature) > 1.0e-9,
    )
    centripetal = velocity**2 / radius  # unsigned magnitude [m/s^2]
    centripetal_signed = velocity**2 * curvature  # along left normal [m/s^2]

    # Signed normal load factor. This is the physically useful version of
    # (g*cos(theta) + v^2/R)/g because signed curvature correctly reduces
    # rider normal load over a convex hill crest.
    g_net = (G * np.cos(theta) + centripetal_signed) / G
    tangential_acceleration = np.array(
        [derivatives(t, np.array([qi, vi]), track, vehicle)[1]
         for t, qi, vi in zip(time, q, velocity)]
    )

    return {
        "time": time,
        "q": q,
        "x": x,
        "z": z,
        "theta": theta,
        "curvature": curvature,
        "radius": radius,
        "velocity": velocity,
        "distance": distance,
        "centripetal": centripetal,
        "tangential_acceleration": tangential_acceleration,
        "g_net": g_net,
        "metric": metric,
    }


def _speed_colored_track(
    axis: plt.Axes, x: np.ndarray, z: np.ndarray, speed: np.ndarray
) -> LineCollection:
    """Add a speed-colored track line to an axis."""

    points = np.column_stack((x, z)).reshape(-1, 1, 2)
    segments = np.concatenate((points[:-1], points[1:]), axis=1)
    collection = LineCollection(
        segments,
        cmap="turbo",
        norm=plt.Normalize(float(speed.min()), float(speed.max())),
        linewidth=4.0,
    )
    collection.set_array(0.5 * (speed[:-1] + speed[1:]))
    axis.add_collection(collection)
    axis.autoscale()
    return collection


def plot_results(
    results: dict[str, np.ndarray],
    output: str | Path = "coaster_results.png",
    show: bool = True,
) -> None:
    """Create track-speed and normal-g plots and save them as one figure."""

    figure, (track_axis, g_axis) = plt.subplots(
        2, 1, figsize=(12, 9), constrained_layout=True
    )

    colored_track = _speed_colored_track(
        track_axis, results["x"], results["z"], results["velocity"]
    )
    colorbar = figure.colorbar(colored_track, ax=track_axis, pad=0.02)
    colorbar.set_label("Vehicle speed [m/s]")
    track_axis.set(
        title="Parametric 2D Track Profile Colored by Vehicle Speed",
        xlabel="Horizontal position, x [m]",
        ylabel="Elevation, z [m]",
        aspect="equal",
    )
    track_axis.grid(alpha=0.25)

    distance = results["distance"]
    g_net = results["g_net"]
    g_axis.plot(distance, g_net, color="#172554", linewidth=1.8, label="Normal load")
    g_axis.axhspan(-1.0, 3.0, color="#22c55e", alpha=0.10)
    g_axis.axhline(3.0, color="#16a34a", linestyle="--", label="3 g guide")
    g_axis.axhline(4.0, color="#f59e0b", linestyle="--", label="4 g caution guide")
    g_axis.axhline(5.0, color="#dc2626", linestyle="--", label="5 g upper guide")
    g_axis.set(
        title="Rider Normal Load Factor (Simplified, Not an ASTM Compliance Envelope)",
        xlabel="Distance traveled along track [m]",
        ylabel="Normal acceleration [g]",
    )
    g_axis.grid(alpha=0.25)
    g_axis.legend(ncol=4, fontsize=9, loc="best")

    figure.savefig(output, dpi=180)
    if show:
        plt.show()
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("coaster_results.png"),
        help="output image path (default: coaster_results.png)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="save the figure without opening an interactive window",
    )
    return parser.parse_args()


def main() -> None:
    """Run the default simulation and print a compact engineering summary."""

    args = _parse_args()
    track = track_profile()
    vehicle = Vehicle()
    results = simulate(track, vehicle)
    plot_results(results, output=args.output, show=not args.no_show)

    print(f"Ride time:          {results['time'][-1]:8.2f} s")
    print(f"Track distance:     {results['distance'][-1]:8.2f} m")
    print(f"Maximum speed:      {results['velocity'].max():8.2f} m/s")
    print(f"Maximum normal load:{results['g_net'].max():8.2f} g")
    print(f"Minimum normal load:{results['g_net'].min():8.2f} g")
    print(f"Figure saved to:    {args.output.resolve()}")


if __name__ == "__main__":
    main()
