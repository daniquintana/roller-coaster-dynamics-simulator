# Roller Coaster Dynamics Simulator

A clean, standalone scientific Python model of a point-mass roller-coaster
vehicle on a smooth 2D track containing:

- an initial gravitational drop;
- a symmetric variable-curvature (clothoid-style) vertical loop; and
- a smooth camelback hill.

The model integrates the vehicle motion with SciPy, including gravity, rolling
resistance, and quadratic aerodynamic drag. It calculates local curvature,
radius of curvature, tangential acceleration, centripetal acceleration, and
the signed rider normal-load factor.

## Why the track is parametric

A vertical loop doubles back in horizontal position, so it cannot be described
by a single-valued function `z(x)`. The code therefore uses the physically
general representation `x(q), z(q)` and fits both coordinates with SciPy
`CubicSpline`. The ODE converts the spline parameter rate to physical speed
using the local metric `ds/dq`.

## Run

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python roller_coaster_dynamics.py
```

For a headless run:

```bash
python roller_coaster_dynamics.py --no-show --output coaster_results.png
```

## Equations

The tangential equation of motion is

```text
m dv/dt = -m g sin(theta)
          - mu m g |cos(theta)|
          - 0.5 rho (Cd A) v |v|
```

with `dq/dt = v / (ds/dq)`. Signed planar curvature is

```text
kappa = (x' z'' - z' x'') / (x'^2 + z'^2)^(3/2)
```

and the signed normal load factor is

```text
n/g = cos(theta) + v^2 kappa / g
```

Using signed curvature is important: it increases normal load in the loop and
valleys while reducing it over a convex hill crest.

## Engineering note

The 3 g, 4 g, and 5 g horizontal lines are visualization guides only. ASTM
ride-design acceleration criteria depend on acceleration direction, duration,
onset rate, restraint/rider configuration, and operating scenario. This
educational point-mass model is not an ASTM F2291 compliance analysis and is
not suitable for certifying a real ride.

## Output

The script saves a two-panel PNG showing:

1. the 2D track colored by vehicle speed; and
2. rider normal load versus distance traveled.

![Example simulation output](coaster_results.png)

## License

MIT
