# COORDINATES.MD
Below is a “map” of the angles that appear in MARS and the exact transformations that the code performs.  If you duplicate these steps outside of MARS (e.g. in Python for a deep-learning pipeline) your numbers will agree event-by-event with the official reconstruction.
────────────────────────────────────────────────────────────────────────

1. Angle conventions that MStarCamTrans expects
────────────────────────────────────────────────────────────────────────
Local sky (horizontal system)
- θ  ≡  Zenith-distance [deg] 0° = straight up, 90° = horizon
- φ  ≡  Azimuth [deg] 0° = North, 90° = East, 180° = South, 270° = West
    (positive anti-clockwise when looking at the sky)
Camera plane
- X, Y  ≡  Cartesian coordinates on the focal plane, units mm.
 X points to the RIGHT when you stand behind the dish and look toward the camera
 Y points UP
The numerical projection is a gnomonic projection (tangent plane):
XC = –sin θ · sin(φ-φ₀) / D
YC = (–sin θ₀ · cos θ + cos θ₀ · sin θ · cos(φ-φ₀)) / D
where D = cos θ₀ · cos θ + sin θ₀ · sin θ · cos(φ-φ₀)
and (θ₀, φ₀) is the telescope pointing direction.
Physical coordinates are X = XC·f, Y = YC·f with f = fDistCam (distance camera–mirror, read from `MGeomCam`).
The inverse (camera → sky) uses the routine `Loc0CamToLoc`; analytically it is the inverse gnomonic projection.
────────────────────────────────────────────────────────────────────────
1. Why the code does “180 – φ” for Monte-Carlo
────────────────────────────────────────────────────────────────────────
CORSIKA/Reflector writes azimuth in the **physics convention**
 0° = **South**, increasing **westwards** (clock-wise).
MARS works in the **astronomical convention** above (0° = North, anti-clock-wise).
Changing handedness is exactly a 180° rotation:
φ_astro = 180° – φ_physics              (wrap to 0-360 afterwards)
You see that in `MSrcPosCalc::Process`:

```cpp
fTrans->Loc0LocToCam(telθ*Rad2Deg,
                     180.-telφ*Rad2Deg,      // telescope azimuth
                     θ*Rad2Deg,
                     180.-φ*Rad2Deg,         // event azimuth
                     x,y);
```

So any time you use the Monte-Carlo numbers outside the framework you must
apply this shift to **both** the telescope azimuth and the event azimuth.
────────────────────────────────────────────────────────────────────────
3.  Recipe – MC event  →  camera (X,Y)
────────────────────────────────────────────────────────────────────────
Input from `MMcEvt` (all in radians):
θ              = fTheta
φ              = fPhi          (range –π … π)
telθ           = fTelescopeTheta
telφ           = fTelescopePhi (range 0 … 2π)
Python-style pseudo-code:

```python
import numpy as np
# 1. convert rad → deg
θ     = np.degrees(event.theta)
φ     = np.degrees(event.phi)
telθ  = np.degrees(event.tel_theta)
telφ  = np.degrees(event.tel_phi)
# 2. flip azimuth convention
φ_astro    = (180.0 - φ ) % 360.0
telφ_astro = (180.0 - telφ) % 360.0
# 3. gnomonic projection (same formula as Loc0LocToCam)
Δφ = np.radians(φ_astro - telφ_astro)
θ0 = np.radians(telθ)
θ  = np.radians(θ)
D  = np.cos(θ0)*np.cos(θ) + np.sin(θ0)*np.sin(θ)*np.cos(Δφ)
XC = - np.sin(θ)*np.sin(Δφ) / D
YC = (- np.sin(θ0)*np.cos(θ) + np.cos(θ0)*np.sin(θ)*np.cos(Δφ)) / D
# 4. physical camera coordinates
f  = fDistCam_mm                       # read once from the geometry file
X  = XC * f
Y  = YC * f
```

Use `(X, Y)` (or the dimension-less `(XC, YC)`) as the regression target
for your neural network.
────────────────────────────────────────────────────────────────────────
4.  Recipe – camera (X,Y)  →  predicted arrival (θ,φ)
────────────────────────────────────────────────────────────────────────

1. Obtain telescope pointing (θ₀, φ₀) for that event (real data: from
drive system, MC: the simulated values).
  Remember to convert `φ₀` as above for MC.
2. Normalise: `XC = X/f`, `YC = Y/f`
3. Inverse gnomonic projection:

```python
XC2  = XC*XC
YC2  = YC*YC
den  = np.sqrt(1 + XC2 + YC2)
sinθ = den**-1 * np.sqrt(XC2 + (YC*np.cos(np.radians(θ₀)) -
                                XC*np.sin(np.radians(θ₀)))**2)
θ     = np.arcsin(sinθ)                        # radians
φ     = np.arctan2(-XC,
                   np.sin(np.radians(θ₀)) + YC*np.cos(np.radians(θ₀))) \
        + np.radians(φ₀)
θ_deg = np.degrees(θ)
φ_deg = (np.degrees(φ) + 360) % 360
```

For Monte-Carlo you would convert back to the physics convention for
comparison:

```python
φ_mc_conv = (180.0 - φ_deg)          # now compare with original fPhi*Rad2Deg
```

────────────────────────────────────────────────────────────────────────
5.  Real data path
────────────────────────────────────────────────────────────────────────
`MSrcPosCalc` for real data does two extra steps internally:
(1) Converts RA/Dec of telescope centre → local (θ₀, φ₀) using
`CalcHa()` and `CelToLoc()`
(2) Converts RA/Dec of source → local (θ,φ)
After that it *still* calls `Cel0CelToCam`, which inside ends up in
`Loc0LocToCam`.  Therefore, **once you know (θ₀, φ₀) and (θ,φ) in the
astronomical local system you can reuse exactly the same equations as for
MC.**
────────────────────────────────────────────────────────────────────────
6.  Summary of things to watch
────────────────────────────────────────────────────────────────────────

- Always work in **degrees** when you call the formulae above
- Flip azimuth by `180° – φ` for any value that comes from the MC files
- Use the **same** focal-length (`fDistCam`) that the run used
- Wrap angles to [0,360) after every subtraction/addition to avoid
numerical surprises
- For real data the azimuth already follows the astronomical convention —
do **not** apply the 180° shift there.
If you follow the two small conversion rules (rad→deg and the azimuth
flip for MC) and then use the gnomonic projection you will reproduce the
camera positions used by MARS exactly, and you can train your network on
those.

