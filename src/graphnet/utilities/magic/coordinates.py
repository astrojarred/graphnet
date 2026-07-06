"""MARS-compatible camera / local / celestial coordinate transforms for MAGIC.

This module is a NumPy re-implementation of ``Mars/mstarcam/MStarCamTrans.cc``,
the MARS (MAGIC Analysis and Reconstruction Software) class used to convert
between the three coordinate systems used throughout the MAGIC reconstruction
chain:

- **Celestial**: declination ``dec`` [deg] and hour angle ``ha`` [hours].
  Right ascension is *not* one of the two celestial coordinates used here;
  instead ``RA = LST - HA``, wrapped to ``[0, 24)`` hours, where ``LST`` is
  the local sidereal time at the moment of the observation. MARS hour angles
  and right ascensions are always expressed in **hours**, not degrees.
- **Local** (a.k.a. horizontal): zenith angle ``theta``/``zd`` [deg] and
  azimuth ``phi``/``az`` [deg], with the MARS/TDAS convention
  ``az = 0`` at north and ``az = 90`` at east.
- **Camera**: focal-plane coordinates ``X``, ``Y`` [mm], with ``X`` horizontal
  and ``Y`` pointing up, for a camera centered on some local- or
  celestial-coordinate direction ("camera center", subscript/prefix ``0``).

No transform in this module hard-codes a focal distance or plate scale: the
camera-to-reflector distance (``dist_cam`` / ``camera_dist_mm``) is always an
explicit argument, since it varies between telescopes/epochs and must be
read from data (e.g. exported MARS metadata) rather than assumed.

The projection equations
-------------------------
Given the observatory latitude via ``cos_lat = cos(lat)`` and
``sin_lat = sin(lat)``, celestial coordinates ``(dec, ha)`` are transformed to
a local Cartesian triad via a rotation about the local east-west axis
(``a1 = cos_lat``, ``a3 = -sin_lat``):

.. code-block:: text

    xB = cos(dec) * cos(h),  yB = cos(dec) * sin(h),  zB = -sin(dec)
    xA = a3 * xB - a1 * zB
    yA = -yB
    zA = -a1 * xB - a3 * zB
    theta = arccos(-zA),  phi = atan2(yA, xA)

and the inverse rotation recovers ``(dec, ha)`` from ``(theta, phi)``. The
camera projection (:func:`loc0_loc_to_cam` / :func:`loc0_cam_to_loc`) is a
gnomonic (tangent-plane) projection of the local direction ``(theta, phi)``
about a camera-center direction ``(theta0, phi0)``:

.. code-block:: text

    XC = -sin(theta) * sin(phi - phi0) / D
    YC = (-sin(theta0) * cos(theta)
          + cos(theta0) * sin(theta) * cos(phi - phi0)) / D
    D  = cos(theta0) * cos(theta) + sin(theta0) * sin(theta) * cos(phi - phi0)
    X = XC * dist_cam,  Y = YC * dist_cam

For Monte Carlo (MC) simulated showers exported by ``MExportParquet``, the
MARS local azimuth is recovered from the CORSIKA polar angle ``phi`` [rad] via
``AZ_mc = 180 - phi_rad * 180 / pi`` (see :func:`mc_phi_rad_to_mars_az_deg`).

All functions accept NumPy arrays *or* Python scalars and return NumPy
arrays/scalars of matching shape (0-d arrays are unwrapped back to Python
floats so that scalar-in, scalar-out holds).
"""

from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import NDArray

FloatOrArray = Union[float, NDArray[np.floating]]

# Constants matching the C++ code (``TMath::Pi()`` / ``TMath::RadToDeg()``).
RAD2DEG: float = 180.0 / np.pi
TWOPI: float = 2.0 * np.pi

#: Tolerance used by the C++-parity fixture test for camera coordinates [mm].
PARITY_ATOL_MM: float = 1e-3
#: Tolerance used by the C++-parity fixture test for angular quantities [deg].
PARITY_ATOL_DEG: float = 1e-9

__all__ = [
    "RAD2DEG",
    "TWOPI",
    "PARITY_ATOL_MM",
    "PARITY_ATOL_DEG",
    "FloatOrArray",
    "normalize_hour_angle",
    "wrap24",
    "wrap360",
    "cel_to_loc",
    "loc_to_cel",
    "loc0_loc_to_cam",
    "loc0_cam_to_loc",
    "cel0_cam_to_cel",
    "cel0_cel_to_cam",
    "hour_angle_hours_from_lst_and_ra",
    "ra_hours_from_lst_and_ha",
    "angular_separation_deg",
    "camera_prediction_to_radec",
    "mc_phi_rad_to_mars_az_deg",
    "mc_theta_rad_to_zd_deg",
    "StarCamTrans",
]


def _match_scalar(
    result: NDArray[np.floating], *inputs: FloatOrArray
) -> FloatOrArray:
    """Return a Python float if all ``inputs`` were scalars, else the array."""
    if all(np.ndim(x) == 0 for x in inputs):
        return result.item()
    return result


def wrap24(hours: FloatOrArray) -> FloatOrArray:
    """Wrap an hour-angle-like quantity to the half-open interval ``[0, 24)``.

    Args:
        hours: Value(s) in hours, any sign/magnitude.

    Returns:
        Value(s) wrapped to ``[0, 24)`` hours, same shape/type as ``hours``.
    """
    arr = np.asarray(hours, dtype=float)
    wrapped = arr % 24.0
    wrapped = np.where(wrapped < 0.0, wrapped + 24.0, wrapped)
    return _match_scalar(wrapped, hours)


def wrap360(deg: FloatOrArray) -> FloatOrArray:
    """Wrap a degree-like quantity to the half-open interval ``[0, 360)``.

    Args:
        deg: Value(s) in degrees, any sign/magnitude.

    Returns:
        Value(s) wrapped to ``[0, 360)`` degrees, same shape/type as ``deg``.
    """
    arr = np.asarray(deg, dtype=float)
    wrapped = arr % 360.0
    wrapped = np.where(wrapped < 0.0, wrapped + 360.0, wrapped)
    return _match_scalar(wrapped, deg)


def normalize_hour_angle(hhour: FloatOrArray) -> FloatOrArray:
    """Normalize an hour angle (or RA) to the MARS convention ``[0, 24)`` hours.

    This matches ``MStarCamTrans.cc``'s internal normalization, which is
    applied to hour angles (and, by the same convention, to right ascensions
    derived as ``RA = LST - HA``) rather than the astronomical ``[-12, 12)``
    convention sometimes used elsewhere.

    Args:
        hhour: Hour angle in hours (can be negative or >= 24).

    Returns:
        Hour angle normalized to ``[0, 24)`` hours.
    """
    return wrap24(hhour)


def cel_to_loc(
    decdeg: FloatOrArray,
    hhour: FloatOrArray,
    cos_lat: float,
    sin_lat: float,
) -> tuple[FloatOrArray, FloatOrArray]:
    """Convert celestial ``(dec, ha)`` to local ``(zenith angle, azimuth)``.

    Args:
        decdeg: Declination in degrees.
        hhour: Hour angle in hours.
        cos_lat: Cosine of the observatory latitude.
        sin_lat: Sine of the observatory latitude.

    Returns:
        Tuple ``(thetadeg, phideg)``: zenith angle and azimuth, both in
        degrees (``az = 0`` at north, ``az = 90`` at east).
    """
    dec_in = np.asarray(decdeg, dtype=float)
    h_in = np.asarray(hhour, dtype=float)

    a1 = cos_lat
    a3 = -sin_lat

    dec = np.deg2rad(dec_in)
    h = h_in / 24.0 * TWOPI

    xB = np.cos(dec) * np.cos(h)
    yB = np.cos(dec) * np.sin(h)
    zB = -np.sin(dec)

    xA = a3 * xB - a1 * zB
    yA = -yB
    zA = -a1 * xB - a3 * zB

    thetadeg = np.arccos(-zA) * RAD2DEG
    phideg = np.arctan2(yA, xA) * RAD2DEG

    return _match_scalar(thetadeg, decdeg, hhour), _match_scalar(
        phideg, decdeg, hhour
    )


def loc_to_cel(
    thetadeg: FloatOrArray,
    phideg: FloatOrArray,
    cos_lat: float,
    sin_lat: float,
) -> tuple[FloatOrArray, FloatOrArray]:
    """Convert local ``(zenith angle, azimuth)`` to celestial ``(dec, ha)``.

    Args:
        thetadeg: Zenith angle in degrees.
        phideg: Azimuth angle in degrees (``0`` = north, ``90`` = east).
        cos_lat: Cosine of the observatory latitude.
        sin_lat: Sine of the observatory latitude.

    Returns:
        Tuple ``(decdeg, hhour)``: declination in degrees and hour angle in
        hours, wrapped to ``[0, 24)``.
    """
    theta_in = np.asarray(thetadeg, dtype=float)
    phi_in = np.asarray(phideg, dtype=float)

    a1 = cos_lat
    a3 = -sin_lat

    theta = np.deg2rad(theta_in)
    phi = np.deg2rad(phi_in)

    xA = np.sin(theta) * np.cos(phi)
    yA = np.sin(theta) * np.sin(phi)
    zA = -np.cos(theta)

    xB = a3 * xA - a1 * zA
    yB = -yA
    zB = -a1 * xA - a3 * zA

    dec = np.arcsin(-zB)
    h = np.arctan2(yB, xB)

    decdeg = dec * RAD2DEG
    hhour = normalize_hour_angle(h * 24.0 / TWOPI)
    hhour = np.asarray(hhour, dtype=float)

    return _match_scalar(decdeg, thetadeg, phideg), _match_scalar(
        hhour, thetadeg, phideg
    )


def loc0_cam_to_loc(
    theta0deg: FloatOrArray,
    phi0deg: FloatOrArray,
    X: FloatOrArray,
    Y: FloatOrArray,
    dist_cam: float,
) -> tuple[FloatOrArray, FloatOrArray]:
    """Convert camera coordinates to local coordinates about a camera center.

    Args:
        theta0deg: Zenith angle of the camera center, in degrees.
        phi0deg: Azimuth angle of the camera center, in degrees.
        X: Camera X coordinate in mm (horizontal).
        Y: Camera Y coordinate in mm (up).
        dist_cam: Camera-to-reflector distance in mm. Always pass this
            explicitly from telescope/epoch metadata; there is no default.

    Returns:
        Tuple ``(thetadeg, phideg)`` in degrees.
    """
    inputs = (theta0deg, phi0deg, X, Y)
    theta0deg_a = np.asarray(theta0deg, dtype=float)
    phi0deg_a = np.asarray(phi0deg, dtype=float)
    X_a = np.asarray(X, dtype=float)
    Y_a = np.asarray(Y, dtype=float)

    theta0 = np.deg2rad(theta0deg_a)
    phi0 = np.deg2rad(phi0deg_a)

    XC = X_a / dist_cam
    YC = Y_a / dist_cam

    sip = -XC
    cop = np.sin(theta0) + YC * np.cos(theta0)

    sit = np.sqrt(cop * cop + XC * XC)
    cot = np.cos(theta0) - YC * np.sin(theta0)

    # Handle the sign ambiguity of cos(theta): if theta0 is far from 90 deg,
    # pick theta closest to theta0; otherwise pick theta compatible with a
    # small |phi - phi0| (mirrors MStarCamTrans.cc exactly).
    mask = np.abs(theta0deg_a - 90.0) > 45.0

    cot_sign_far = np.abs(cot) * np.sign(np.cos(theta0))

    cop_div_cot = np.where(
        np.abs(cot) > 1e-10, cop / np.where(cot == 0, 1.0, cot), 0.0
    )
    cot_sign_close = np.abs(cot) * np.sign(cop_div_cot)

    cot_sign = np.where(mask, cot_sign_far, cot_sign_close)
    theta = np.arctan2(sit, cot_sign)

    sig = np.sign(cot * np.tan(theta))
    phiminphi0 = np.arctan2(sig * sip, sig * cop)

    phideg = (phi0 + phiminphi0) * RAD2DEG
    thetadeg = theta * RAD2DEG

    return _match_scalar(thetadeg, *inputs), _match_scalar(phideg, *inputs)


def loc0_loc_to_cam(
    theta0deg: FloatOrArray,
    phi0deg: FloatOrArray,
    thetadeg: FloatOrArray,
    phideg: FloatOrArray,
    dist_cam: float,
) -> tuple[FloatOrArray, FloatOrArray]:
    """Convert local coordinates to camera coordinates about a camera center.

    Args:
        theta0deg: Zenith angle of the camera center, in degrees.
        phi0deg: Azimuth angle of the camera center, in degrees.
        thetadeg: Zenith angle in degrees.
        phideg: Azimuth angle in degrees.
        dist_cam: Camera-to-reflector distance in mm. Always pass this
            explicitly from telescope/epoch metadata; there is no default.

    Returns:
        Tuple ``(X, Y)`` camera coordinates in mm.
    """
    inputs = (theta0deg, phi0deg, thetadeg, phideg)
    theta0deg_a = np.asarray(theta0deg, dtype=float)
    phi0deg_a = np.asarray(phi0deg, dtype=float)
    thetadeg_a = np.asarray(thetadeg, dtype=float)
    phideg_a = np.asarray(phideg, dtype=float)

    sintheta0 = np.sin(np.deg2rad(theta0deg_a))
    costheta0 = np.cos(np.deg2rad(theta0deg_a))
    phi0 = np.deg2rad(phi0deg_a)

    sintheta = np.sin(np.deg2rad(thetadeg_a))
    costheta = np.cos(np.deg2rad(thetadeg_a))
    phi = np.deg2rad(phideg_a)

    phi_diff = phi - phi0

    denom = costheta0 * costheta + sintheta0 * sintheta * np.cos(phi_diff)

    XC = -sintheta * np.sin(phi_diff) / denom
    YC = (
        -sintheta0 * costheta + costheta0 * sintheta * np.cos(phi_diff)
    ) / denom

    X = XC * dist_cam
    Y = YC * dist_cam

    return _match_scalar(X, *inputs), _match_scalar(Y, *inputs)


def cel0_cam_to_cel(
    dec0deg: FloatOrArray,
    h0hour: FloatOrArray,
    X: FloatOrArray,
    Y: FloatOrArray,
    dist_cam: float,
    cos_lat: float,
    sin_lat: float,
) -> tuple[FloatOrArray, FloatOrArray]:
    """Convert camera coordinates to celestial ``(dec, ha)`` via the camera center.

    Chains :func:`cel_to_loc` (camera center) -> :func:`loc0_cam_to_loc`
    (camera point) -> :func:`loc_to_cel`.

    Args:
        dec0deg: Declination of the camera center, in degrees.
        h0hour: Hour angle of the camera center, in hours.
        X: Camera X coordinate in mm.
        Y: Camera Y coordinate in mm.
        dist_cam: Camera-to-reflector distance in mm (explicit, no default).
        cos_lat: Cosine of the observatory latitude.
        sin_lat: Sine of the observatory latitude.

    Returns:
        Tuple ``(decdeg, hhour)``.
    """
    theta0deg, phi0deg = cel_to_loc(dec0deg, h0hour, cos_lat, sin_lat)
    thetadeg, phideg = loc0_cam_to_loc(theta0deg, phi0deg, X, Y, dist_cam)
    decdeg, hhour = loc_to_cel(thetadeg, phideg, cos_lat, sin_lat)
    return decdeg, hhour


def cel0_cel_to_cam(
    dec0deg: FloatOrArray,
    h0hour: FloatOrArray,
    decdeg: FloatOrArray,
    hhour: FloatOrArray,
    dist_cam: float,
    cos_lat: float,
    sin_lat: float,
) -> tuple[FloatOrArray, FloatOrArray]:
    """Convert celestial ``(dec, ha)`` to camera coordinates via the camera center.

    Chains :func:`cel_to_loc` (camera center) -> :func:`cel_to_loc` (target
    point) -> :func:`loc0_loc_to_cam`.

    Args:
        dec0deg: Declination of the camera center, in degrees.
        h0hour: Hour angle of the camera center, in hours.
        decdeg: Declination of the target point, in degrees.
        hhour: Hour angle of the target point, in hours.
        dist_cam: Camera-to-reflector distance in mm (explicit, no default).
        cos_lat: Cosine of the observatory latitude.
        sin_lat: Sine of the observatory latitude.

    Returns:
        Tuple ``(X, Y)`` camera coordinates in mm.
    """
    theta0deg, phi0deg = cel_to_loc(dec0deg, h0hour, cos_lat, sin_lat)
    thetadeg, phideg = cel_to_loc(decdeg, hhour, cos_lat, sin_lat)
    X, Y = loc0_loc_to_cam(theta0deg, phi0deg, thetadeg, phideg, dist_cam)
    return X, Y


# --- Right ascension / hour angle / sidereal time helpers -----------------


def ra_hours_from_lst_and_ha(
    lst_hours: FloatOrArray, ha_hours: FloatOrArray
) -> FloatOrArray:
    """Right ascension ``RA = LST - HA`` [hours], wrapped to ``[0, 24)``.

    Args:
        lst_hours: Local sidereal time, in hours.
        ha_hours: Hour angle, in hours (MARS convention, ``[0, 24)``).

    Returns:
        Right ascension in hours, wrapped to ``[0, 24)``.
    """
    lst = np.asarray(lst_hours, dtype=float)
    ha = np.asarray(ha_hours, dtype=float)
    ra = normalize_hour_angle(lst - ha)
    return _match_scalar(np.asarray(ra, dtype=float), lst_hours, ha_hours)


def hour_angle_hours_from_lst_and_ra(
    lst_hours: FloatOrArray, ra_hours: FloatOrArray
) -> FloatOrArray:
    """Hour angle ``HA = LST - RA`` [hours], wrapped to ``[0, 24)``.

    Inverse of :func:`ra_hours_from_lst_and_ha`.

    Args:
        lst_hours: Local sidereal time, in hours.
        ra_hours: Right ascension, in hours.

    Returns:
        Hour angle in hours, wrapped to ``[0, 24)``.
    """
    lst = np.asarray(lst_hours, dtype=float)
    ra = np.asarray(ra_hours, dtype=float)
    ha = normalize_hour_angle(lst - ra)
    return _match_scalar(np.asarray(ha, dtype=float), lst_hours, ra_hours)


def mc_theta_rad_to_zd_deg(theta_rad: FloatOrArray) -> FloatOrArray:
    """Zenith distance [deg] from a CORSIKA/MMcEvt polar angle ``theta`` [rad].

    Args:
        theta_rad: MC polar (zenith) angle in radians.

    Returns:
        Zenith distance in degrees.
    """
    arr = np.rad2deg(np.asarray(theta_rad, dtype=float))
    return _match_scalar(arr, theta_rad)


def mc_phi_rad_to_mars_az_deg(phi_rad: FloatOrArray) -> FloatOrArray:
    """MARS local azimuth [deg] from a CORSIKA/MMcEvt azimuth ``phi`` [rad].

    Implements the remap used by ``MExportParquet``:
    ``AZ_mc = 180 - phi_rad * 180 / pi``.

    Args:
        phi_rad: MC azimuth angle in radians.

    Returns:
        MARS-convention local azimuth in degrees (not wrapped; apply
        :func:`wrap360` if a ``[0, 360)`` range is required downstream).
    """
    arr = 180.0 - np.rad2deg(np.asarray(phi_rad, dtype=float))
    return _match_scalar(arr, phi_rad)


# --- Angular separation -----------------------------------------------


def angular_separation_deg(
    ra1: FloatOrArray,
    dec1_deg: FloatOrArray,
    ra2: FloatOrArray,
    dec2_deg: FloatOrArray,
    ra_unit: str = "deg",
) -> FloatOrArray:
    """Great-circle angular separation between two points, in degrees.

    Uses the numerically stable Vincenty formula (equivalent to but more
    robust than the haversine formula for both small and near-antipodal
    separations):

    .. code-block:: text

        d_ra = ra2 - ra1
        num = sqrt((cos(dec2) sin(d_ra))^2
                    + (cos(dec1) sin(dec2) - sin(dec1) cos(dec2) cos(d_ra))^2)
        den = sin(dec1) sin(dec2) + cos(dec1) cos(dec2) cos(d_ra)
        separation = atan2(num, den)

    Args:
        ra1: Right ascension (or hour angle) of point 1. Unit given by
            ``ra_unit``. **Note**: MARS right ascensions/hour angles are
            conventionally expressed in *hours*, not degrees -- pass
            ``ra_unit="hours"`` in that case.
        dec1_deg: Declination of point 1, in degrees (always degrees,
            regardless of ``ra_unit``).
        ra2: Right ascension (or hour angle) of point 2, same unit as ``ra1``.
        dec2_deg: Declination of point 2, in degrees.
        ra_unit: Either ``"deg"`` (default) or ``"hours"``, describing the
            unit of ``ra1``/``ra2``.

    Returns:
        Angular separation in degrees, always >= 0.

    Raises:
        ValueError: If ``ra_unit`` is not ``"deg"`` or ``"hours"``.
    """
    if ra_unit == "hours":
        ra1_deg = np.asarray(ra1, dtype=float) * 15.0
        ra2_deg = np.asarray(ra2, dtype=float) * 15.0
    elif ra_unit == "deg":
        ra1_deg = np.asarray(ra1, dtype=float)
        ra2_deg = np.asarray(ra2, dtype=float)
    else:
        raise ValueError(f"ra_unit must be 'deg' or 'hours', got {ra_unit!r}")

    dec1 = np.deg2rad(np.asarray(dec1_deg, dtype=float))
    dec2 = np.deg2rad(np.asarray(dec2_deg, dtype=float))
    d_ra = np.deg2rad(ra2_deg - ra1_deg)

    num = np.sqrt(
        (np.cos(dec2) * np.sin(d_ra)) ** 2
        + (
            np.cos(dec1) * np.sin(dec2)
            - np.sin(dec1) * np.cos(dec2) * np.cos(d_ra)
        )
        ** 2
    )
    den = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(
        d_ra
    )

    sep_deg = np.rad2deg(np.arctan2(num, den))
    return _match_scalar(sep_deg, ra1, dec1_deg, ra2, dec2_deg)


# --- Camera prediction to RA/Dec (explicit exact / approximate modes) -----


def camera_prediction_to_radec(
    x_cam_mm: FloatOrArray,
    y_cam_mm: FloatOrArray,
    *,
    camera_dist_mm: float,
    cos_lat: float,
    sin_lat: float,
    lst_hours: FloatOrArray | None = None,
    corr_dec_deg: FloatOrArray | None = None,
    corr_ha_hours: FloatOrArray | None = None,
    mode: str = "exact",
    nominal_dec_deg: FloatOrArray | None = None,
    nominal_ha_hours: FloatOrArray | None = None,
) -> tuple[FloatOrArray, FloatOrArray]:
    """Convert a camera-plane prediction ``(X, Y)`` to RA [hours] / Dec [deg].

    This is the "last mile" of the reconstruction chain: given a camera-frame
    position (e.g. an event's reconstructed or true source position, in mm),
    recover celestial coordinates. Two modes are supported, and the caller
    must opt into which is used -- **there is no silent fallback**:

    - ``mode="exact"`` (default): requires the telescope's *corrected*
      pointing direction at the event time (``corr_dec_deg``,
      ``corr_ha_hours`` -- i.e. after pointing-model / active-mirror-control
      corrections, not the commanded "nominal"/drive values) and the event's
      local sidereal time (``lst_hours``), together with the camera's
      exported ``camera_dist_mm``. This is the numerically correct
      transform and matches what MARS does when producing ``src_cam`` /
      ``src_x``/``src_y`` for real data. If any of ``corr_dec_deg``,
      ``corr_ha_hours``, or ``lst_hours`` is missing, this function raises
      ``ValueError`` rather than silently substituting a nominal/drive
      pointing.
    - ``mode="approx_nominal"``: an explicitly-requested **approximate**
      fallback that uses the nominal/commanded pointing
      (``nominal_dec_deg``, ``nominal_ha_hours``) instead of the corrected
      pointing. This ignores pointing-model corrections and is only
      appropriate when corrected pointing is unavailable (e.g. quick-look
      MC studies where nominal == true pointing). It must be requested
      explicitly; it is never used as a silent substitute for
      ``mode="exact"``.

    Args:
        x_cam_mm: Camera X coordinate(s) in mm.
        y_cam_mm: Camera Y coordinate(s) in mm.
        camera_dist_mm: Camera-to-reflector distance in mm, from telescope
            metadata (no default is ever assumed).
        cos_lat: Cosine of the observatory latitude.
        sin_lat: Sine of the observatory latitude.
        lst_hours: Local sidereal time in hours at the event time. Required
            for ``mode="exact"``.
        corr_dec_deg: Corrected pointing declination of the camera center, in
            degrees. Required for ``mode="exact"``.
        corr_ha_hours: Corrected pointing hour angle of the camera center, in
            hours. Required for ``mode="exact"``.
        mode: ``"exact"`` (default) or ``"approx_nominal"``.
        nominal_dec_deg: Nominal/commanded pointing declination, in degrees.
            Required (and only used) for ``mode="approx_nominal"``.
        nominal_ha_hours: Nominal/commanded pointing hour angle, in hours.
            Required (and only used) for ``mode="approx_nominal"``.

    Returns:
        Tuple ``(ra_hours, dec_deg)``.

    Raises:
        ValueError: If ``mode="exact"`` and any of ``corr_dec_deg``,
            ``corr_ha_hours``, ``lst_hours`` is ``None``; if
            ``mode="approx_nominal"`` and either of ``nominal_dec_deg``,
            ``nominal_ha_hours`` is ``None``; or if ``mode`` is neither of
            the two supported values.
    """
    if mode == "exact":
        missing = [
            name
            for name, value in (
                ("corr_dec_deg", corr_dec_deg),
                ("corr_ha_hours", corr_ha_hours),
                ("lst_hours", lst_hours),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "camera_prediction_to_radec(mode='exact') requires corrected "
                "pointing and event LST; missing: " + ", ".join(missing) + ". "
                "Refusing to silently substitute nominal/drive pointing -- "
                "either supply these, or explicitly opt into the documented "
                "approximation via mode='approx_nominal'."
            )
        assert corr_dec_deg is not None and corr_ha_hours is not None
        dec0deg, h0hour = corr_dec_deg, corr_ha_hours
    elif mode == "approx_nominal":
        missing = [
            name
            for name, value in (
                ("nominal_dec_deg", nominal_dec_deg),
                ("nominal_ha_hours", nominal_ha_hours),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "camera_prediction_to_radec(mode='approx_nominal') requires "
                "nominal_dec_deg and nominal_ha_hours; missing: "
                + ", ".join(missing)
            )
        if lst_hours is None:
            raise ValueError(
                "camera_prediction_to_radec(mode='approx_nominal') still "
                "requires lst_hours to compute RA = LST - HA."
            )
        assert nominal_dec_deg is not None and nominal_ha_hours is not None
        dec0deg, h0hour = nominal_dec_deg, nominal_ha_hours
    else:
        raise ValueError(
            f"mode must be 'exact' or 'approx_nominal', got {mode!r}"
        )

    dec_deg, ha_hours = cel0_cam_to_cel(
        dec0deg, h0hour, x_cam_mm, y_cam_mm, camera_dist_mm, cos_lat, sin_lat
    )
    assert lst_hours is not None
    ra_hours = ra_hours_from_lst_and_ha(lst_hours, ha_hours)
    return ra_hours, dec_deg


class StarCamTrans:
    """Convenience wrapper around the module-level transform functions.

    Example:
        >>> trans = StarCamTrans(dist_cam=28000.0, cos_lat=0.876, sin_lat=0.482)
        >>> dec, ha = trans.cel0_cam_to_cel(dec0, ha0, X, Y)
        >>> X, Y = trans.cel0_cel_to_cam(dec0, ha0, dec, ha)
    """

    def __init__(
        self, dist_cam: float, cos_lat: float, sin_lat: float
    ) -> None:
        """Construct a `StarCamTrans`.

        Args:
            dist_cam: Camera-to-reflector distance in mm (explicit, no
                default).
            cos_lat: Cosine of the observatory latitude.
            sin_lat: Sine of the observatory latitude.
        """
        self.dist_cam = dist_cam
        self.cos_lat = cos_lat
        self.sin_lat = sin_lat

    def cel_to_loc(
        self, decdeg: FloatOrArray, hhour: FloatOrArray
    ) -> tuple[FloatOrArray, FloatOrArray]:
        """See :func:`cel_to_loc`."""
        return cel_to_loc(decdeg, hhour, self.cos_lat, self.sin_lat)

    def loc_to_cel(
        self, thetadeg: FloatOrArray, phideg: FloatOrArray
    ) -> tuple[FloatOrArray, FloatOrArray]:
        """See :func:`loc_to_cel`."""
        return loc_to_cel(thetadeg, phideg, self.cos_lat, self.sin_lat)

    def loc0_cam_to_loc(
        self,
        theta0deg: FloatOrArray,
        phi0deg: FloatOrArray,
        X: FloatOrArray,
        Y: FloatOrArray,
    ) -> tuple[FloatOrArray, FloatOrArray]:
        """See :func:`loc0_cam_to_loc`."""
        return loc0_cam_to_loc(theta0deg, phi0deg, X, Y, self.dist_cam)

    def loc0_loc_to_cam(
        self,
        theta0deg: FloatOrArray,
        phi0deg: FloatOrArray,
        thetadeg: FloatOrArray,
        phideg: FloatOrArray,
    ) -> tuple[FloatOrArray, FloatOrArray]:
        """See :func:`loc0_loc_to_cam`."""
        return loc0_loc_to_cam(
            theta0deg, phi0deg, thetadeg, phideg, self.dist_cam
        )

    def cel0_cam_to_cel(
        self,
        dec0deg: FloatOrArray,
        h0hour: FloatOrArray,
        X: FloatOrArray,
        Y: FloatOrArray,
    ) -> tuple[FloatOrArray, FloatOrArray]:
        """See :func:`cel0_cam_to_cel`."""
        return cel0_cam_to_cel(
            dec0deg, h0hour, X, Y, self.dist_cam, self.cos_lat, self.sin_lat
        )

    def cel0_cel_to_cam(
        self,
        dec0deg: FloatOrArray,
        h0hour: FloatOrArray,
        decdeg: FloatOrArray,
        hhour: FloatOrArray,
    ) -> tuple[FloatOrArray, FloatOrArray]:
        """See :func:`cel0_cel_to_cam`."""
        return cel0_cel_to_cam(
            dec0deg,
            h0hour,
            decdeg,
            hhour,
            self.dist_cam,
            self.cos_lat,
            self.sin_lat,
        )
