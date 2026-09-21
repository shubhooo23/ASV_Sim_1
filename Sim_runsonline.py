#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time as wallclock
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 0. Little helpers and constants we use everywhere
# ---------------------------------------------------------------------------

G = 9.81            # gravity, m/s^2
RHO_AIR = 1.225     # air density, kg/m^3
RHO_WATER = 1025.0  # sea water density, kg/m^3 (only informational here)

# Indices into the big state vector - so the rest of the code reads like
# words instead of magic numbers.  (14 states in total.)
IX, IY, IPSI = 0, 1, 2          # position and heading (NED)
IU, IV, IR = 3, 4, 5            # body velocities: surge, sway, yaw rate
ITP, ITS = 6, 7                 # actual thrust of port / starboard thruster
IZ, IZD = 8, 9                  # heave (up is positive) and its rate
IPHI, IPHID = 10, 11            # roll  (starboard down is positive) and rate
ITH, ITHD = 12, 13              # pitch (bow up is positive) and rate
N_STATES = 14


class ConfigError(ValueError):
    """Raised when a parameter is silly (negative mass, zero timestep...)."""


class SimulationError(RuntimeError):
    """Raised when the simulation blows up (NaNs, infinite speeds...)."""


def wrap_pi(angle):
    """Wrap an angle (or numpy array of angles) into [-pi, pi).

    Heading maths goes wrong in *very* annoying ways if you forget this, so
    every angle difference in this file goes through here.
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def clip(value: float, lo: float, hi: float) -> float:
    """Plain-float clip; faster than np.clip for scalars in the hot loop."""
    return lo if value < lo else hi if value > hi else value


def _positive(name: str, value: float, allow_zero: bool = False) -> None:
    """Tiny validation helper so the validate() methods stay readable."""
    if not np.isfinite(value):
        raise ConfigError(f"{name} must be a finite number, got {value!r}")
    if value < 0.0 or (value == 0.0 and not allow_zero):
        cmp = ">= 0" if allow_zero else "> 0"
        raise ConfigError(f"{name} must be {cmp}, got {value!r}")


# ---------------------------------------------------------------------------
# 1. Parameter blocks (dataclasses) - every tunable number lives here
# ---------------------------------------------------------------------------

@dataclass
class VesselParams:
    """Everything physical about the boat."""

    name: str = "Twin-hull ASV 'Kelpie'"

    # --- geometry / mass -------------------------------------------------
    length: float = 2.0          # m, overall length (only used for drawing + wind moment)
    beam: float = 1.0            # m, overall beam   (only used for drawing)
    mass: float = 85.0           # kg
    Iz: float = 35.0             # kg m^2, yaw inertia about the centre of gravity
    x_g: float = 0.0             # m, CG position ahead of the body-frame origin

    # --- added mass (given as POSITIVE numbers, i.e. -X_udot etc.) -------
    added_surge: float = 10.0    # kg
    added_sway: float = 60.0     # kg   (sway added mass is big for slender hulls)
    added_yaw: float = 15.0      # kg m^2
    added_sway_yaw: float = 0.0  # kg m (sway-yaw coupling, usually tiny)

    # --- hydrodynamic damping (linear + quadratic) -----------------------
    Xu: float = 25.0             # N / (m/s)
    Yv: float = 80.0             # N / (m/s)
    Nr: float = 40.0             # N m / (rad/s)
    Xuu: float = 35.0            # N / (m/s)^2
    Yvv: float = 120.0           # N / (m/s)^2   (cross-flow drag)
    Nrr: float = 30.0            # N m / (rad/s)^2

    # --- thrusters (two of them, one per hull) ---------------------------
    thruster_arm: float = 0.40   # m, lateral distance of each thruster from the centreline
    thrust_max: float = 80.0     # N, forward (per thruster) -> ~1.8 m/s top speed in still water
    thrust_min: float = -50.0    # N, reverse (props are usually weaker backwards)
    thruster_tau: float = 0.30   # s, first-order lag between command and real thrust
    thruster_rate_max: float = 200.0  # N/s, slew-rate limit
    power_coeff: float = 1.1     # W per N^1.5  (P ~ T^1.5 is the classic prop rule of thumb)

    # --- wind exposure ---------------------------------------------------
    wind_area_front: float = 0.35    # m^2, projected area seen from ahead
    wind_area_side: float = 0.90     # m^2, projected area seen from the side
    wind_cd_front: float = 0.70
    wind_cd_side: float = 1.10
    wind_cp_offset: float = 0.10     # m, centre of wind pressure ahead of CG (weathercock effect)

    # --- wave forcing (cheap model, not full RAOs) -----------------------
    wave_force_gain: float = 40.0    # N per metre of wave amplitude (first-order oscillating force)
    wave_drift_gain: float = 8.0     # N per m^2 of Hs (steady drift force)
    wave_yaw_gain: float = 15.0      # N m per metre of wave amplitude

    # --- seakeeping layer: heave / roll / pitch as 2nd-order followers ---
    heave_wn: float = 3.9            # rad/s natural frequency
    heave_zeta: float = 0.25         # damping ratio
    heave_gain: float = 0.90         # how much of the wave elevation the hull follows
    roll_wn: float = 3.0
    roll_zeta: float = 0.25
    roll_gain: float = 0.45          # fraction of the wave slope the hull follows
    pitch_wn: float = 3.3
    pitch_zeta: float = 0.30
    pitch_gain: float = 0.50
    heel_gain: float = 0.80          # heel in turns:  phi ~ -heel_gain * u * r / g  (leans outward)

    def validate(self) -> None:
        for n in ("length", "beam", "mass", "Iz", "thruster_arm", "thrust_max",
                  "thruster_tau", "thruster_rate_max"):
            _positive(n, getattr(self, n))
        for n in ("added_surge", "added_sway", "added_yaw", "Xu", "Yv", "Nr", "Xuu",
                  "Yvv", "Nrr", "power_coeff", "wind_area_front", "wind_area_side",
                  "wind_cd_front", "wind_cd_side", "wave_force_gain", "wave_drift_gain",
                  "wave_yaw_gain", "heel_gain", "heave_gain", "roll_gain", "pitch_gain"):
            _positive(n, getattr(self, n), allow_zero=True)
        for n in ("heave_wn", "roll_wn", "pitch_wn"):
            _positive(n, getattr(self, n))
        for n in ("heave_zeta", "roll_zeta", "pitch_zeta"):
            _positive(n, getattr(self, n), allow_zero=True)
        if self.thrust_min > 0.0:
            raise ConfigError("thrust_min should be <= 0 (it's the reverse limit)")
        if self.thrust_max <= self.thrust_min:
            raise ConfigError("thrust_max must be larger than thrust_min")
        # the mass matrix has to be positive definite or physics breaks
        eig = np.linalg.eigvalsh(mass_matrix(self))
        if np.any(eig <= 0.0):
            raise ConfigError("Mass matrix is not positive definite - check mass/added mass/coupling")


@dataclass
class EnvironmentParams:
    """Wind, current and waves.  Set speeds/heights to 0 to switch things off."""

    # Wind: direction is where the wind comes FROM (meteorological convention),
    # degrees clockwise from North.
    wind_speed: float = 5.0            # m/s mean
    wind_dir_from_deg: float = 300.0
    wind_gust_sigma: float = 1.0       # m/s, std-dev of the gusts
    wind_gust_tau: float = 8.0         # s, gust correlation time

    # Current: direction is where the water flows TOWARDS (oceanographic convention).
    current_speed: float = 0.30        # m/s mean
    current_dir_to_deg: float = 90.0
    current_var_sigma: float = 0.05    # m/s, slow variation of the current speed
    current_var_tau: float = 60.0      # s

    # Waves: Bretschneider spectrum from significant height + peak period.
    wave_hs: float = 0.35              # m, significant wave height (0 = flat water)
    wave_tp: float = 3.5               # s, peak period
    wave_dir_from_deg: float = 300.0   # waves come FROM this direction
    wave_spread_deg: float = 25.0      # +/- directional spreading of the components
    wave_components: int = 30          # how many sine waves we superimpose

    def validate(self) -> None:
        for n in ("wind_speed", "wind_gust_sigma", "current_speed", "current_var_sigma",
                  "wave_hs", "wave_spread_deg"):
            _positive(n, getattr(self, n), allow_zero=True)
        for n in ("wind_gust_tau", "current_var_tau", "wave_tp"):
            _positive(n, getattr(self, n))
        if int(self.wave_components) < 3:
            raise ConfigError("wave_components should be at least 3 (20-40 is a good range)")


@dataclass
class SensorParams:
    """Noisy sensors.  Turn `enabled` off for perfect measurements."""

    enabled: bool = True
    gps_rate_hz: float = 5.0
    gps_sigma: float = 0.6             # m, position noise (1-sigma) per axis
    imu_rate_hz: float = 20.0          # compass + gyro rate
    compass_sigma_deg: float = 0.6     # deg
    compass_bias_deg: float = 0.0      # deg, constant heading offset (try 2.0 for fun)
    gyro_sigma_dps: float = 0.3        # deg/s
    gyro_bias_dps: float = 0.10        # deg/s, constant gyro bias
    speed_rate_hz: float = 10.0        # speed log / DVL rate
    speed_sigma: float = 0.03          # m/s

    def validate(self) -> None:
        for n in ("gps_rate_hz", "imu_rate_hz", "speed_rate_hz"):
            _positive(n, getattr(self, n))
        for n in ("gps_sigma", "compass_sigma_deg", "gyro_sigma_dps", "speed_sigma"):
            _positive(n, getattr(self, n), allow_zero=True)


@dataclass
class ControllerParams:
    """Guidance + autopilot settings."""

    # --- guidance --------------------------------------------------------
    cruise_speed: float = 1.2          # m/s
    turn_speed: float = 0.7            # m/s, speed we slow down to when we're pointing the wrong way
    min_speed: float = 0.25            # m/s, floor for the final-approach slowdown
    approach_radius: float = 10.0      # m, start slowing down this far from the final waypoint
    accel_limit: float = 0.30          # m/s^2, how fast the speed reference may change
    lookahead: float = 6.0             # m, LOS lookahead distance
    slip_gain: float = 0.02            # adaptive sideslip estimator gain (rad / (m s))
    acceptance_radius: float = 3.0     # m, waypoint "reached" circle
    capture_growth: float = 0.05       # m/s, how fast the final circle grows while we keep missing it

    # --- heading autopilot (PD with reference model + integral) ----------
    heading_kp: float = 60.0           # N m / rad
    heading_ki: float = 2.0            # N m / (rad s)
    heading_kd: float = 55.0           # N m / (rad/s)
    heading_int_limit: float = 20.0    # N m, anti-windup clamp
    ref_omega: float = 0.8             # rad/s, bandwidth of the heading reference filter
    ref_zeta: float = 1.0              # damping of the reference filter
    ref_rate_max_deg: float = 30.0     # deg/s, max commanded turn rate

    # --- speed autopilot (PI + drag feed-forward) ------------------------
    speed_kp: float = 80.0             # N / (m/s)
    speed_ki: float = 25.0             # N / m
    speed_int_limit: float = 40.0      # N

    def validate(self) -> None:
        for n in ("cruise_speed", "turn_speed", "lookahead", "acceptance_radius",
                  "accel_limit", "ref_omega", "ref_rate_max_deg"):
            _positive(n, getattr(self, n))
        for n in ("min_speed", "approach_radius", "slip_gain", "capture_growth", "heading_kp", "heading_ki",
                  "heading_kd", "heading_int_limit", "ref_zeta", "speed_kp", "speed_ki",
                  "speed_int_limit"):
            _positive(n, getattr(self, n), allow_zero=True)
        if self.turn_speed > self.cruise_speed:
            raise ConfigError("turn_speed should not exceed cruise_speed")


@dataclass
class SimParams:
    """Numerics, timing and initial conditions."""

    dt: float = 0.02                   # s, physics step (RK4, so 0.02 is very comfortable)
    t_end: float = 900.0               # s, hard stop
    control_hz: float = 10.0           # autopilot rate
    log_hz: float = 10.0               # how often we store data
    seed: int = 42                     # random seed -> reproducible noise/waves
    stop_on_finish: bool = True        # end the run shortly after the last waypoint
    finish_hold_s: float = 10.0        # keep simulating this long after arrival (watch it stop)
    x0: float = -4.0                   # m North, initial position (deliberately off the path)
    y0: float = 6.0                    # m East
    psi0_deg: float = 40.0             # deg, initial heading
    u0: float = 0.0                    # m/s initial surge speed

    def validate(self) -> None:
        for n in ("dt", "t_end", "control_hz", "log_hz"):
            _positive(n, getattr(self, n))
        if self.dt > 0.1:
            raise ConfigError("dt > 0.1 s is too coarse for the thruster/heave dynamics; use <= 0.05")
        if 1.0 / self.control_hz < self.dt - 1e-12:
            raise ConfigError("control period is shorter than dt - lower control_hz or dt")
        if 1.0 / self.log_hz < self.dt - 1e-12:
            raise ConfigError("log period is shorter than dt - lower log_hz or dt")
        _positive("finish_hold_s", self.finish_hold_s, allow_zero=True)


# Some ready-made weather.  CLI options override individual values on top of these.
SEA_STATES: Dict[str, Dict[str, float]] = {
    "none":     dict(wind_speed=0.0, wind_gust_sigma=0.0, current_speed=0.0,
                     current_var_sigma=0.0, wave_hs=0.0),
    "calm":     dict(wind_speed=1.5, wind_gust_sigma=0.3, current_speed=0.05,
                     current_var_sigma=0.01, wave_hs=0.05, wave_tp=2.5),
    "moderate": dict(),  # the dataclass defaults
    "rough":    dict(wind_speed=9.0, wind_gust_sigma=2.0, current_speed=0.5,
                     current_var_sigma=0.1, wave_hs=0.8, wave_tp=4.5),
}


# ---------------------------------------------------------------------------
# 2. Environment: wind, current, waves
# ---------------------------------------------------------------------------

class GaussMarkov:
    """First-order Gauss-Markov (Ornstein-Uhlenbeck) process = 'coloured noise'.

    Used for wind gusts and slow current variations.  Stationary std-dev is
    `sigma`, memory time is `tau`.  Way nicer than raw white noise, which would
    make the wind flicker 50 times a second like a broken lightbulb.
    """

    def __init__(self, sigma: float, tau: float, rng: np.random.Generator):
        self.sigma = sigma
        self.tau = tau
        self.rng = rng
        # start from a random point of the stationary distribution (not always 0)
        self.value = float(rng.normal(0.0, sigma)) if sigma > 0.0 else 0.0

    def step(self, dt: float) -> float:
        if self.sigma <= 0.0:
            self.value = 0.0
            return 0.0
        a = math.exp(-dt / self.tau)
        self.value = a * self.value + self.sigma * math.sqrt(1.0 - a * a) * float(self.rng.standard_normal())
        return self.value


class WaveField:
    """Long-crested-ish irregular sea built from a sum of sine waves.

    The amplitudes come from a Bretschneider spectrum (Hs, Tp).  After building
    the components we rescale them so the resulting significant height matches
    the requested Hs exactly (a truncated spectrum would otherwise be a bit off).
    """

    def __init__(self, ep: EnvironmentParams, vp: VesselParams, rng: np.random.Generator):
        n = int(ep.wave_components)
        self.hs = ep.wave_hs
        self.f_gain = vp.wave_force_gain
        self.d_gain = vp.wave_drift_gain
        self.n_gain = vp.wave_yaw_gain

        wp = 2.0 * math.pi / ep.wave_tp
        # spread frequencies from 0.5*wp to 3*wp, and jitter them a little so the
        # sea state doesn't repeat itself exactly after 2*pi/dw seconds
        w = np.linspace(0.5 * wp, 3.0 * wp, n)
        dw = w[1] - w[0]
        w = w + rng.uniform(-0.4, 0.4, n) * dw
        w = np.clip(w, 0.2 * wp, None)

        if self.hs > 0.0:
            S = (5.0 / 16.0) * self.hs ** 2 * wp ** 4 / w ** 5 * np.exp(-1.25 * (wp / w) ** 4)
            a = np.sqrt(2.0 * S * dw)
            # rescale so that 4*sqrt(m0) == Hs exactly
            m0 = 0.5 * float(np.sum(a ** 2))
            a *= self.hs / (4.0 * math.sqrt(m0))
        else:
            a = np.zeros(n)

        self.w = w
        self.k = w ** 2 / G                                   # deep-water dispersion relation
        self.a = a
        self.phase = rng.uniform(0.0, 2.0 * math.pi, n)
        # waves travel TOWARDS (from + 180 deg), each component with a bit of spread
        self.mu0 = math.radians(ep.wave_dir_from_deg) + math.pi
        self.mu = self.mu0 + math.radians(ep.wave_spread_deg) * rng.uniform(-1.0, 1.0, n)
        self.cmu = np.cos(self.mu)
        self.smu = np.sin(self.mu)

    def evaluate(self, x: float, y: float, t: float, psi: float
                 ) -> Tuple[float, float, float, np.ndarray]:
        """Return (elevation, d_eta/dx, d_eta/dy, wave_force_body[Fx, Fy, N])."""
        if self.hs <= 0.0:
            return 0.0, 0.0, 0.0, np.zeros(3)

        arg = self.k * (x * self.cmu + y * self.smu) - self.w * t + self.phase
        s = np.sin(arg)
        eta = float(np.sum(self.a * np.cos(arg)))
        dedx = -float(np.sum(self.a * self.k * self.cmu * s))
        dedy = -float(np.sum(self.a * self.k * self.smu * s))

        # first-order oscillating force pushing along each wave's travel direction
        f_n = self.f_gain * float(np.sum(self.a * s * self.cmu))
        f_e = self.f_gain * float(np.sum(self.a * s * self.smu))
        # plus the steady wave drift (grows with Hs^2, points the way the waves go)
        f_n += self.d_gain * self.hs ** 2 * math.cos(self.mu0)
        f_e += self.d_gain * self.hs ** 2 * math.sin(self.mu0)

        cps, sps = math.cos(psi), math.sin(psi)
        fx = cps * f_n + sps * f_e
        fy = -sps * f_n + cps * f_e
        # yaw moment: waves hitting us from the beam twist the hull, head/following seas don't
        n_yaw = self.n_gain * float(np.sum(self.a * np.sin(arg + 0.8) * np.sin(2.0 * (self.mu - psi))))
        return eta, dedx, dedy, np.array([fx, fy, n_yaw])


class Environment:
    """Bundles wind + current + waves and steps the slow random processes."""

    def __init__(self, ep: EnvironmentParams, vp: VesselParams, rng_seq: List[np.random.Generator]):
        self.ep = ep
        self.gust = GaussMarkov(ep.wind_gust_sigma, ep.wind_gust_tau, rng_seq[0])
        self.cur_var = GaussMarkov(ep.current_var_sigma, ep.current_var_tau, rng_seq[1])
        self.waves = WaveField(ep, vp, rng_seq[2])
        self._wind_to = math.radians(ep.wind_dir_from_deg) + math.pi   # wind blows TOWARDS this
        self._cur_to = math.radians(ep.current_dir_to_deg)

    def step(self, dt: float) -> None:
        self.gust.step(dt)
        self.cur_var.step(dt)

    @property
    def wind_speed(self) -> float:
        return max(0.0, self.ep.wind_speed + self.gust.value)

    @property
    def current_speed(self) -> float:
        return max(0.0, self.ep.current_speed + self.cur_var.value)

    def wind_ned(self) -> Tuple[float, float]:
        s = self.wind_speed
        return s * math.cos(self._wind_to), s * math.sin(self._wind_to)

    def current_ned(self) -> Tuple[float, float]:
        s = self.current_speed
        return s * math.cos(self._cur_to), s * math.sin(self._cur_to)


# ---------------------------------------------------------------------------
# 3. The vessel: equations of motion + RK4 integrator
# ---------------------------------------------------------------------------

def mass_matrix(p: VesselParams) -> np.ndarray:
    """Rigid-body + added-mass matrix (3x3, surge/sway/yaw)."""
    m11 = p.mass + p.added_surge
    m22 = p.mass + p.added_sway
    m23 = p.mass * p.x_g + p.added_sway_yaw
    m33 = p.Iz + p.mass * p.x_g ** 2 + p.added_yaw
    return np.array([[m11, 0.0, 0.0],
                     [0.0, m22, m23],
                     [0.0, m23, m33]])


class ASVDynamics:
    """Turns 'state + commands + environment' into 'state derivative'.

    Notes on what is (and isn't) modelled:
      * Planar motion is the real deal (3-DOF with relative velocity to the current).
      * Heave/roll/pitch just follow the waves through 2nd-order filters and do
        not couple back into surge/sway/yaw.  Good for visualising motion and for
        checking sensor tilt effects, not for hydrodynamic research.
      * The current is treated as constant within one integration step, and we
        neglect its (small) time derivative in the relative-velocity dynamics.
    """

    def __init__(self, vp: VesselParams):
        self.p = vp
        self.M = mass_matrix(vp)
        self.Minv = np.linalg.inv(self.M)

    # -- the big one: right-hand side of all 14 ODEs ------------------------
    def derivatives(self, t: float, s: np.ndarray, cmd: Tuple[float, float],
                    wind_ned: Tuple[float, float], cur_ned: Tuple[float, float],
                    waves: WaveField, want_info: bool = False):
        p = self.p
        psi, u, v, r = s[IPSI], s[IU], s[IV], s[IR]
        cps, sps = math.cos(psi), math.sin(psi)

        ds = np.empty(N_STATES)

        # ---- kinematics: body velocities -> NED position rates -----------
        ds[IX] = u * cps - v * sps
        ds[IY] = u * sps + v * cps
        ds[IPSI] = r

        # ---- current in body axes, then velocity relative to the water ---
        uc = cps * cur_ned[0] + sps * cur_ned[1]
        vc = -sps * cur_ned[0] + cps * cur_ned[1]
        ur, vr = u - uc, v - vc

        # ---- Coriolis + centripetal (uses total mass incl. added mass) ---
        m11, m22 = self.M[0, 0], self.M[1, 1]
        m23 = self.M[1, 2]
        c13 = -(m22 * vr + m23 * r)
        c23 = m11 * ur
        coriolis = np.array([c13 * r, c23 * r, -c13 * ur - c23 * vr])

        # ---- damping: linear + quadratic ---------------------------------
        damping = np.array([
            p.Xu * ur + p.Xuu * abs(ur) * ur,
            p.Yv * vr + p.Yvv * abs(vr) * vr,
            p.Nr * r + p.Nrr * abs(r) * r,
        ])

        # ---- thrusters: forces and yaw moment ----------------------------
        Tp, Ts = s[ITP], s[ITS]
        f_thr = Tp + Ts
        n_thr = p.thruster_arm * (Tp - Ts)   # more thrust on port -> turn to starboard (clockwise)
        tau_thr = np.array([f_thr, 0.0, n_thr])

        # ---- wind: quadratic drag using the air velocity relative to the hull
        w_n, w_e = wind_ned
        wbx = cps * w_n + sps * w_e
        wby = -sps * w_n + cps * w_e
        rx, ry = wbx - u, wby - v
        vrel = math.hypot(rx, ry)
        fwx = 0.5 * RHO_AIR * p.wind_cd_front * p.wind_area_front * vrel * rx
        fwy = 0.5 * RHO_AIR * p.wind_cd_side * p.wind_area_side * vrel * ry
        nw = p.wind_cp_offset * fwy
        tau_wind = np.array([fwx, fwy, nw])

        # ---- waves ---------------------------------------------------------
        eta, dedx, dedy, tau_wave = waves.evaluate(s[IX], s[IY], t, psi)

        # ---- Newton's second law, in body axes -----------------------------
        rhs = tau_thr + tau_wind + tau_wave - coriolis - damping
        nu_dot = self.Minv @ rhs
        ds[IU], ds[IV], ds[IR] = nu_dot

        # ---- thruster dynamics: 1st-order lag with a slew-rate limit -------
        ds[ITP] = clip((cmd[0] - Tp) / p.thruster_tau, -p.thruster_rate_max, p.thruster_rate_max)
        ds[ITS] = clip((cmd[1] - Ts) / p.thruster_tau, -p.thruster_rate_max, p.thruster_rate_max)

        # ---- seakeeping layer: heave / roll / pitch ------------------------
        slope_x = dedx * cps + dedy * sps       # wave slope along the bow direction
        slope_y = -dedx * sps + dedy * cps      # wave slope towards starboard
        z_tgt = p.heave_gain * eta
        th_tgt = p.pitch_gain * math.atan(slope_x)                       # water rising ahead -> bow up
        phi_tgt = (-p.roll_gain * math.atan(slope_y)                      # water rising to stbd -> stbd up
                   - p.heel_gain * u * r / G)                             # centrifugal outward heel

        ds[IZ] = s[IZD]
        ds[IZD] = -2.0 * p.heave_zeta * p.heave_wn * s[IZD] - p.heave_wn ** 2 * (s[IZ] - z_tgt)
        ds[IPHI] = s[IPHID]
        ds[IPHID] = -2.0 * p.roll_zeta * p.roll_wn * s[IPHID] - p.roll_wn ** 2 * (s[IPHI] - phi_tgt)
        ds[ITH] = s[ITHD]
        ds[ITHD] = -2.0 * p.pitch_zeta * p.pitch_wn * s[ITHD] - p.pitch_wn ** 2 * (s[ITH] - th_tgt)

        if want_info:
            info = {"eta": eta, "ur": ur, "vr": vr, "wind_force": tau_wind,
                    "wave_force": tau_wave, "thr_force": tau_thr}
            return ds, info
        return ds

    # -- classic 4th-order Runge-Kutta step ---------------------------------
    def step(self, t: float, s: np.ndarray, cmd: Tuple[float, float],
             wind_ned: Tuple[float, float], cur_ned: Tuple[float, float],
             waves: WaveField, dt: float) -> np.ndarray:
        f = self.derivatives
        k1 = f(t, s, cmd, wind_ned, cur_ned, waves)
        k2 = f(t + 0.5 * dt, s + 0.5 * dt * k1, cmd, wind_ned, cur_ned, waves)
        k3 = f(t + 0.5 * dt, s + 0.5 * dt * k2, cmd, wind_ned, cur_ned, waves)
        k4 = f(t + dt, s + dt * k3, cmd, wind_ned, cur_ned, waves)
        s_new = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        s_new[IPSI] = wrap_pi(s_new[IPSI])

        # If anything went NaN/huge, shout right now instead of plotting garbage later.
        if not np.all(np.isfinite(s_new)) or abs(s_new[IU]) > 50.0 or abs(s_new[IV]) > 50.0:
            raise SimulationError(
                f"State blew up at t={t:.2f}s (u={s_new[IU]:.3g}, v={s_new[IV]:.3g}, r={s_new[IR]:.3g}). "
                "Try a smaller --dt or check your gains/parameters.")
        return s_new


# ---------------------------------------------------------------------------
# 4. Sensors and the position Kalman filter
# ---------------------------------------------------------------------------

@dataclass
class Measurements:
    gps_x: float
    gps_y: float
    gps_fresh: bool     # True only on the step where a brand new GPS fix arrived
    psi: float          # compass heading, rad
    r: float            # gyro yaw rate, rad/s
    u: float            # measured surge speed, m/s


class SensorSuite:
    """GPS + compass + gyro + speed log, each with its own rate and noise.

    Between samples we simply hold the last value (zero-order hold), exactly
    what a real flight-computer sees.
    """

    def __init__(self, sp: SensorParams, rng: np.random.Generator):
        self.sp = sp
        self.rng = rng
        self._next = {"gps": 0.0, "imu": 0.0, "spd": 0.0}
        self._gps = (0.0, 0.0)
        self._psi = 0.0
        self._r = 0.0
        self._u = 0.0

    def sample(self, t: float, s: np.ndarray) -> Measurements:
        sp = self.sp
        if not sp.enabled:
            return Measurements(s[IX], s[IY], True, s[IPSI], s[IR], s[IU])

        eps = 1e-9
        fresh = False
        if t >= self._next["gps"] - eps:
            while t >= self._next["gps"] - eps:
                self._next["gps"] += 1.0 / sp.gps_rate_hz
            self._gps = (s[IX] + self.rng.normal(0.0, sp.gps_sigma),
                         s[IY] + self.rng.normal(0.0, sp.gps_sigma))
            fresh = True
        if t >= self._next["imu"] - eps:
            while t >= self._next["imu"] - eps:
                self._next["imu"] += 1.0 / sp.imu_rate_hz
            self._psi = wrap_pi(s[IPSI] + math.radians(sp.compass_bias_deg)
                                + self.rng.normal(0.0, math.radians(sp.compass_sigma_deg)))
            self._r = (s[IR] + math.radians(sp.gyro_bias_dps)
                       + self.rng.normal(0.0, math.radians(sp.gyro_sigma_dps)))
        if t >= self._next["spd"] - eps:
            while t >= self._next["spd"] - eps:
                self._next["spd"] += 1.0 / sp.speed_rate_hz
            self._u = s[IU] + self.rng.normal(0.0, sp.speed_sigma)
        return Measurements(self._gps[0], self._gps[1], fresh, self._psi, self._r, self._u)


class PositionKalmanFilter:
    """Constant-velocity Kalman filter for the 2-D position.

    State = [x, y, vx, vy].  We predict every control step and correct whenever
    a GPS fix shows up.  It smooths the 0.6 m GPS jitter so the cross-track
    error fed to the guidance isn't a random walk.
    """

    def __init__(self, gps_sigma: float, accel_sigma: float = 0.5):
        self.R = np.eye(2) * max(gps_sigma, 1e-3) ** 2
        self.q = accel_sigma ** 2
        self.x = np.zeros(4)
        self.P = np.eye(4)
        self.ready = False

    def predict(self, dt: float) -> None:
        if not self.ready:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        # process noise from an unknown random acceleration (white-noise accel model)
        q = self.q
        Q = np.zeros((4, 4))
        for i in range(2):
            Q[i, i] = q * dt ** 4 / 4.0
            Q[i, i + 2] = Q[i + 2, i] = q * dt ** 3 / 2.0
            Q[i + 2, i + 2] = q * dt ** 2
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, zx: float, zy: float) -> None:
        z = np.array([zx, zy])
        if not self.ready:                       # first fix just initialises the filter
            self.x = np.array([zx, zy, 0.0, 0.0])
            self.P = np.diag([self.R[0, 0], self.R[1, 1], 1.0, 1.0])
            self.ready = True
            return
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1.0
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P

    @property
    def position(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[1])


# ---------------------------------------------------------------------------
# 5. Guidance + autopilot
# ---------------------------------------------------------------------------

@dataclass
class ControlOutput:
    T_port_cmd: float
    T_stbd_cmd: float
    F_cmd: float
    N_cmd: float
    psi_d: float
    psi_ref: float
    r_ref: float
    u_ref: float
    xte: float
    seg: int
    finished: bool
    saturated: bool


def allocate_thrust(F: float, N: float, arm: float, t_min: float, t_max: float
                    ) -> Tuple[float, float, bool]:
    """Turn a desired surge force F and yaw moment N into two thruster commands.

        F = Tp + Ts          N = arm * (Tp - Ts)

    If the thrusters can't deliver everything we sacrifice surge force first and
    keep the yaw moment (steering beats speed - a boat that can't turn is a
    boat that hits things).
    """
    diff = N / arm                                # = Tp - Ts
    sat = False
    max_diff = t_max - t_min
    if abs(diff) > max_diff:
        diff = math.copysign(max_diff, diff)
        sat = True
    s_lo = 2.0 * t_min + abs(diff)
    s_hi = 2.0 * t_max - abs(diff)
    s_cmd = clip(F, s_lo, s_hi)
    if s_cmd != F:
        sat = True
    return 0.5 * (s_cmd + diff), 0.5 * (s_cmd - diff), sat


class ASVController:
    """Waypoint guidance (adaptive LOS) + heading PD/I + speed PI."""

    def __init__(self, cp: ControllerParams, vp: VesselParams, waypoints: np.ndarray, psi0: float):
        self.cp = cp
        self.vp = vp
        self.wps = np.asarray(waypoints, dtype=float)
        self.seg = 0                         # index of the leg we're on: wps[seg] -> wps[seg+1]
        self.finished = False
        self.finish_time: Optional[float] = None
        self.events: List[Tuple[float, int, float]] = []   # (time, waypoint index reached, distance)

        self.beta_hat = 0.0                  # estimated sideslip/crab angle (rad)
        self.capture_time = 0.0              # seconds spent turning back for a missed final waypoint
        self.psi_ref = psi0                  # filtered heading reference
        self.r_ref = 0.0
        self.h_int = 0.0                     # heading integral (in N m)
        self.s_int = 0.0                     # speed integral (in N)
        self.u_ref = 0.0
        self._sat_prev = False
        self._last_psi_d = psi0
        self._last_xte = 0.0

    # -- geometry of the current leg ---------------------------------------
    def _leg(self, x: float, y: float):
        p0, p1 = self.wps[self.seg], self.wps[self.seg + 1]
        d = p1 - p0
        length = float(np.hypot(d[0], d[1]))
        pi_p = math.atan2(d[1], d[0])                       # path direction
        c, s = math.cos(pi_p), math.sin(pi_p)
        dx, dy = x - p0[0], y - p0[1]
        along = dx * c + dy * s
        cross = -dx * s + dy * c                             # >0 means we're to the right of the path
        dist_end = math.hypot(x - p1[0], y - p1[1])
        return pi_p, along, cross, length, dist_end

    def _guidance(self, t: float, dt: float, x: float, y: float, psi: float
                  ) -> Tuple[float, float, float]:
        """Returns (desired heading, desired speed, cross-track error)."""
        cp = self.cp
        if self.finished:
            return self._last_psi_d, 0.0, self._last_xte

        n = len(self.wps)
        pi_p, along, cross, length, dist_end = self._leg(x, y)
        last_leg = (self.seg == n - 2)

        # Have we reached the end of this leg?  (inside the circle, or - for
        # intermediate corners - already past the perpendicular line)
        # On the last leg the circle slowly grows the longer we've been circling back for it -
        # a boat with no sideways thruster can't always hit a tight circle in a stiff crosswind.
        r_eff = cp.acceptance_radius + (cp.capture_growth * self.capture_time if last_leg else 0.0)
        if dist_end < r_eff or (not last_leg and along > length):
            self.events.append((t, self.seg + 1, dist_end))
            self.seg += 1
            if self.seg >= n - 1:
                self.finished = True
                self.finish_time = t
                return self._last_psi_d, 0.0, self._last_xte
            pi_p, along, cross, length, dist_end = self._leg(x, y)
            last_leg = (self.seg == n - 2)

        if last_leg and along > length:
            # Overshoot!  We flew past the final waypoint without touching its
            # acceptance circle (a strong crosswind can easily do that).  Following
            # the path line any further would just carry us away forever, so
            # ignore the line and head straight back for the waypoint itself
            # (still crab-compensated) until we finally get inside the circle.
            bearing = math.atan2(self.wps[-1][1] - y, self.wps[-1][0] - x)
            psi_d = wrap_pi(bearing - self.beta_hat)
            self.capture_time += dt
        else:
            # Adaptive LOS: aim at a point `lookahead` metres ahead on the path, and
            # learn the crab angle that wind/current force on us (beta_hat).
            delta = cp.lookahead
            self.beta_hat += dt * cp.slip_gain * delta * cross / math.sqrt(delta ** 2 + cross ** 2)
            self.beta_hat = clip(self.beta_hat, -math.radians(30.0), math.radians(30.0))
            psi_d = wrap_pi(pi_p - math.atan2(cross, delta) - self.beta_hat)

        # Speed: cruise, slow down if we're pointing the wrong way, slow down near the end.
        err = abs(wrap_pi(psi_d - psi))
        min_frac = cp.turn_speed / cp.cruise_speed
        u_goal = cp.cruise_speed * clip(1.0 - err / math.radians(70.0), min_frac, 1.0)
        if last_leg and dist_end < cp.approach_radius:
            u_goal = min(u_goal, max(cp.min_speed, cp.cruise_speed * dist_end / cp.approach_radius))

        self._last_psi_d = psi_d
        self._last_xte = cross
        return psi_d, u_goal, cross

    def update(self, t: float, dt: float, x: float, y: float,
               psi: float, r: float, u: float) -> ControlOutput:
        cp, vp = self.cp, self.vp
        psi_d, u_goal, xte = self._guidance(t, dt, x, y, psi)

        # ---- speed reference with an acceleration limit ------------------
        max_step = cp.accel_limit * dt
        self.u_ref += clip(u_goal - self.u_ref, -max_step, max_step)

        # ---- heading reference model (2nd-order filter with rate limit) --
        e_ref = wrap_pi(psi_d - self.psi_ref)
        r_dot = cp.ref_omega ** 2 * e_ref - 2.0 * cp.ref_zeta * cp.ref_omega * self.r_ref
        self.r_ref += dt * r_dot
        r_max = math.radians(cp.ref_rate_max_deg)
        self.r_ref = clip(self.r_ref, -r_max, r_max)
        self.psi_ref = wrap_pi(self.psi_ref + dt * self.r_ref)

        # ---- heading controller: PD + I (derivative acts on measured rate) -
        e_psi = wrap_pi(self.psi_ref - psi)
        if not self._sat_prev and abs(e_psi) < math.radians(30.0):      # anti-windup
            self.h_int = clip(self.h_int + cp.heading_ki * e_psi * dt,
                              -cp.heading_int_limit, cp.heading_int_limit)
        N_cmd = cp.heading_kp * e_psi + self.h_int + cp.heading_kd * (self.r_ref - r)

        # ---- speed controller: drag feed-forward + PI ---------------------
        e_u = self.u_ref - u
        if not self._sat_prev:
            self.s_int = clip(self.s_int + cp.speed_ki * e_u * dt,
                              -cp.speed_int_limit, cp.speed_int_limit)
        F_ff = vp.Xu * self.u_ref + vp.Xuu * abs(self.u_ref) * self.u_ref   # the drag we expect
        F_cmd = F_ff + cp.speed_kp * e_u + self.s_int

        # ---- split into two thruster commands ------------------------------
        Tp, Ts, sat = allocate_thrust(F_cmd, N_cmd, vp.thruster_arm, vp.thrust_min, vp.thrust_max)
        self._sat_prev = sat
        return ControlOutput(Tp, Ts, F_cmd, N_cmd, psi_d, self.psi_ref, self.r_ref,
                             self.u_ref, xte, self.seg, self.finished, sat)


# ---------------------------------------------------------------------------
# 6. The main simulation loop
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    log: Dict[str, np.ndarray]
    events: List[Tuple[float, int, float]]
    metrics: Dict[str, float]
    finished: bool


LOG_KEYS = ["t", "x", "y", "psi", "u", "v", "r", "sog", "cog",
            "Tp", "Ts", "Tp_cmd", "Ts_cmd", "F_cmd", "N_cmd",
            "psi_d", "psi_ref", "r_ref", "u_ref", "xte", "seg",
            "z", "phi", "theta", "eta", "wind", "cur",
            "x_est", "y_est", "gps_x", "gps_y", "sat", "power"]

LOG_UNITS = {"t": "s", "x": "m", "y": "m", "psi": "rad", "u": "m/s", "v": "m/s", "r": "rad/s",
             "sog": "m/s", "cog": "rad", "Tp": "N", "Ts": "N", "Tp_cmd": "N", "Ts_cmd": "N",
             "F_cmd": "N", "N_cmd": "Nm", "psi_d": "rad", "psi_ref": "rad", "r_ref": "rad/s",
             "u_ref": "m/s", "xte": "m", "seg": "-", "z": "m", "phi": "rad", "theta": "rad",
             "eta": "m", "wind": "m/s", "cur": "m/s", "x_est": "m", "y_est": "m",
             "gps_x": "m", "gps_y": "m", "sat": "0/1", "power": "W"}


def run_simulation(vp: VesselParams, ep: EnvironmentParams, sp: SensorParams,
                   cp: ControllerParams, simp: SimParams, waypoints: np.ndarray,
                   verbose: bool = True) -> SimResult:
    """Run one full mission and hand back logs, events and summary metrics."""
    # validate everything up-front: a clear error now beats a mystery NaN later
    for block in (vp, ep, sp, cp, simp):
        block.validate()
    waypoints = validate_waypoints(waypoints)

    # independent random streams so changing (say) sensor noise doesn't reshuffle the waves
    streams = [np.random.default_rng(ss) for ss in np.random.SeedSequence(simp.seed).spawn(4)]
    env = Environment(ep, vp, streams[:3])
    sensors = SensorSuite(sp, streams[3])
    dyn = ASVDynamics(vp)
    kf = PositionKalmanFilter(sp.gps_sigma)
    psi0 = math.radians(simp.psi0_deg)
    ctrl = ASVController(cp, vp, waypoints, psi0)

    # initial state: thrusters idle, sea-keeping states settle on their own
    s = np.zeros(N_STATES)
    s[IX], s[IY], s[IPSI], s[IU] = simp.x0, simp.y0, wrap_pi(psi0), simp.u0

    dt = simp.dt
    control_dt = 1.0 / simp.control_hz
    log_dt = 1.0 / simp.log_hz
    n_steps = int(math.ceil(simp.t_end / dt))

    log: Dict[str, List[float]] = {k: [] for k in LOG_KEYS}
    cmd = (0.0, 0.0)
    out: Optional[ControlOutput] = None
    next_ctrl = 0.0
    next_log = 0.0
    next_print = 0.0
    pending_gps: Optional[Tuple[float, float]] = None
    x_est, y_est = s[IX], s[IY]

    distance = 0.0
    energy_j = 0.0
    n_ctrl = 0
    n_sat = 0
    t = 0.0
    wall0 = wallclock.time()

    if verbose:
        print(f"Simulating '{vp.name}': {len(waypoints)} waypoints, up to {simp.t_end:.0f} s, dt={dt} s")

    for k in range(n_steps + 1):
        t = k * dt
        wind_ned = env.wind_ned()
        cur_ned = env.current_ned()

        # ---- sensors -------------------------------------------------------
        meas = sensors.sample(t, s)
        if meas.gps_fresh:
            pending_gps = (meas.gps_x, meas.gps_y)

        # ---- controller (runs slower than physics) -------------------------
        if out is None or t >= next_ctrl - 1e-9:
            if sp.enabled:
                kf.predict(control_dt)
                if pending_gps is not None:
                    kf.update(*pending_gps)
                    pending_gps = None
                if kf.ready:
                    x_est, y_est = kf.position
            else:
                x_est, y_est = s[IX], s[IY]
            out = ctrl.update(t, control_dt, x_est, y_est, meas.psi, meas.r, meas.u)
            cmd = (out.T_port_cmd, out.T_stbd_cmd)
            next_ctrl += control_dt
            n_ctrl += 1
            n_sat += int(out.saturated)

        # ---- logging ---------------------------------------------------------
        if t >= next_log - 1e-9:
            _, info = dyn.derivatives(t, s, cmd, wind_ned, cur_ned, env.waves, want_info=True)
            power = vp.power_coeff * (abs(s[ITP]) ** 1.5 + abs(s[ITS]) ** 1.5)
            xd = s[IU] * math.cos(s[IPSI]) - s[IV] * math.sin(s[IPSI])
            yd = s[IU] * math.sin(s[IPSI]) + s[IV] * math.cos(s[IPSI])
            row = {
                "t": t, "x": s[IX], "y": s[IY], "psi": s[IPSI], "u": s[IU], "v": s[IV], "r": s[IR],
                "sog": math.hypot(s[IU], s[IV]), "cog": math.atan2(yd, xd),
                "Tp": s[ITP], "Ts": s[ITS], "Tp_cmd": cmd[0], "Ts_cmd": cmd[1],
                "F_cmd": out.F_cmd, "N_cmd": out.N_cmd,
                "psi_d": out.psi_d, "psi_ref": out.psi_ref, "r_ref": out.r_ref,
                "u_ref": out.u_ref, "xte": out.xte, "seg": out.seg,
                "z": s[IZ], "phi": s[IPHI], "theta": s[ITH], "eta": info["eta"],
                "wind": env.wind_speed, "cur": env.current_speed,
                "x_est": x_est, "y_est": y_est, "gps_x": meas.gps_x, "gps_y": meas.gps_y,
                "sat": float(out.saturated), "power": power,
            }
            for key in LOG_KEYS:
                log[key].append(float(row[key]))
            next_log += log_dt

        if verbose and t >= next_print - 1e-9:
            print(f"  t={t:6.1f}s  leg {out.seg + 1}/{len(waypoints) - 1}  "
                  f"xte={out.xte:6.2f} m  u={s[IU]:4.2f} m/s  psi={math.degrees(s[IPSI]):7.1f} deg")
            next_print += 60.0

        # ---- done yet? -----------------------------------------------------
        if (simp.stop_on_finish and ctrl.finished and ctrl.finish_time is not None
                and t >= ctrl.finish_time + simp.finish_hold_s):
            break
        if k == n_steps:
            break

        # ---- accumulate stats using the state at the start of this step -----
        distance += math.hypot(s[IU], s[IV]) * dt
        energy_j += vp.power_coeff * (abs(s[ITP]) ** 1.5 + abs(s[ITS]) ** 1.5) * dt

        # ---- advance the world by one physics step ---------------------------
        env.step(dt)
        s = dyn.step(t, s, cmd, wind_ned, cur_ned, env.waves, dt)

    arrays = {k: np.asarray(v) for k, v in log.items()}
    metrics = compute_metrics(arrays, distance, energy_j, n_sat, n_ctrl, t, ctrl.finished, ctrl.finish_time)
    if verbose:
        print(f"Done in {wallclock.time() - wall0:.1f} s of wall-clock time "
              f"({t:.0f} s simulated).")
    return SimResult(arrays, ctrl.events, metrics, ctrl.finished)


def compute_metrics(L: Dict[str, np.ndarray], distance: float, energy_j: float,
                    n_sat: int, n_ctrl: int, t_end: float, finished: bool,
                    finish_time: Optional[float]) -> Dict[str, float]:
    """Boil the run down to a handful of numbers you can compare between runs."""
    t = L["t"]
    xte = L["xte"]
    mask = t >= 30.0                       # ignore the initial convergence from our deliberately bad start
    if not np.any(mask):
        mask = np.ones_like(t, dtype=bool)
    heading_err = np.abs(wrap_pi(L["psi"] - L["psi_ref"]))
    return {
        "mission_time_s": finish_time if finish_time is not None else t_end,
        "finished": float(finished),
        "distance_m": distance,
        "mean_sog_mps": float(np.mean(L["sog"])),
        "max_sog_mps": float(np.max(L["sog"])),
        "xte_max_m": float(np.max(np.abs(xte))),
        "xte_rms_m": float(np.sqrt(np.mean(xte ** 2))),
        "xte_rms_after_30s_m": float(np.sqrt(np.mean(xte[mask] ** 2))),
        "xte_max_after_30s_m": float(np.max(np.abs(xte[mask]))),
        "heading_err_rms_deg": float(np.degrees(np.sqrt(np.mean(heading_err ** 2)))),
        "max_roll_deg": float(np.degrees(np.max(np.abs(L["phi"])))),
        "max_pitch_deg": float(np.degrees(np.max(np.abs(L["theta"])))),
        "max_heave_m": float(np.max(np.abs(L["z"]))),
        "energy_wh": energy_j / 3600.0,
        "thruster_saturation_pct": 100.0 * n_sat / max(n_ctrl, 1),
        "position_est_rms_err_m": float(np.sqrt(np.mean((L["x_est"] - L["x"]) ** 2
                                                        + (L["y_est"] - L["y"]) ** 2))),
        "gps_raw_rms_err_m": float(np.sqrt(np.mean((L["gps_x"] - L["x"]) ** 2
                                                   + (L["gps_y"] - L["y"]) ** 2))),
    }


def print_summary(res: SimResult, waypoints: np.ndarray) -> None:
    m = res.metrics
    print("\n" + "=" * 62)
    print(" MISSION SUMMARY")
    print("=" * 62)
    status = "completed" if res.finished else "NOT completed (hit the time limit)"
    print(f" Mission {status}")
    for (te, idx, dist) in res.events:
        print(f"   reached waypoint {idx:2d} at t = {te:7.1f} s  (miss distance {dist:4.2f} m)")
    print("-" * 62)
    print(f" mission time            : {m['mission_time_s']:9.1f} s")
    print(f" distance travelled      : {m['distance_m']:9.1f} m")
    print(f" mean / max speed        : {m['mean_sog_mps']:9.2f} / {m['max_sog_mps']:.2f} m/s")
    print(f" cross-track error (RMS) : {m['xte_rms_m']:9.2f} m   (after 30 s: {m['xte_rms_after_30s_m']:.2f} m)")
    print(f" cross-track error (max) : {m['xte_max_m']:9.2f} m   (after 30 s: {m['xte_max_after_30s_m']:.2f} m)")
    print(f" heading error (RMS)     : {m['heading_err_rms_deg']:9.2f} deg")
    print(f" max roll / pitch        : {m['max_roll_deg']:9.1f} / {m['max_pitch_deg']:.1f} deg")
    print(f" max heave               : {m['max_heave_m']:9.2f} m")
    print(f" thruster energy         : {m['energy_wh']:9.2f} Wh")
    print(f" thruster saturation     : {m['thruster_saturation_pct']:9.1f} % of control steps")
    print(f" position error RMS      : {m['position_est_rms_err_m']:9.2f} m filtered vs {m['gps_raw_rms_err_m']:.2f} m raw GPS")
    print("=" * 62)


def save_csv(res: SimResult, path: str) -> None:
    """Dump every logged channel to a CSV (angles in rad, everything SI)."""
    L = res.log
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([f"{k}[{LOG_UNITS[k]}]" for k in LOG_KEYS])
        for i in range(len(L["t"])):
            w.writerow([f"{L[k][i]:.6g}" for k in LOG_KEYS])


# ---------------------------------------------------------------------------
# 7. Waypoint utilities
# ---------------------------------------------------------------------------

def default_waypoints() -> np.ndarray:
    """A lazy loop with a mix of gentle and sharp turns.  (North, East) in metres."""
    return np.array([[0, 0], [50, 10], [90, 50], [90, 110],
                     [40, 140], [-10, 110], [-10, 50], [0, 0]], dtype=float)


def parse_waypoints(text: str) -> np.ndarray:
    """Parse "x,y;x,y;..." (North,East) into an (N,2) array."""
    pts = []
    for chunk in text.replace("\n", ";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p for p in chunk.replace(" ", ",").split(",") if p]
        if len(parts) != 2:
            raise ConfigError(f"Bad waypoint '{chunk}' - expected 'north,east'")
        try:
            pts.append([float(parts[0]), float(parts[1])])
        except ValueError:
            raise ConfigError(f"Waypoint '{chunk}' contains something that isn't a number")
    return validate_waypoints(np.array(pts, dtype=float))


def validate_waypoints(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    if w.ndim != 2 or w.shape[1] != 2 or w.shape[0] < 2:
        raise ConfigError("Need at least 2 waypoints, each as (north, east)")
    if not np.all(np.isfinite(w)):
        raise ConfigError("Waypoints must be finite numbers")
    legs = np.hypot(*(w[1:] - w[:-1]).T)
    if np.any(legs < 1e-6):
        raise ConfigError("Two consecutive waypoints are identical - remove the duplicate")
    return w


# ---------------------------------------------------------------------------
# 8. Plotting (and the optional GIF)
# ---------------------------------------------------------------------------

def _vessel_outline(length: float, beam: float, scale: float = 1.0) -> np.ndarray:
    """A pointy little hull shape in body coordinates (x forward, y starboard)."""
    hl, hb = 0.5 * length * scale, 0.5 * beam * scale
    return np.array([[hl, 0.0], [0.55 * hl, hb * 0.6], [-0.9 * hl, hb * 0.6], [-hl, hb * 0.35],
                     [-hl, -hb * 0.35], [-0.9 * hl, -hb * 0.6], [0.55 * hl, -hb * 0.6]])


def _outline_on_map(outline: np.ndarray, x: float, y: float, psi: float) -> np.ndarray:
    """Rotate/translate the body outline into (East, North) plotting coordinates."""
    c, s = math.cos(psi), math.sin(psi)
    xn = x + outline[:, 0] * c - outline[:, 1] * s
    ye = y + outline[:, 0] * s + outline[:, 1] * c
    return np.column_stack([ye, xn])          # plot East horizontally, North vertically


def make_plots(res: SimResult, waypoints: np.ndarray, vp: VesselParams, cp: ControllerParams,
               ep: EnvironmentParams, outdir: str, show: bool = False,
               animate: bool = False) -> List[str]:
    """Draw the trajectory map + a big dashboard of time series. Returns file paths."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")                # no display needed, works on servers/WSL/CI
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle, Polygon

    os.makedirs(outdir, exist_ok=True)
    L = res.log
    t = L["t"]
    saved: List[str] = []

    # ------------------------------------------------------------------ map
    fig, ax = plt.subplots(figsize=(9, 8))
    pts = np.column_stack([L["y"], L["x"]]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="viridis", linewidth=2.6,
                        norm=plt.Normalize(0.0, max(float(np.max(L["sog"])), 1e-3)))
    lc.set_array(L["sog"][:-1])
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax, label="speed over ground (m/s)", fraction=0.046, pad=0.03)

    ax.plot(waypoints[:, 1], waypoints[:, 0], "k--", lw=1.0, alpha=0.6, label="planned path")
    for i, (wx, wy) in enumerate(waypoints):
        ax.add_patch(Circle((wy, wx), cp.acceptance_radius, fill=False, ec="crimson", lw=1.0, alpha=0.7))
        ax.annotate(str(i), (wy, wx), textcoords="offset points", xytext=(6, 6), fontsize=9, color="crimson")
    ax.plot(waypoints[:, 1], waypoints[:, 0], "o", color="crimson", ms=4, label="waypoints")
    ax.plot(L["gps_y"], L["gps_x"], ".", color="0.7", ms=1.5, alpha=0.6, label="raw GPS fixes")
    ax.plot(L["y"][0], L["x"][0], "s", color="limegreen", ms=8, mec="k", label="start")

    # a few vessel silhouettes along the way (blown up so you can actually see them)
    extent = max(np.ptp(L["x"]), np.ptp(L["y"]), 1.0)
    scale = max(1.0, extent / 45.0)
    outline = _vessel_outline(vp.length, vp.beam, scale)
    for idx in np.linspace(0, len(t) - 1, 12).astype(int):
        ax.add_patch(Polygon(_outline_on_map(outline, L["x"][idx], L["y"][idx], L["psi"][idx]),
                             closed=True, fc="tab:orange", ec="k", lw=0.6, alpha=0.85, zorder=5))

    # little wind / current arrows in the corner
    for label, ang_to, spd, ypos, col in (
            ("wind", math.radians(ep.wind_dir_from_deg) + math.pi, ep.wind_speed, 0.93, "tab:blue"),
            ("current", math.radians(ep.current_dir_to_deg), ep.current_speed, 0.83, "tab:green")):
        if spd > 0:
            ax.annotate("", xy=(0.07 + 0.05 * math.sin(ang_to), ypos + 0.05 * math.cos(ang_to)),
                        xytext=(0.07, ypos), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color=col, lw=2))
            ax.text(0.14, ypos - 0.01, f"{label} {spd:.1f} m/s", transform=ax.transAxes,
                    fontsize=9, color=col, va="center")

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(f"{vp.name} - trajectory")
    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale_view()
    ax.margins(0.08)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    p1 = os.path.join(outdir, "trajectory.png")
    fig.savefig(p1, dpi=140, bbox_inches="tight")
    saved.append(p1)
    if not show:
        plt.close(fig)

    # ------------------------------------------------------------ dashboard
    psi_u = np.unwrap(L["psi"])
    psi_d_plot = psi_u + wrap_pi(L["psi_d"] - L["psi"])      # keep the curves on the same "branch"
    psi_ref_plot = psi_u + wrap_pi(L["psi_ref"] - L["psi"])

    fig, axs = plt.subplots(5, 2, figsize=(14, 16), sharex=True, constrained_layout=True)
    a = axs.ravel()

    a[0].plot(t, L["u"], label="surge u")
    a[0].plot(t, L["v"], label="sway v")
    a[0].plot(t, L["sog"], "--", label="SOG")
    a[0].plot(t, L["u_ref"], "k:", label="u ref")
    a[0].set_ylabel("m/s"); a[0].set_title("Velocities"); a[0].legend(fontsize=8, ncol=2)

    a[1].plot(t, np.degrees(psi_u), label="heading")
    a[1].plot(t, np.degrees(psi_d_plot), "--", label="LOS desired")
    a[1].plot(t, np.degrees(psi_ref_plot), ":", label="filtered ref")
    a[1].set_ylabel("deg"); a[1].set_title("Heading"); a[1].legend(fontsize=8)

    a[2].plot(t, np.degrees(L["r"]), label="yaw rate")
    a[2].plot(t, np.degrees(L["r_ref"]), "--", label="ref rate")
    a[2].set_ylabel("deg/s"); a[2].set_title("Yaw rate"); a[2].legend(fontsize=8)

    a[3].plot(t, L["xte"], color="tab:red")
    a[3].axhline(0, color="k", lw=0.5)
    a[3].set_ylabel("m"); a[3].set_title("Cross-track error (+ = right of path)")

    a[4].plot(t, L["Tp"], label="port actual")
    a[4].plot(t, L["Ts"], label="stbd actual")
    a[4].plot(t, L["Tp_cmd"], ":", label="port cmd", alpha=0.7)
    a[4].plot(t, L["Ts_cmd"], ":", label="stbd cmd", alpha=0.7)
    a[4].axhline(vp.thrust_max, color="k", lw=0.5, ls="--")
    a[4].axhline(vp.thrust_min, color="k", lw=0.5, ls="--")
    a[4].set_ylabel("N"); a[4].set_title("Thrusters (dashed = limits)"); a[4].legend(fontsize=8, ncol=2)

    a[5].plot(t, L["F_cmd"], label="surge force cmd (N)")
    a[5].plot(t, L["N_cmd"], label="yaw moment cmd (N m)")
    a[5].set_title("Controller demands"); a[5].legend(fontsize=8)

    a[6].plot(t, np.degrees(L["phi"]), label="roll")
    a[6].plot(t, np.degrees(L["theta"]), label="pitch")
    a[6].set_ylabel("deg"); a[6].set_title("Roll / pitch (waves + heel in turns)"); a[6].legend(fontsize=8)

    a[7].plot(t, L["eta"], label="wave elevation", alpha=0.6)
    a[7].plot(t, L["z"], label="vessel heave")
    a[7].set_ylabel("m"); a[7].set_title("Waves and heave"); a[7].legend(fontsize=8)

    a[8].plot(t, L["wind"], label="wind speed")
    a[8].plot(t, L["cur"], label="current speed")
    a[8].set_ylabel("m/s"); a[8].set_title("Environment"); a[8].set_xlabel("time (s)"); a[8].legend(fontsize=8)

    est_err = np.hypot(L["x_est"] - L["x"], L["y_est"] - L["y"])
    raw_err = np.hypot(L["gps_x"] - L["x"], L["gps_y"] - L["y"])
    a[9].plot(t, raw_err, alpha=0.5, label="raw GPS error")
    a[9].plot(t, est_err, label="Kalman filter error")
    a[9].set_ylabel("m"); a[9].set_title("Position estimation error"); a[9].set_xlabel("time (s)")
    a[9].legend(fontsize=8)

    for axx in a:
        axx.grid(alpha=0.3)
    for (te, idx, _) in res.events:                     # faint vertical lines at waypoint arrivals
        for axx in a:
            axx.axvline(te, color="crimson", lw=0.5, alpha=0.3)
    fig.suptitle(f"{vp.name} - time histories (red lines = waypoint reached)", fontsize=14)
    p2 = os.path.join(outdir, "dashboard.png")
    fig.savefig(p2, dpi=110)
    saved.append(p2)
    if not show:
        plt.close(fig)

    # ------------------------------------------------------------ optional GIF
    if animate:
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter
            figa, axa = plt.subplots(figsize=(6.5, 6.5))
            axa.plot(waypoints[:, 1], waypoints[:, 0], "k--", lw=1, alpha=0.5)
            axa.plot(waypoints[:, 1], waypoints[:, 0], "o", color="crimson", ms=4)
            trail, = axa.plot([], [], "-", color="tab:blue", lw=1.5)
            hull = Polygon(_outline_on_map(outline, L["x"][0], L["y"][0], L["psi"][0]),
                           closed=True, fc="tab:orange", ec="k", zorder=5)
            axa.add_patch(hull)
            # square limits so equal aspect just works (no matplotlib complaints)
            all_e = np.concatenate([L["y"], waypoints[:, 1]])
            all_n = np.concatenate([L["x"], waypoints[:, 0]])
            half = 0.5 * max(np.ptp(all_e), np.ptp(all_n)) + 10.0
            cen_e, cen_n = 0.5 * (all_e.min() + all_e.max()), 0.5 * (all_n.min() + all_n.max())
            axa.set_xlim(cen_e - half, cen_e + half)
            axa.set_ylim(cen_n - half, cen_n + half)
            axa.set_aspect("equal", adjustable="box")
            axa.set_xlabel("East (m)"); axa.set_ylabel("North (m)"); axa.grid(alpha=0.3)
            idxs = np.arange(0, len(t), max(1, len(t) // 250))

            def _frame(i: int):
                k = idxs[i]
                hull.set_xy(_outline_on_map(outline, L["x"][k], L["y"][k], L["psi"][k]))
                trail.set_data(L["y"][:k + 1], L["x"][:k + 1])
                axa.set_title(f"t = {t[k]:.0f} s   speed = {L['sog'][k]:.2f} m/s")
                return hull, trail

            anim = FuncAnimation(figa, _frame, frames=len(idxs), interval=50, blit=False)
            p3 = os.path.join(outdir, "animation.gif")
            anim.save(p3, writer=PillowWriter(fps=20), dpi=70)
            saved.append(p3)
            plt.close(figa)
        except Exception as exc:                        # animation is a bonus, never a crash
            print(f"(!) Couldn't make the GIF: {exc}  - is Pillow installed?")

    if show:
        plt.show()
    return saved


# ---------------------------------------------------------------------------
# 9. Self-test: a few physics sanity checks you can run in a few seconds
# ---------------------------------------------------------------------------

def run_selftest() -> bool:
    """Checks that don't need eyeballs.  Returns True when everything passes."""
    print("Running self-test ...")
    vp = VesselParams()
    vp.validate()
    dyn = ASVDynamics(vp)
    calm = EnvironmentParams(wind_speed=0, wind_gust_sigma=0, current_speed=0,
                             current_var_sigma=0, wave_hs=0)
    calm.validate()
    streams = [np.random.default_rng(i) for i in range(3)]
    env = Environment(calm, vp, streams)
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

    def integrate(cmd, seconds, s0=None, environment=env, dt=0.02):
        s = np.zeros(N_STATES) if s0 is None else s0.copy()
        t = 0.0
        for _ in range(int(seconds / dt)):
            s = dyn.step(t, s, cmd, environment.wind_ned(), environment.current_ned(),
                         environment.waves, dt)
            t += dt
        return s

    # 1. nothing pushes -> nothing moves
    s = integrate((0.0, 0.0), 10.0)
    check("still water, no thrust -> stays put", np.max(np.abs(s)) < 1e-9)

    # 2. 30 N + 30 N against Xu=25, Xuu=35 gives exactly 1.0 m/s at steady state
    s = integrate((30.0, 30.0), 150.0)
    check("steady speed under 60 N thrust", abs(s[IU] - 1.0) < 0.01, f"(u = {s[IU]:.4f}, expected 1.0)")

    # 3. more port thrust -> yaw to starboard (positive r, clockwise)
    # (only 3 s here - give it longer and the boat spins past 180 deg and the wrapped heading flips sign)
    s = integrate((30.0, 10.0), 3.0)
    check("port>stbd thrust turns clockwise", s[IR] > 0.0 and s[IPSI] > 0.0,
          f"(r = {math.degrees(s[IR]):.1f} deg/s)")

    # 4. no thrust, current of 0.5 m/s to the north -> boat drifts along with it
    cur_env = Environment(EnvironmentParams(wind_speed=0, wind_gust_sigma=0, current_speed=0.5,
                                            current_dir_to_deg=0.0, current_var_sigma=0, wave_hs=0),
                          vp, [np.random.default_rng(i) for i in range(3)])
    s = integrate((0.0, 0.0), 200.0, environment=cur_env)
    check("drifts with the current", abs(s[IX] - 0.5 * 200.0) < 8.0 and abs(s[IU] - 0.5) < 0.01,
          f"(x = {s[IX]:.1f} m, u = {s[IU]:.3f} m/s)")

    # 5. wave spectrum reproduces the requested Hs (std of elevation = Hs/4)
    wave_env = EnvironmentParams(wave_hs=0.6, wave_tp=4.0)
    wf = WaveField(wave_env, vp, np.random.default_rng(5))
    eta = np.array([wf.evaluate(0.0, 0.0, tt, 0.0)[0] for tt in np.arange(0.0, 3000.0, 0.5)])
    hs_est = 4.0 * float(np.std(eta))
    check("wave field gives requested Hs", abs(hs_est - 0.6) < 0.06, f"(estimated {hs_est:.3f} m vs 0.6 m)")

    # 6. thrust allocation respects the limits and preserves yaw first
    tp, ts, sat = allocate_thrust(200.0, 10.0, 0.4, vp.thrust_min, vp.thrust_max)
    check("allocation stays inside limits",
          vp.thrust_min - 1e-9 <= tp <= vp.thrust_max + 1e-9 and vp.thrust_min - 1e-9 <= ts <= vp.thrust_max + 1e-9 and sat)
    check("allocation keeps the yaw moment", abs(0.4 * (tp - ts) - 10.0) < 1e-9)

    # 7. angle wrapping
    check("wrap_pi works", abs(wrap_pi(3 * math.pi) - (-math.pi)) < 1e-9 or abs(wrap_pi(3 * math.pi) - math.pi) < 1e-9)

    # 8. calm-water full mission actually finishes
    res = run_simulation(vp, calm, SensorParams(), ControllerParams(), SimParams(t_end=600.0),
                         default_waypoints(), verbose=False)
    check("calm mission completes", res.finished, f"(t = {res.metrics['mission_time_s']:.0f} s)")
    check("calm mission cross-track RMS < 2 m", res.metrics["xte_rms_after_30s_m"] < 2.0,
          f"(rms = {res.metrics['xte_rms_after_30s_m']:.2f} m)")

    # 9a. overshoot logic: 5 m past the end of the last leg and 4 m off to the side, the
    #     controller must aim straight back at the waypoint instead of following the line onwards
    c = ASVController(ControllerParams(), vp, np.array([[0.0, 0.0], [80.0, 0.0]]), 0.0)
    o = c.update(0.0, 0.1, 85.0, 4.0, 0.0, 0.0, 1.0)
    bearing_back = math.atan2(0.0 - 4.0, 80.0 - 85.0)
    check("overshoot -> heads back for the waypoint",
          abs(wrap_pi(o.psi_d - bearing_back)) < math.radians(2.0) and not o.finished,
          f"(psi_d = {math.degrees(o.psi_d):.1f} deg, bearing = {math.degrees(bearing_back):.1f} deg)")

    # 9b. full run in a stiff crosswind with a deliberately tight circle - must still finish
    windy = EnvironmentParams(wind_speed=10.0, wind_dir_from_deg=270.0, wind_gust_sigma=0.0,
                              current_speed=0.4, current_dir_to_deg=90.0, current_var_sigma=0.0,
                              wave_hs=0.0)
    res = run_simulation(vp, windy, SensorParams(), replace(ControllerParams(), acceptance_radius=1.0),
                         SimParams(t_end=400.0), np.array([[0.0, 0.0], [80.0, 0.0]]), verbose=False)
    check("finishes despite crosswind + tight circle", res.finished,
          f"(finished = {res.finished}, t = {res.metrics['mission_time_s']:.0f} s)")

    print("Self-test", "PASSED" if ok else "FAILED")
    return ok


# ---------------------------------------------------------------------------
# 10. Command-line interface
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="3-DOF ASV simulator with wind, current, waves, sensors, guidance and control.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = ap.add_argument_group("mission")
    g.add_argument("--waypoints", type=str, default=None,
                   help="'north,east;north,east;...' in metres (default: built-in loop)")
    g.add_argument("--speed", type=float, default=None, help="cruise speed, m/s")
    g.add_argument("--lookahead", type=float, default=None, help="LOS lookahead distance, m")
    g.add_argument("--accept-radius", type=float, default=None, help="waypoint acceptance radius, m")

    g = ap.add_argument_group("environment (overrides on top of --sea-state)")
    g.add_argument("--sea-state", choices=sorted(SEA_STATES), default="moderate")
    g.add_argument("--wind", type=float, default=None, help="mean wind speed, m/s")
    g.add_argument("--wind-dir", type=float, default=None, help="direction wind comes FROM, deg")
    g.add_argument("--gust", type=float, default=None, help="gust std-dev, m/s")
    g.add_argument("--current", type=float, default=None, help="current speed, m/s")
    g.add_argument("--current-dir", type=float, default=None, help="direction current flows TO, deg")
    g.add_argument("--hs", type=float, default=None, help="significant wave height, m")
    g.add_argument("--tp", type=float, default=None, help="peak wave period, s")
    g.add_argument("--wave-dir", type=float, default=None, help="direction waves come FROM, deg")

    g = ap.add_argument_group("sensors")
    g.add_argument("--no-noise", action="store_true", help="perfect sensors, no Kalman filter")
    g.add_argument("--gps-sigma", type=float, default=None, help="GPS noise, m (1-sigma per axis)")
    g.add_argument("--compass-bias", type=float, default=None, help="compass bias, deg")

    g = ap.add_argument_group("simulation")
    g.add_argument("--duration", type=float, default=None, help="max simulated time, s")
    g.add_argument("--dt", type=float, default=None, help="physics time step, s")
    g.add_argument("--seed", type=int, default=None, help="random seed")
    g.add_argument("--x0", type=float, default=None, help="initial north position, m")
    g.add_argument("--y0", type=float, default=None, help="initial east position, m")
    g.add_argument("--psi0", type=float, default=None, help="initial heading, deg")
    g.add_argument("--no-stop", action="store_true", help="keep running after the last waypoint")

    g = ap.add_argument_group("output")
    g.add_argument("--outdir", type=str, default="asv_output")
    g.add_argument("--no-plots", action="store_true")
    g.add_argument("--show", action="store_true", help="open plot windows too")
    g.add_argument("--animate", action="store_true", help="also write animation.gif (needs Pillow)")
    g.add_argument("--quiet", action="store_true")
    g.add_argument("--selftest", action="store_true", help="run built-in checks and exit")
    return ap


def _override(obj, **kwargs):
    """dataclasses.replace() but skipping the None values (= 'user didn't set it')."""
    return replace(obj, **{k: v for k, v in kwargs.items() if v is not None})


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.selftest:
        return 0 if run_selftest() else 1

    try:
        vp = VesselParams()
        ep = replace(EnvironmentParams(), **SEA_STATES[args.sea_state])
        ep = _override(ep, wind_speed=args.wind, wind_dir_from_deg=args.wind_dir,
                       wind_gust_sigma=args.gust, current_speed=args.current,
                       current_dir_to_deg=args.current_dir, wave_hs=args.hs, wave_tp=args.tp,
                       wave_dir_from_deg=args.wave_dir)
        sp = _override(SensorParams(), gps_sigma=args.gps_sigma, compass_bias_deg=args.compass_bias)
        if args.no_noise:
            sp = replace(sp, enabled=False)
        cp = _override(ControllerParams(), cruise_speed=args.speed, lookahead=args.lookahead,
                       acceptance_radius=args.accept_radius)
        # if someone asks for a slow cruise speed, don't let turn_speed silently exceed it
        if cp.turn_speed > cp.cruise_speed:
            cp = replace(cp, turn_speed=0.6 * cp.cruise_speed)
        simp = _override(SimParams(), t_end=args.duration, dt=args.dt, seed=args.seed,
                         x0=args.x0, y0=args.y0, psi0_deg=args.psi0)
        if args.no_stop:
            simp = replace(simp, stop_on_finish=False)

        waypoints = parse_waypoints(args.waypoints) if args.waypoints else default_waypoints()

        res = run_simulation(vp, ep, sp, cp, simp, waypoints, verbose=not args.quiet)
    except (ConfigError, SimulationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "asv_log.csv")
    save_csv(res, csv_path)
    if not args.quiet:
        print_summary(res, waypoints)
        print(f"Log written to {csv_path}")

    if not args.no_plots:
        try:
            files = make_plots(res, waypoints, vp, cp, ep, args.outdir, show=args.show, animate=args.animate)
            if not args.quiet:
                for f in files:
                    print(f"Saved {f}")
        except ImportError:
            print("matplotlib isn't installed, so no plots this time (pip install matplotlib).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
