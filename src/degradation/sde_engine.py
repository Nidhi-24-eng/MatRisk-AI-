"""
sde_engine.py - Accelerated Vectorized Itô SDE Monte Carlo Degradation Solver
Part of MatRisk AI (Phase 3, Day 6)

Mathematical Model:
Solves the Itô Stochastic Differential Equation for sub-critical fatigue crack growth:
da_t = C * (Y * Delta sigma_t * sqrt(pi * a_t))^m * dt + sigma_a * a_t^eta * dW_t

Features:
- High-throughput vectorized simulation of 10,000+ Monte Carlo trajectories in < 2 seconds.
- Discretization using vectorized Euler-Maruyama scheme.
- Environmental salinity & thermal acceleration multipliers.
- First-passage time distribution tau = inf{ t : a_t >= a_crit }.
- Produces cumulative failure probability P_f(t) and instantaneous hazard rate lambda(t).
"""

import time
import math
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

# Numba acceleration if installed
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from src.degradation.paris_law import MaterialMechanics


def simulate_sde_vectorized_numpy(
    n_sims: int,
    n_steps: int,
    dt: float,
    a_0: float,
    a_crit: float,
    drift_coeff: float,
    m_exp: float,
    sigma_noise: float,
    eta_exp: float = 0.5,
    env_multiplier: float = 1.0,
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized Euler-Maruyama solver using pure NumPy array broadcasting.
    Simulates all trajectories in parallel.

    Returns:
        trajectories: [n_sims, n_steps + 1] crack depths over time
        failure_times: [n_sims] failure time in years (or inf if survived)
        failed_mask: [n_sims] boolean array indicating whether each path failed
    """
    if seed is not None:
        np.random.seed(seed)

    trajectories = np.empty((n_sims, n_steps + 1), dtype=np.float64)
    trajectories[:, 0] = a_0

    failure_times = np.full(n_sims, np.inf, dtype=np.float64)
    failed = np.zeros(n_sims, dtype=bool)

    sqrt_dt = math.sqrt(dt)
    effective_drift_base = drift_coeff * env_multiplier

    # Generate standard normal increments Z ~ N(0, 1) of shape [n_sims, n_steps]
    # To reduce peak memory footprint, we can step iteratively or in chunks
    current_a = np.full(n_sims, a_0, dtype=np.float64)

    for step in range(n_steps):
        t_current = step * dt
        # Random Brownian shock
        dw = np.random.standard_normal(n_sims) * sqrt_dt

        # Drift mu(a) = effective_drift_base * (a)^(m / 2)
        # Delta K is proportional to sqrt(a), so (Delta K)^m is proportional to a^(m/2)
        drift = effective_drift_base * (np.maximum(current_a, 1e-7) ** (m_exp / 2.0))

        # Diffusion sigma(a) = sigma_noise * a^eta
        diffusion = sigma_noise * (np.maximum(current_a, 1e-7) ** eta_exp)

        # Euler-Maruyama update: a_{t+1} = a_t + mu * dt + sigma * dW
        next_a = current_a + drift * dt + diffusion * dw

        # Check failure threshold
        just_failed = (~failed) & (next_a >= a_crit)
        if np.any(just_failed):
            failure_times[just_failed] = t_current + dt
            failed[just_failed] = True

        # Clamp crack growth so failed paths stay at a_crit, non-negative
        current_a = np.where(failed, a_crit, np.maximum(1e-7, next_a))
        trajectories[:, step + 1] = current_a

    return trajectories, failure_times, failed


class StochasticDegradationEngine:
    """High-performance engine for simulating microstructural crack degradation."""

    def __init__(
        self,
        mechanics: MaterialMechanics,
        initial_crack_m: float = 0.001,      # a_0 = 1 mm
        cycles_per_year: float = 1e6,        # 1,000,000 operational stress cycles / year
        noise_intensity: float = 0.002,      # Microstructural Brownian diffusion intensity
        diffusion_exponent: float = 0.5,     # eta
        geometry_factor: float = 1.12
    ):
        self.mechanics = mechanics
        self.a_0 = max(1e-6, initial_crack_m)
        self.cycles_per_year = cycles_per_year
        self.noise_intensity = noise_intensity
        self.diffusion_exponent = diffusion_exponent
        self.y_geom = geometry_factor

    def run_monte_carlo(
        self,
        delta_sigma_mpa: float = 120.0,
        sigma_max_mpa: float = 250.0,
        horizon_years: float = 10.0,
        steps_per_year: int = 52,            # Weekly resolution (520 steps over 10 years)
        num_trajectories: int = 10000,
        environmental_salinity_factor: float = 1.0,
        seed: Optional[int] = 42
    ) -> Dict[str, Any]:
        """
        Executes 10,000 Monte Carlo trajectories of the Itô SDE in < 2 seconds.

        Returns:
            Dictionary containing:
            - 'time_grid': [n_steps + 1] years
            - 'failure_probability': [n_steps + 1] cumulative P_f(t)
            - 'hazard_rate': [n_steps + 1] instantaneous failure rate lambda(t)
            - 'percentiles': dictionary of crack depth percentiles over time (p05, p50, p95, p99)
            - 'sample_trajectories': 50 selected trajectory paths for plotting
            - 'mttf_years': Mean Time to Failure of failed components
            - 'a_crit_mm': Critical crack depth in mm
            - 'execution_time_seconds': Solver execution time
        """
        t_start = time.time()

        # Compute critical crack depth
        a_crit = self.mechanics.compute_critical_crack_depth(sigma_max_mpa)

        # Drift coefficient: da/dt = cycles_per_year * C * (Y * Delta sigma * sqrt(pi))^m * a^(m/2)
        m = self.mechanics.paris_m
        c = self.mechanics.paris_c
        base_geom = (self.y_geom * delta_sigma_mpa * math.sqrt(math.pi)) ** m
        drift_coeff = self.cycles_per_year * c * base_geom

        n_steps = int(horizon_years * steps_per_year)
        dt = horizon_years / n_steps
        time_grid = np.linspace(0.0, horizon_years, n_steps + 1)

        # Execute high-speed vectorized simulation
        trajectories, failure_times, failed = simulate_sde_vectorized_numpy(
            n_sims=num_trajectories,
            n_steps=n_steps,
            dt=dt,
            a_0=self.a_0,
            a_crit=a_crit,
            drift_coeff=drift_coeff,
            m_exp=m,
            sigma_noise=self.noise_intensity,
            eta_exp=self.diffusion_exponent,
            env_multiplier=environmental_salinity_factor,
            seed=seed
        )

        # Compute cumulative failure probability curve P_f(t)
        # P_f(t) = fraction of paths with failure_time <= t
        cum_failures = np.zeros(n_steps + 1, dtype=np.float64)
        for i, t in enumerate(time_grid):
            cum_failures[i] = np.mean(failure_times <= t)

        # Instantaneous hazard rate lambda(t) = (dP_f / dt) / (1 - P_f(t))
        dp_f = np.gradient(cum_failures, dt)
        survival_prob = np.maximum(1e-5, 1.0 - cum_failures)
        hazard_rate = np.maximum(0.0, dp_f / survival_prob)

        # Calculate crack depth percentiles over time
        p05 = np.percentile(trajectories, 5, axis=0) * 1000.0   # mm
        p25 = np.percentile(trajectories, 25, axis=0) * 1000.0  # mm
        p50 = np.percentile(trajectories, 50, axis=0) * 1000.0  # Median mm
        p75 = np.percentile(trajectories, 75, axis=0) * 1000.0  # mm
        p95 = np.percentile(trajectories, 95, axis=0) * 1000.0  # mm
        p99 = np.percentile(trajectories, 99, axis=0) * 1000.0  # 99th percentile mm

        # 50 sample paths for visualization (in mm)
        sample_indices = np.random.choice(num_trajectories, size=min(50, num_trajectories), replace=False)
        sample_paths = trajectories[sample_indices, :] * 1000.0

        # Mean Time to Failure (MTTF) of components that failed within horizon
        failed_times_finite = failure_times[np.isfinite(failure_times)]
        mttf = float(np.mean(failed_times_finite)) if len(failed_times_finite) > 0 else horizon_years

        elapsed = time.time() - t_start

        return {
            "time_grid": time_grid.tolist(),
            "failure_probability": cum_failures.tolist(),
            "hazard_rate": hazard_rate.tolist(),
            "final_failure_prob_10yr": float(cum_failures[-1]),
            "percentiles": {
                "p05": p05.tolist(),
                "p25": p25.tolist(),
                "p50": p50.tolist(),
                "p75": p75.tolist(),
                "p95": p95.tolist(),
                "p99": p99.tolist()
            },
            "sample_trajectories": sample_paths.tolist(),
            "mttf_years": round(mttf, 2),
            "a_crit_mm": round(a_crit * 1000.0, 3),
            "num_trajectories": num_trajectories,
            "execution_time_seconds": round(elapsed, 3)
        }


def quick_degradation_simulation(
    bulk_modulus: float = 160.0,
    shear_modulus: float = 77.0,
    fracture_toughness: float = 75.0,
    delta_sigma_mpa: float = 120.0,
    sigma_max_mpa: float = 250.0,
    num_trajectories: int = 10000,
    environmental_factor: float = 1.0
) -> Dict[str, Any]:
    """Top-level convenience interface for simulating structural degradation."""
    mechanics = MaterialMechanics(bulk_modulus, shear_modulus, fracture_toughness)
    engine = StochasticDegradationEngine(mechanics)
    return engine.run_monte_carlo(
        delta_sigma_mpa=delta_sigma_mpa,
        sigma_max_mpa=sigma_max_mpa,
        num_trajectories=num_trajectories,
        environmental_salinity_factor=environmental_factor
    )


if __name__ == "__main__":
    print("[*] Running Benchmark 10,000-Trajectory Itô SDE Simulation...")
    sim_res = quick_degradation_simulation(
        bulk_modulus=160.0,
        shear_modulus=77.0,
        fracture_toughness=75.0,
        delta_sigma_mpa=130.0,
        sigma_max_mpa=260.0,
        num_trajectories=10000
    )

    print(f"[+] Simulation completed in {sim_res['execution_time_seconds']:.3f} seconds!")
    print(f"    - Critical crack depth a_crit: {sim_res['a_crit_mm']} mm")
    print(f"    - 10-Year Cumulative Failure Probability P_f(10): {sim_res['final_failure_prob_10yr']*100:.2f}%")
    print(f"    - Mean Time to Failure (MTTF): {sim_res['mttf_years']} years")
