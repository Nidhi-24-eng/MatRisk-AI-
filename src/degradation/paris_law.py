"""
paris_law.py - Fracture Mechanics & Paris' Law Fatigue Parameter Derivation
Part of MatRisk AI (Phase 3, Day 5)

Theoretical Foundation:
1. Elastic Moduli Conversion:
   - Young's Modulus: E = 9*K*G / (3*K + G)
   - Poisson's Ratio: nu = (3*K - 2*G) / (2*(3*K + G))

2. Stress Intensity Factor (Mode I):
   K_I = Y * sigma * sqrt(pi * a)
   Delta K = Y * Delta sigma * sqrt(pi * a)
   where Y is the dimensionless geometry factor (typically Y = 1.12 for edge crack in semi-infinite plate).

3. Critical Crack Depth (Fast Catastrophic Fracture):
   K_max = Y * sigma_max * sqrt(pi * a_crit) = K_Ic
   => a_crit = (1 / pi) * (K_Ic / (Y * sigma_max))^2

4. Paris-Erdogan Fatigue Propagation Law:
   da/dN = C * (Delta K)^m   for Delta K >= Delta K_th
   da/dN = 0                 for Delta K < Delta K_th

5. Paris Constants Derivation & Scaling:
   - Exponent m: Typical structural alloy ranges (Steels ~ 3.0, Ti ~ 3.2, Al ~ 3.5, Superalloys ~ 2.8)
   - Constant C: Follows Donahue/Speidel empirical scaling relating to compliance:
     C ~= C_0 / (E^m)
"""

import math
from typing import Dict, Any, Optional, Tuple
import numpy as np


class MaterialMechanics:
    """Computes continuum and fracture mechanics parameters from predicted atomistic properties."""

    def __init__(
        self,
        bulk_modulus: float,        # K in GPa
        shear_modulus: float,       # G in GPa
        fracture_toughness: float,  # K_Ic in MPa*m^0.5
        geometry_factor: float = 1.12
    ):
        self.k_gpa = max(1.0, float(bulk_modulus))
        self.g_gpa = max(0.5, float(shear_modulus))
        self.k_ic = max(5.0, float(fracture_toughness))
        self.y_geom = geometry_factor

        # Derived Elastic Moduli
        self.e_gpa = (9.0 * self.k_gpa * self.g_gpa) / (3.0 * self.k_gpa + self.g_gpa)
        self.nu = (3.0 * self.k_gpa - 2.0 * self.g_gpa) / (2.0 * (3.0 * self.k_gpa + self.g_gpa))
        self.nu = max(-0.99, min(0.499, self.nu))  # Physical bounds

        # Derive Paris law exponent (m) and pre-factor (C)
        # Structural alloys with higher ductility (lower E/G) exhibit higher m
        # High-strength alloys with lower K_Ic experience faster acceleration
        self.paris_m, self.paris_c = self._derive_paris_constants()

        # Threshold stress intensity Delta K_th (MPa*m^0.5)
        # Empirical threshold typically scales with Young's modulus: Delta K_th ~ 1.5e-4 * E (MPa)
        self.delta_k_th = max(2.0, min(10.0, 1.6e-4 * (self.e_gpa * 1000.0)))

    def _derive_paris_constants(self) -> Tuple[float, float]:
        """
        Derives calibrated Paris law parameters (C, m) from elastic moduli.
        Unit convention:
        - a in meters (m)
        - Delta K in MPa*m^0.5
        - da/dN in meters/cycle
        """
        # Exponent m: typically 2.8 to 3.8 for structural metals
        # Materials with high K_Ic / G have higher toughness, lower m
        pugh_ratio = self.k_gpa / self.g_gpa
        if pugh_ratio > 2.0:
            # Ductile behavior (e.g., austenitic steel, nickel alloys)
            m = 2.9 + 0.1 * min(3.0, pugh_ratio - 2.0)
        else:
            # More brittle / high-strength (e.g., high-strength aluminum, hardened steel)
            m = 3.3 + 0.2 * (2.0 - pugh_ratio)

        m = float(np.clip(m, 2.5, 4.2))

        # Paris constant C: Speidel / Donahue compliance scaling
        # In SI units (m/cycle, MPa*m^0.5), C is typically ~ 1e-12 to 1e-10
        # C = A / (E^m) where E is in GPa
        # Calibration constant A ~ 5e-8 to 5e-7
        c = 3.5e-7 / (self.e_gpa ** m)
        c = float(np.clip(c, 1e-14, 1e-9))

        return m, c

    def compute_critical_crack_depth(self, sigma_max_mpa: float) -> float:
        """
        Computes the critical crack depth a_crit (in meters) where catastrophic fracture occurs:
        a_crit = (1 / pi) * ( K_Ic / (Y * sigma_max) )^2
        """
        sigma_eff = max(1.0, sigma_max_mpa)
        a_crit = (1.0 / math.pi) * ((self.k_ic / (self.y_geom * sigma_eff)) ** 2)
        # Cap a_crit to realistic structural thickness bounds [0.005m, 0.50m]
        return float(np.clip(a_crit, 0.002, 0.50))

    def compute_stress_intensity(self, crack_depth_m: float, delta_sigma_mpa: float) -> float:
        """
        Computes stress intensity range Delta K (MPa*m^0.5):
        Delta K = Y * Delta sigma * sqrt(pi * a)
        """
        a_eff = max(1e-7, crack_depth_m)
        delta_k = self.y_geom * delta_sigma_mpa * math.sqrt(math.pi * a_eff)
        return float(delta_k)

    def crack_growth_rate(self, crack_depth_m: float, delta_sigma_mpa: float) -> float:
        """
        Computes da/dN (meters per cycle) under Paris' Law:
        da/dN = C * (Delta K)^m   if Delta K >= Delta K_th else 0
        """
        delta_k = self.compute_stress_intensity(crack_depth_m, delta_sigma_mpa)
        if delta_k < self.delta_k_th:
            return 0.0
        return float(self.paris_c * (delta_k ** self.paris_m))

    def deterministic_cycles_to_failure(
        self,
        a_0_m: float,
        sigma_max_mpa: float,
        delta_sigma_mpa: float
    ) -> Dict[str, float]:
        """
        Analytically integrates Paris' Law from initial crack depth a_0 to a_crit:
        N_f = integral_{a_0}^{a_crit} da / (C * (Y * Delta sigma * sqrt(pi * a))^m)
        """
        a_crit = self.compute_critical_crack_depth(sigma_max_mpa)
        a_0 = max(1e-6, min(a_0_m, a_crit * 0.99))

        m = self.paris_m
        c = self.paris_c
        geom_factor = (self.y_geom * delta_sigma_mpa * math.sqrt(math.pi)) ** m

        if abs(m - 2.0) < 1e-4:
            # Special case m = 2
            n_cycles = math.log(a_crit / a_0) / (c * geom_factor)
        else:
            exponent = 1.0 - (m / 2.0)
            n_cycles = (a_crit ** exponent - a_0 ** exponent) / (c * geom_factor * exponent)

        n_cycles = max(0.0, float(n_cycles))

        return {
            "a_0_m": a_0,
            "a_crit_m": a_crit,
            "delta_k_initial": self.compute_stress_intensity(a_0, delta_sigma_mpa),
            "delta_k_critical": self.k_ic,
            "paris_m": m,
            "paris_c": c,
            "n_cycles_to_failure": n_cycles
        }


def analyze_alloy_mechanics(
    bulk_modulus: float,
    shear_modulus: float,
    fracture_toughness: float,
    sigma_max_mpa: float = 250.0,
    delta_sigma_mpa: float = 120.0,
    a_0_m: float = 0.001
) -> Dict[str, Any]:
    """Helper function to produce complete mechanics diagnostic summary."""
    mech = MaterialMechanics(bulk_modulus, shear_modulus, fracture_toughness)
    life_info = mech.deterministic_cycles_to_failure(a_0_m, sigma_max_mpa, delta_sigma_mpa)

    return {
        "youngs_modulus_gpa": round(mech.e_gpa, 2),
        "poissons_ratio": round(mech.nu, 3),
        "paris_m": round(mech.paris_m, 3),
        "paris_c": f"{mech.paris_c:.3e}",
        "delta_k_th_mpa_m05": round(mech.delta_k_th, 2),
        "a_crit_mm": round(life_info["a_crit_m"] * 1000.0, 3),
        "n_cycles_deterministic": int(life_info["n_cycles_to_failure"])
    }


if __name__ == "__main__":
    print("[*] Testing Material Mechanics & Paris' Law derivation...")

    # Case 1: Structural Steel (K = 160 GPa, G = 77 GPa, K_Ic = 75 MPa*m^0.5)
    steel_diag = analyze_alloy_mechanics(
        bulk_modulus=160.0,
        shear_modulus=77.0,
        fracture_toughness=75.0,
        sigma_max_mpa=280.0,
        delta_sigma_mpa=140.0,
        a_0_m=0.002
    )

    print("[+] Structural Steel Mechanics Diagnostic:")
    for k, v in steel_diag.items():
        print(f"    - {k:<25}: {v}")
