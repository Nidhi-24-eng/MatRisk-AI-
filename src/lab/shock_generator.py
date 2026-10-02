"""
shock_generator.py - Environmental & Macro Financial Shock Scenario Presets
Part of MatRisk AI (Phase 6, Day 12)

Provides one-click scenario shock presets that override sidebar slider values
to demonstrate real-time cross-domain cascading through all 5 MatRisk engines.
"""

from typing import Dict, Any


# Pre-loaded structural alloy mechanical benchmarks
ALLOY_PROFILES = {
    "Inconel 718 Superalloy": {
        "bulk_modulus": 182.0,
        "shear_modulus": 88.0,
        "fracture_toughness": 85.0,
        "description": "Ni₃Al γ′-precipitate hardened nickel superalloy (FCC). Jet turbine discs, offshore risers.",
        "lattice": "FCC (a = 3.568 Å)",
        "atoms": 4,
    },
    "Structural Steel A36": {
        "bulk_modulus": 160.0,
        "shear_modulus": 77.0,
        "fracture_toughness": 55.0,
        "description": "Ferritic/pearlitic mild carbon steel (BCC). Bridges, I-beams, structural frames.",
        "lattice": "BCC (a = 2.866 Å)",
        "atoms": 2,
    },
    "Austenitic Stainless 316L": {
        "bulk_modulus": 160.0,
        "shear_modulus": 77.0,
        "fracture_toughness": 75.0,
        "description": "Austenitic stainless steel with Mo addition (FCC). Marine platforms, chemical reactors.",
        "lattice": "FCC (a = 3.590 Å)",
        "atoms": 4,
    },
    "Ti-6Al-4V Titanium": {
        "bulk_modulus": 110.0,
        "shear_modulus": 44.0,
        "fracture_toughness": 55.0,
        "description": "α+β titanium alloy (HCP). Aerospace frames, biomedical implants, subsea fasteners.",
        "lattice": "HCP (a = 2.950 Å, c = 4.686 Å)",
        "atoms": 2,
    },
    "Aluminum 7075-T6": {
        "bulk_modulus": 76.0,
        "shear_modulus": 26.0,
        "fracture_toughness": 29.0,
        "description": "High-strength precipitation-hardened aluminum (FCC). Aircraft wings, automotive chassis.",
        "lattice": "FCC (a = 4.049 Å)",
        "atoms": 4,
    },
    "Structural Copper Alloy": {
        "bulk_modulus": 137.0,
        "shear_modulus": 48.0,
        "fracture_toughness": 60.0,
        "description": "Structural copper for electrical and thermal conductors (FCC). Bus bars, heat exchangers.",
        "lattice": "FCC (a = 3.615 Å)",
        "atoms": 4,
    },
    "Tungsten Carbide (WC)": {
        "bulk_modulus": 390.0,
        "shear_modulus": 280.0,
        "fracture_toughness": 15.0,
        "description": "Ultra-hard ceramic-metallic compound (Hex). Cutting tools, drill bits, wear plates.",
        "lattice": "Hex (a = 2.906 Å, c = 2.837 Å)",
        "atoms": 2,
    },
}


# Default baseline parameters
BASELINE_PARAMS = {
    "alloy": "Inconel 718 Superalloy",
    "delta_sigma": 180.0,
    "initial_flaw_mm": 1.0,
    "cycles_per_year": 1_000_000,
    "salinity_pct": 1.5,
    "temperature_c": 25.0,
    "capex_millions": 100.0,
    "debt_pct": 70.0,
    "base_ebitda_millions": 18.0,
    "interest_rate_pct": 6.5,
    "target_dscr": 1.30,
    "sculpting_enabled": False,
}


def get_shock_preset(preset_name: str) -> Dict[str, Any]:
    """Returns parameter overrides for a given shock scenario preset."""

    if preset_name == "extreme_marine":
        return {
            "label": "🔴 Extreme Thermal Marine Shock",
            "description": "Simulates simultaneous high-salinity marine splash zone corrosion and extreme thermal fatigue — typical of North Sea offshore riser joints or tropical refinery piping.",
            "overrides": {
                "salinity_pct": 5.5,
                "temperature_c": 450.0,
                "delta_sigma": 320.0,
            }
        }

    elif preset_name == "over_capacity":
        return {
            "label": "🟠 Operational Over-Capacity Stress",
            "description": "Asset operator runs the plant beyond design capacity to maximize short-term revenue, increasing cyclic stress by +40% while boosting EBITDA by +15%.",
            "overrides": {
                "delta_sigma": 252.0,  # 180 * 1.40
                "base_ebitda_millions": 20.7,  # 18.0 * 1.15
            }
        }

    elif preset_name == "stagflation":
        return {
            "label": "🟡 Stagflation + Corrosion Double Shock",
            "description": "Central bank rate hikes (+300 bps) combined with accelerated environmental corrosion doubling the Paris' Law crack propagation constant C.",
            "overrides": {
                "interest_rate_pct": 9.5,  # 6.5 + 3.0
                "salinity_pct": 3.5,
                "temperature_c": 120.0,
            }
        }

    elif preset_name == "reset":
        return {
            "label": "🟢 Reset to Baseline",
            "description": "Restores all parameters to nominal design operating conditions.",
            "overrides": {k: v for k, v in BASELINE_PARAMS.items() if k != "alloy"}
        }

    return {"label": "Unknown", "description": "", "overrides": {}}


def compute_environmental_multiplier(salinity_pct: float, temperature_c: float) -> float:
    """
    Calculates environmental acceleration multiplier for Paris' Law constant C.
    Accounts for chloride-induced stress corrosion cracking (SCC) and thermal creep.
    """
    # Salinity acceleration: chloride pitting accelerates crack initiation
    # At 0% salinity: multiplier = 1.0, at 5%: multiplier ~ 2.5
    salt_mult = 1.0 + 0.30 * salinity_pct

    # Thermal acceleration: Arrhenius-type creep contribution
    # Below 100°C: negligible, above 300°C: significant
    if temperature_c <= 100.0:
        thermal_mult = 1.0
    elif temperature_c <= 300.0:
        thermal_mult = 1.0 + 0.002 * (temperature_c - 100.0)
    else:
        thermal_mult = 1.4 + 0.003 * (temperature_c - 300.0)

    return round(salt_mult * thermal_mult, 3)
