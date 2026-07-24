"""Splice synth/validation_results.json into README.md's validation table.

Replaces everything between the `<!-- VALIDATION_TABLE_START -->` and
`<!-- VALIDATION_TABLE_END -->` markers with a freshly generated table.
Used by .github/workflows/refresh-data.yml after synth.validate runs;
safe to run manually too: `python -m synth.update_readme`.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
RESULTS_PATH = Path(__file__).parent / "validation_results.json"
README_PATH = REPO_ROOT / "README.md"
START_MARKER = "<!-- VALIDATION_TABLE_START -->"
END_MARKER = "<!-- VALIDATION_TABLE_END -->"


def _fmt_pct(x: float) -> str:
    return f"{x:+.1f}%"


def build_table(results: dict) -> str:
    pe = results["price_elasticity"]
    seg = results["segmentation"]
    nrw = results["nrw"]
    de = results["demand_ensemble"]

    pe_t = pe["targets"]
    seg_t = seg["targets"]

    a = nrw.get("system_a", {})
    b = nrw.get("system_b", {})

    real_pt = "significant" if pe_t["did_p_value"] < 0.05 else "not significant"
    synth_pt = "significant" if pe["did_p_value"] < 0.05 else "not significant"
    real_sign = "negative" if pe_t["did_effect_mgd"] < 0 else "positive"
    synth_sign = "negative" if pe["did_effect_mgd"] < 0 else "positive"

    rows = [
        ("Price-elasticity DiD: tier1 step (% of weather-expected demand)",
         _fmt_pct(pe_t["tier1_step_pct"]), _fmt_pct(pe["tier1_step_pct"])),
        ("Price-elasticity DiD: tier2/3 step (% of weather-expected demand)",
         _fmt_pct(pe_t["tier23_step_pct"]), _fmt_pct(pe["tier23_step_pct"])),
        ("DiD effect sign / significance",
         f"{real_sign}, p={pe_t['did_p_value']:.3f} ({real_pt})",
         f"{synth_sign}, p={pe['did_p_value']:.4f} ({synth_pt})"),
        ("Segmentation: top-consumers volume share",
         f"1.0% of accounts → {seg_t['top_consumers_volume_share_pct']:.1f}% of volume",
         f"1.0% → {seg['top_consumers_volume_share_pct']:.1f}%"),
        ("Segmentation: vacant/intermittent share",
         f"{seg_t['vacant_share_pct']:.1f}%", f"{seg['vacant_share_pct']:.1f}%"),
        ("Segmentation cluster count (k)", str(seg_t["segmentation_k"]), str(seg["k"])),
        ("NRW correlation, system A (master-meter-summed input)",
         f"{a.get('target', float('nan')):.3f}" if a.get("target") is not None else "n/a",
         f"{a.get('correlation', float('nan')):.3f}" if a.get("correlation") is not None else "n/a"),
        ("NRW correlation, system B (WTP production input)",
         f"{b.get('target', float('nan')):.3f}" if b.get("target") is not None else "n/a",
         f"{b.get('correlation', float('nan')):.3f}" if b.get("correlation") is not None else "n/a"),
        ("Demand forecast series: mean (MGD)",
         f"{de['targets']['demand_mean_mgd']:.1f}", f"{de['demand_mean_mgd']:.1f}"),
        ("Demand forecast series: autocorrelation (lag 1 / 7 / 14)",
         " / ".join(f"{de['targets']['demand_acf'][lag]:.2f}" for lag in ("1", "7", "14")),
         " / ".join(f"{de['demand_acf'][lag]:.2f}" for lag in ("1", "7", "14"))),
    ]

    lines = ["| Metric | Real | Synthetic |", "|---|---:|---:|"]
    for label, real, synth in rows:
        lines.append(f"| {label} | {real} | {synth} |")
    return "\n".join(lines)


def main():
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    table = build_table(results)

    readme = README_PATH.read_text(encoding="utf-8")
    start = readme.index(START_MARKER) + len(START_MARKER)
    end = readme.index(END_MARKER)
    new_readme = readme[:start] + "\n" + table + "\n" + readme[end:]
    README_PATH.write_text(new_readme, encoding="utf-8")
    print(f"Updated {README_PATH} validation table.")


if __name__ == "__main__":
    main()
