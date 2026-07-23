"""Fictional pressure-zone + master-meter geography generator.

Produces a fabricated pressure-zone shapefile and master-meter site table
with the same STRUCTURE as a real water utility's system -- zone count,
area distribution, system split, meter-role counts -- but no real
coordinates, addresses, names, or account numbers. Nothing in this module
was derived from or transformed out of any real utility's GIS export or
site list; every polygon here is procedurally generated from scratch.

The handful of constants below (zone count, acreage distribution, HGL
range, site-role counts) are aggregate structural statistics -- counts and
magnitude distributions only, never a name, coordinate, or per-zone
assignment -- pulled by hand from a real utility's GIS/config data during
this project's calibration phase (see the README for the full
calibrate-then-generate methodology).

Run: python -m synth.geography
"""
from __future__ import annotations

import json
import string
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapefile
import shapely
import shapely.affinity
from shapely.geometry import MultiPoint, box

# ── Real structural stats (aggregate only) ───────────
N_ZONES = 20
SYSTEM_A_ZONE_COUNT = 5     # 5-zone / 15-zone split, matching a real two-system utility
SYSTEM_B_ZONE_COUNT = N_ZONES - SYSTEM_A_ZONE_COUNT
REAL_ACREAGES_SORTED = [0.0, 135.0, 149.0, 169.0, 195.0, 195.0, 208.0, 316.0,
                         329.0, 344.0, 372.0, 426.0, 523.0, 531.0, 771.0,
                         1277.0, 1470.0, 1649.0, 1804.0, 3382.0]
TOTAL_ACREAGE = sum(REAL_ACREAGES_SORTED)   # ~14,245 ac (~22.3 sq mi)
HGL_MIN_FT, HGL_MAX_FT = 5620, 6385
SITE_ROLE_COUNTS = {"dw_master": 24, "retired": 5, "dw_submeter": 3,
                    "mg_internal": 2, "emergency": 2}   # 36 beacon sites total

SYSTEM_LABELS = {"system_a": "Ridgeline", "system_b": "Cottonwood"}

# Deliberately NOT any real utility's actual location: an arbitrary,
# fabricated origin, EPSG:2232 (Colorado Central State Plane, feet)
# projection so the dashboard's OSM basemap still shows "somewhere in the
# Front Range foothills" for narrative realism, without corresponding to
# any real service area.
_ORIGIN_X, _ORIGIN_Y = 3_180_000.0, 1_760_000.0
_BBOX_WIDTH_FT, _BBOX_HEIGHT_FT = 31_680.0, 19_800.0   # ~6mi x 3.75mi, rescaled to match TOTAL_ACREAGE below

ZONE_WORDS = ["Sagebrush", "Wildrose", "Antler", "Limestone", "Bluestem",
    "Cinder", "Foxglove", "Granite", "Harrier", "Ironwood", "Juniper",
    "Kestrel", "Larkspur", "Mesa", "Nighthawk", "Obsidian", "Pinyon",
    "Quartzite", "Ridgeback", "Sandhill", "Talus", "Umber", "Vireo", "Windrow"]
ZONE_PREFIXES = list(string.ascii_uppercase[:N_ZONES])   # A..T, matches real letter-code style

STREET_WORDS = ["Antler", "Bramble", "Cinderpath", "Dovetail", "Elkhorn",
    "Foxtail", "Granite", "Harrier", "Ironwood", "Juniper", "Kestrel",
    "Larkspur", "Meadowlark", "Nighthawk", "Obsidian", "Pinyon", "Quartz",
    "Ridgeline", "Sandhill", "Talus", "Umber", "Vireo", "Windrow", "Yarrow"]
STREET_SUFFIXES = ["St", "Ave", "Rd", "Ct", "Dr", "Ln", "Way", "Pl"]
FICTIONAL_TOWN = "Antler Creek, CO"
ROLE_LABELS = {"dw_master": "DW Master", "dw_submeter": "DW Submeter",
              "mg_internal": "MG Internal", "retired": "Retired",
              "emergency": "Emergency Interconnect"}


@dataclass
class FictionalZone:
    name: str
    system: str
    acreage: float
    hgl_ft: int
    polygon: object   # shapely.Polygon


@dataclass
class FictionalSite:
    id: str
    kind: str
    label: str
    role: str | None
    beacon_account: str | None
    meter_id: str | None
    zone: str | None


def _voronoi_zones(rng: np.random.Generator, n: int) -> list:
    """n roughly-equal-area irregular polygons partitioning a bounded
    rectangle, via a Voronoi diagram on jittered grid seed points --
    structurally similar in KIND to a real pressure-zone layout (irregular,
    contiguous, space-filling) without being derived from any real geometry.
    Jittered grid rather than pure-random points so no zone degenerates to
    a sliver.
    """
    bounds = box(_ORIGIN_X, _ORIGIN_Y, _ORIGIN_X + _BBOX_WIDTH_FT, _ORIGIN_Y + _BBOX_HEIGHT_FT)
    cols = int(np.ceil(np.sqrt(n * _BBOX_WIDTH_FT / _BBOX_HEIGHT_FT)))
    rows = int(np.ceil(n / cols))
    xs = np.linspace(_ORIGIN_X, _ORIGIN_X + _BBOX_WIDTH_FT, cols + 2)[1:-1]
    ys = np.linspace(_ORIGIN_Y, _ORIGIN_Y + _BBOX_HEIGHT_FT, rows + 2)[1:-1]
    pts = [(x, y) for y in ys for x in xs][:n]
    jitter_x, jitter_y = _BBOX_WIDTH_FT / cols * 0.35, _BBOX_HEIGHT_FT / rows * 0.35
    pts = [(x + rng.uniform(-jitter_x, jitter_x), y + rng.uniform(-jitter_y, jitter_y))
           for x, y in pts]
    mp = MultiPoint(pts)
    regions = shapely.voronoi_polygons(mp, extend_to=bounds)
    clipped = [r.intersection(bounds) for r in regions.geoms]
    return [c for c in clipped if not c.is_empty and c.area > 0]


def generate_zones(seed: int = 20260723) -> list[FictionalZone]:
    rng = np.random.default_rng(seed)
    polys = _voronoi_zones(rng, N_ZONES)
    tries = 0
    while len(polys) < N_ZONES and tries < 5:   # extremely unlikely, stay correct anyway
        polys = _voronoi_zones(rng, N_ZONES + 1 + tries)
        tries += 1
    polys = polys[:N_ZONES]
    if len(polys) < N_ZONES:
        raise RuntimeError(f"_voronoi_zones only produced {len(polys)}/{N_ZONES} regions")

    # Rescale so total area matches the real system's total acreage (area
    # scales as length^2, hence sqrt) -- individual zone areas are then
    # rank-matched to the real per-zone distribution below, not left as
    # whatever the raw Voronoi tessellation happened to produce.
    raw_total_sqft = sum(p.area for p in polys)
    target_total_sqft = TOTAL_ACREAGE * 43_560.0
    scale = (target_total_sqft / raw_total_sqft) ** 0.5
    cx, cy = _ORIGIN_X + _BBOX_WIDTH_FT / 2, _ORIGIN_Y + _BBOX_HEIGHT_FT / 2
    polys = [shapely.affinity.scale(p, xfact=scale, yfact=scale, origin=(cx, cy)) for p in polys]

    # Spatial contiguity for the two fictional systems: split by centroid x,
    # not randomly -- a real hydraulic system boundary is spatially
    # contiguous, and the fictional dashboard map should read the same way.
    order_by_x = sorted(range(len(polys)), key=lambda i: polys[i].centroid.x)
    system_of = {}
    for rank, idx in enumerate(order_by_x):
        system_of[idx] = "system_a" if rank < SYSTEM_A_ZONE_COUNT else "system_b"

    names = list(ZONE_WORDS)
    rng.shuffle(names)
    names = [f"{ZONE_PREFIXES[i]}-{names[i % len(names)]}" for i in range(N_ZONES)]

    # Rank-match: the largest generated polygon gets the largest real
    # acreage magnitude, etc. Keeps the real skewed area distribution's
    # shape without tying any specific real zone's area to a specific
    # fictional one (assignment is by generated-size rank, not identity).
    acreages_desc = sorted(REAL_ACREAGES_SORTED, reverse=True)
    order_by_area = sorted(range(len(polys)), key=lambda i: polys[i].area, reverse=True)

    zones = []
    for rank, idx in enumerate(order_by_area):
        zones.append(FictionalZone(
            name=names[idx], system=system_of[idx], acreage=acreages_desc[rank],
            hgl_ft=int(rng.integers(HGL_MIN_FT, HGL_MAX_FT + 1) // 5 * 5),
            polygon=polys[idx],
        ))
    return zones


def generate_sites(zones: list[FictionalZone], seed: int = 20260723) -> list[FictionalSite]:
    rng = np.random.default_rng(seed + 1)
    sites = [FictionalSite(id="wtp_plant", kind="scada",
                           label=f"{FICTIONAL_TOWN} WTP -- SCADA finished-water feed",
                           role=None, beacon_account=None, meter_id=None, zone=None)]

    roles = [r for r, n in SITE_ROLE_COUNTS.items() for _ in range(n)]
    rng.shuffle(roles)
    used_accounts, used_meters = set(), set()

    def _unique(rng, low, high, used):
        while True:
            v = str(int(rng.integers(low, high)))
            if v not in used:
                used.add(v)
                return v

    for i, role in enumerate(roles, start=1):
        street = f"{rng.choice(STREET_WORDS)} {rng.choice(STREET_SUFFIXES)}"
        block = int(rng.integers(100, 9999))
        zone = zones[int(rng.integers(0, len(zones)))]
        account = _unique(rng, 1_000_000_000, 9_999_999_999, used_accounts)
        meter = _unique(rng, 10_000_000, 99_999_999, used_meters)
        sites.append(FictionalSite(
            id=f"site_{i:03d}", kind="beacon",
            label=f"{ROLE_LABELS[role]} Meter - {block} {street}, {FICTIONAL_TOWN}",
            role=role, beacon_account=account, meter_id=meter, zone=zone.name,
        ))
    return sites


def write_shapefile(zones: list[FictionalZone], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with shapefile.Writer(str(out_path), shapeType=shapefile.POLYGON) as w:
        w.field("PressureZo", "C", size=40)
        w.field("HGL", "N", size=10, decimal=0)
        w.field("Acreage", "N", size=12, decimal=2)
        for z in zones:
            rings = [list(z.polygon.exterior.coords)]
            w.poly(rings)
            w.record(PressureZo=z.name, HGL=z.hgl_ft, Acreage=round(z.acreage, 2))

    try:
        import pyproj
        wkt = pyproj.CRS.from_epsg(2232).to_wkt(version="WKT1_ESRI")
        out_path.with_suffix(".prj").write_text(wkt, encoding="utf-8")
    except Exception as e:  # pragma: no cover -- pyproj already a hard dependency elsewhere
        print(f"write_shapefile: could not write .prj ({e}) -- pressure_zones.py "
              f"will fall back to its own _FALLBACK_SOURCE_CRS")


def write_sites_yaml(sites: list[FictionalSite], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in sites:
        row = {"id": s.id, "kind": s.kind, "label": s.label}
        if s.role is not None:
            row["role"] = s.role
        if s.beacon_account is not None:
            row["beacon_account"] = s.beacon_account
        if s.meter_id is not None:
            row["meter_id"] = s.meter_id
        rows.append(row)
    import yaml
    out_path.write_text(yaml.safe_dump(rows, sort_keys=False), encoding="utf-8")


def main():
    repo_root = Path(__file__).parent.parent
    geo_dir = Path(__file__).parent / "geo"
    zones = generate_zones()
    sites = generate_sites(zones)

    write_shapefile(zones, repo_root / "state" / "gis" / "pressure_zones.shp")
    write_sites_yaml(sites, geo_dir / "sites.yaml")

    zone_summary = {
        "n_zones": len(zones),
        "system_a_label": SYSTEM_LABELS["system_a"], "system_a_count": SYSTEM_A_ZONE_COUNT,
        "system_b_label": SYSTEM_LABELS["system_b"], "system_b_count": SYSTEM_B_ZONE_COUNT,
        "total_acreage": round(sum(z.acreage for z in zones), 1),
        "zones": [{"name": z.name, "system": z.system, "acreage": z.acreage, "hgl_ft": z.hgl_ft}
                  for z in zones],
    }
    geo_dir.mkdir(parents=True, exist_ok=True)
    (geo_dir / "zone_summary.json").write_text(json.dumps(zone_summary, indent=2), encoding="utf-8")

    # Drop-in content for pressure_zones.py's ZONE_SYSTEM dict -- keyed by
    # the internal "system_a"/"system_b" labels the rest of the pipeline
    # (meter_zone_map.parquet, nrw.py) actually uses, not the display names.
    zone_system_map = {z.name: z.system for z in zones}
    (geo_dir / "zone_system_map.json").write_text(
        json.dumps(zone_system_map, indent=2), encoding="utf-8")

    print(f"Generated {len(zones)} fictional zones "
          f"({SYSTEM_A_ZONE_COUNT} {SYSTEM_LABELS['system_a']} / "
          f"{SYSTEM_B_ZONE_COUNT} {SYSTEM_LABELS['system_b']}), "
          f"total {zone_summary['total_acreage']:,.0f} acres")
    print(f"Generated {len(sites)} fictional sites ({len(sites) - 1} beacon + 1 scada)")
    print(f"Wrote: {repo_root / 'state' / 'gis'} and {geo_dir}")


if __name__ == "__main__":
    main()
