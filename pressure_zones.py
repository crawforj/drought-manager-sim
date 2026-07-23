"""Pressure zone geometry + hydraulic-system classification — dashboard map
panel.

Source: state/gis/pressure_zones.shp — a fabricated, structurally-analogous
service-area geometry (fictional zone count/area distribution/system split,
placed at a fictional Colorado Front Range location), NOT derived from any
real utility's GIS data. CRS NAD83 StatePlane Colorado Central ftUS
(EPSG:2232), reprojected to lon/lat for the dashboard's OSM basemap. See
this repo's README for how the fictional geometry was generated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pyproj
import shapefile

# Zone (PressureZo field) -> hydraulic system. Anything not listed here
# degrades to UNCLASSIFIED rather than crashing or guessing.
ZONE_SYSTEM: dict[str, str] = {
    "A-Bluestem": "system_a",
    "B-Foxglove": "system_a",
    "C-Larkspur": "system_a",
    "E-Pinyon": "system_a",
    "F-Juniper": "system_a",
    "D-Umber": "system_b",
    "G-Obsidian": "system_b",
    "H-Granite": "system_b",
    "I-Vireo": "system_b",
    "J-Sandhill": "system_b",
    "K-Talus": "system_b",
    "L-Windrow": "system_b",
    "M-Wildrose": "system_b",
    "N-Nighthawk": "system_b",
    "O-Antler": "system_b",
    "P-Limestone": "system_b",
    "Q-Harrier": "system_b",
    "R-Ironwood": "system_b",
    "S-Kestrel": "system_b",
    "T-Cinder": "system_b",
}
UNCLASSIFIED = "unclassified"

_TARGET_CRS = "EPSG:4326"
_FALLBACK_SOURCE_CRS = "EPSG:2232"   # used only if the .prj sidecar is missing


@dataclass
class ZonePart:
    """One renderable ring (ordered lon/lat point list)."""
    lons: list[float]
    lats: list[float]


@dataclass
class Zone:
    name: str                              # display name; "(unnamed zone)" for the blank record
    raw_name: str                          # exact shapefile PressureZo value, "" for blank
    system: str                            # "system_a" | "system_b" | "unclassified"
    hgl_ft: int | None
    acreage: int
    parts: list[ZonePart] = field(default_factory=list)
    n_unattributed_parts: int = 0          # placeholder sub-records folded in


def load_zones(shp_path: Path) -> list[Zone]:
    """Parse the pressure-zone shapefile into display-ready Zone objects.

    Groups shapefile records by PressureZo name (a zone can span multiple
    records/disjoint geometry), splits each record's rings via shape.parts,
    and reprojects every point from the shapefile's own CRS (read live from
    the .prj sidecar, not hardcoded, so this stays correct if the file is
    ever regenerated) to lon/lat for the dashboard's OSM basemap.

    A record counts as a placeholder (real ring geometry, zeroed-out bogus
    attributes) only when its zone has MORE than one record -- a
    single-record zone's 0 acreage is real data, not a placeholder, and
    must display as-is. Placeholder rings still render (never dropped),
    just excluded from the acreage/HGL rollup and counted in
    n_unattributed_parts so the caller can flag it honestly.
    """
    shp_path = Path(shp_path)
    prj_path = shp_path.with_suffix(".prj")
    source_crs = pyproj.CRS.from_wkt(prj_path.read_text()) if prj_path.exists() else _FALLBACK_SOURCE_CRS
    transformer = pyproj.Transformer.from_crs(source_crs, _TARGET_CRS, always_xy=True)

    sf = shapefile.Reader(str(shp_path))

    groups: dict[str, list] = {}
    for sr in sf.shapeRecords():
        rec = sr.record.as_dict()
        name = (rec.get("PressureZo") or "").strip()
        groups.setdefault(name, []).append((rec, sr.shape))

    zones: list[Zone] = []
    for raw_name, records in groups.items():
        display_name = raw_name if raw_name else "(unnamed zone)"
        system = ZONE_SYSTEM.get(raw_name, UNCLASSIFIED)

        is_multi_record = len(records) > 1
        attributed = [
            (r, s) for r, s in records
            if not (is_multi_record and not r.get("HGL") and not r.get("Acreage"))
        ]
        placeholder_count = len(records) - len(attributed)

        acreage = sum(int(r.get("Acreage") or 0) for r, _ in attributed)
        hgl = next((int(r["HGL"]) for r, _ in attributed if r.get("HGL")), None)

        parts: list[ZonePart] = []
        for _, shape in records:   # every ring for every record, placeholder or not
            pts = shape.points
            if not pts:
                continue
            ring_bounds = list(shape.parts) + [len(pts)]
            for i in range(len(ring_bounds) - 1):
                ring = pts[ring_bounds[i]:ring_bounds[i + 1]]
                if not ring:
                    continue
                lons, lats = [], []
                for x, y in ring:
                    lon, lat = transformer.transform(x, y)
                    lons.append(lon)
                    lats.append(lat)
                parts.append(ZonePart(lons=lons, lats=lats))

        zones.append(Zone(
            name=display_name, raw_name=raw_name, system=system,
            hgl_ft=hgl, acreage=acreage, parts=parts,
            n_unattributed_parts=placeholder_count,
        ))

    return sorted(zones, key=lambda z: z.name)


def _self_test() -> None:
    here = Path(__file__).parent
    zones = load_zones(here / "state" / "gis" / "pressure_zones.shp")
    print(f"{len(zones)} zones loaded")
    counts: dict[str, int] = {}
    for z in zones:
        counts[z.system] = counts.get(z.system, 0) + 1
    print("per-system counts:", counts)
    assert len(zones) == 20, f"expected 20 zones, got {len(zones)}"
    assert counts == {"system_a": 5, "system_b": 15}, counts

    n_points = 0
    for z in zones:
        for part in z.parts:
            for lon, lat in zip(part.lons, part.lats):
                assert -105.5 < lon < -104.0, f"{z.name}: lon {lon} outside the fictional service area"
                assert 39.0 < lat < 40.5, f"{z.name}: lat {lat} outside the fictional service area"
                n_points += 1
    print(f"{n_points} points, all within the fictional service area's bounds")

    print("SELF-TEST PASSED.")


if __name__ == "__main__":
    _self_test()
