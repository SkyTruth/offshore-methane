# %%
"""Deterministic development records; never alters the production segments.

Run ``main(nights=1500)`` or ``main(nights=5000)`` after preserving the real
pre-change site. The current calendar year stays real; older years repeat
the real record at the same month/day, leaving archive gaps unobserved.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import resource
import time
from pathlib import Path

import build_viewer
import numpy as np
import pandas as pd


def read_page(path: Path) -> dict:
    """Read an embedded pre-change payload, or round-trip a chunked build."""
    match = re.search(
        r'<script type="application/json" id="flaring-data">(.*?)</script>',
        path.read_text(),
        re.DOTALL,
    )
    if not match:
        raise ValueError(f"no flaring data in {path}")
    data = json.loads(match[1].replace("<\\/", "</"))
    if data["meta"].get("chunks"):
        rows = {r[0]: r for r in data["sites"]}
        for row in rows.values():
            row[4], row[5], row[14] = "", [], ""
        for name in data["meta"].pop("chunks"):
            chunk = json.loads((path.parent / name).read_text())
            present = {r[0]: r for r in chunk["sites"]}
            for sid, row in rows.items():
                _, st, rv = present.get(sid, [sid, "N" * chunk["nights"], []])
                row[4] += st
                row[5].extend(rv)
        pass_chunks = [
            json.loads((path.parent / name).read_text())
            for name in data["meta"].pop("clearpasses")
        ]
        for chunk in sorted(pass_chunks, key=lambda chunk: chunk["start"]):
            present = dict(chunk["sites"])
            for sid, row in rows.items():
                row[14] += present.get(sid, "0" * chunk["nights"])
        data.pop("totals", None)
    return data


def synthetic_data(real: dict, nights: int) -> dict:
    """Repeat whole calendar years with seeded per-site/year start/stop changes."""
    if nights < len(real["nights"]):
        raise ValueError("synthetic record must include the real period")
    data = copy.deepcopy(real)
    dates = pd.date_range(end=real["nights"][-1], periods=nights)
    target = dates.strftime("%Y-%m-%d").tolist()
    current_year = real["nights"][-1][:4]
    lookup = {
        d[5:]: i for i, d in enumerate(real["nights"]) if d.startswith(current_year)
    }
    source_indices = [lookup.get(d[5:]) for d in target]
    years = sorted({d[:4] for d in target})
    spans = {
        year: [i for i, d in enumerate(target) if d.startswith(year)] for year in years
    }
    for row in data["sites"]:
        original_st, original_cp = row[4], row[14]
        heat, _ = build_viewer.decode_nights(row)
        st, rv, cp, values = [], [], [], []
        for year, indices in spans.items():
            seed = int(
                hashlib.sha256(f"D66:{row[0]}:{year}".encode()).hexdigest()[:16], 16
            )
            rng = np.random.default_rng(seed)
            factor, dark = (
                (1.0, False)
                if year == current_year
                else (rng.uniform(0.6, 1.4), rng.random() < 0.15)
            )
            for i in indices:
                src = source_indices[i]
                ch = original_st[src] if src is not None else "N"
                value = heat[src] if src is not None else np.nan
                if dark and ch == "F":
                    ch, value = "C", 0.0
                value = build_viewer._r(value * factor)
                st.append(ch)
                if ch in "FD":
                    rv.append(value)
                if value is not None:
                    values.append(value)
                if original_cp:
                    cp.append(original_cp[src] if src is not None else "0")
        row[4], row[5] = "".join(st), rv
        row[7] = build_viewer._r(np.mean(values)) if values else None
        row[14] = "".join(cp) if original_cp else None
    data["nights"] = target
    data["meta"].update(start=target[0], synthetic=True, pass_start=real["nights"][0])
    return data


def main(nights: int = 1500, real_dir: Path | None = None) -> Path:
    """Build ``outputs/site_long_<nights>/`` and write resource measurements."""
    start = time.perf_counter()
    real_dir = real_dir or build_viewer.OUT_DIR / "site"
    data = synthetic_data(read_page(real_dir / "index.html"), nights)
    target = build_viewer.OUT_DIR / f"site_long_{nights}"
    build_viewer.main(data=data, evidence_dir=real_dir, site_dir=target, skip_emit=True)
    # macOS ru_maxrss is bytes; Linux is KiB.
    import sys

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "nights": nights,
        "sites": len(data["sites"]),
        "seconds": round(time.perf_counter() - start, 2),
        "peak_memory_mb": round(rss / (1e6 if sys.platform == "darwin" else 1e3), 1),
        "index_bytes": (target / "index.html").stat().st_size,
        "chunks": {
            name: (target / name).stat().st_size
            for name in read_manifest(target)["meta"]["chunks"]
        },
        "clearpasses": {
            name: (target / name).stat().st_size
            for name in read_manifest(target)["meta"]["clearpasses"]
        },
    }
    (target / "build_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return target


def read_manifest(site_dir: Path) -> dict:
    """Read the embedded page manifest without expanding site histories."""
    match = re.search(
        r'<script type="application/json" id="flaring-data">(.*?)</script>',
        (site_dir / "index.html").read_text(),
        re.DOTALL,
    )
    return json.loads(match[1].replace("<\\/", "</"))


# %%
if __name__ == "__main__":
    main()
