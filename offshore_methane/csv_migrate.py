"""
csv_migrate.py

One-off helpers to migrate existing CSVs to the new schema:

- event_granule.csv  → process_runs.csv (adds git_hash, last_timestamp)
- events.csv         → structures.csv + windows.csv (normalized split)
- granules.csv       → add git_hash column if missing

Non-destructive by default: writes new files; old files remain unless
delete_old=True is passed. Sorts rows deterministically before writing.

Usage (from repo root or notebooks):

    from offshore_methane.csv_migrate import migrate_all
    migrate_all(delete_old=False)

Or as a module:

    python -m offshore_methane.csv_migrate

"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import os
import pandas as pd

import offshore_methane.config as cfg
from offshore_methane.csv_utils import (
    _atomic_write_csv,
    _read_csv,
    current_git_hash,
)


# -----------------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------------
def _write_sorted(df: pd.DataFrame, path: Path, sort_by: list[str], *, float_fmt: Optional[str] = None) -> None:
    if not df.empty and sort_by:
        for col in sort_by:
            if col not in df.columns:
                df[col] = pd.NA
        df = df.sort_values(sort_by).reset_index(drop=True)
    _atomic_write_csv(df, path, float_format=float_fmt)


# -----------------------------------------------------------------------------
#  Migrations
# -----------------------------------------------------------------------------
def migrate_event_granule_to_process_runs(delete_old: bool = False) -> Tuple[bool, Path]:
    """Create process_runs.csv from event_granule.csv (if present).

    Adds git_hash (current HEAD, best-effort) and an empty last_timestamp.
    Returns (changed, output_path).
    """
    src = cfg.DATA_DIR / "event_granule.csv"
    dst = cfg.PROCESS_RUNS_CSV
    if not src.is_file():
        return False, dst

    df = _read_csv(src).copy()
    if df.empty:
        # Write an empty file with headers (modern schema)
        out = pd.DataFrame(columns=["window_id", "system_index", "git_hash", "last_timestamp"])
        _write_sorted(out, dst, ["window_id", "system_index"])
        if delete_old:
            try:
                os.remove(src)
            except OSError:
                pass
        return True, dst

    # Ensure required columns
    if "event_id" not in df.columns:
        df["event_id"] = pd.NA
    if "system_index" not in df.columns:
        df["system_index"] = ""
    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce").astype("Int64")
    df["system_index"] = df["system_index"].astype("string")

    gh = current_git_hash()
    df["git_hash"] = gh
    if "last_timestamp" not in df.columns:
        df["last_timestamp"] = ""

    # Normalize to modern window_id column and de-duplicate pairs
    df = df.rename(columns={"event_id": "window_id"})
    df = df.drop_duplicates(subset=["window_id", "system_index"], keep="first")
    _write_sorted(
        df[["window_id", "system_index", "git_hash", "last_timestamp"]],
        dst,
        ["window_id", "system_index"],
    )

    if delete_old:
        try:
            os.remove(src)
        except OSError:
            pass
    return True, dst


def migrate_granules_add_git_hash() -> Tuple[bool, Path]:
    """Ensure granules.csv has a git_hash column. Returns (changed, path)."""
    path = cfg.GRANULES_CSV
    if not path.is_file():
        return False, path
    df = _read_csv(path).copy()
    if df.empty:
        return False, path
    if "git_hash" in df.columns:
        return False, path
    df["git_hash"] = ""
    _write_sorted(df, path, ["system_index"], float_fmt="%.4f")
    return True, path


def migrate_process_runs_event_to_window() -> bool:
    """Normalize process_runs.csv to use window_id only (drop legacy event_id).

    Returns True if a change was made.
    """
    path = cfg.PROCESS_RUNS_CSV
    if not path.is_file():
        return False
    df = _read_csv(path).copy()
    if df.empty:
        return False
    changed = False
    if "window_id" not in df.columns and "event_id" in df.columns:
        df = df.rename(columns={"event_id": "window_id"})
        changed = True
    # Drop legacy mirror if present
    if "event_id" in df.columns:
        df = df.drop(columns=["event_id"])
        changed = True or changed
    if changed:
        _write_sorted(df, path, ["window_id", "system_index"])
    return changed


def migrate_move_local_medians_to_runs() -> bool:
    """Move sga_local_median/sgi_median from granules.csv → process_runs.csv.

    For each process_runs row, if granules.csv has local medians for that
    system_index, copy them into process_runs. Then remove these columns from
    granules.csv. Returns True if any change was made.
    """
    gpath = cfg.GRANULES_CSV
    rpath = cfg.PROCESS_RUNS_CSV
    if not gpath.is_file() or not rpath.is_file():
        return False
    g = _read_csv(gpath).copy()
    r = _read_csv(rpath).copy()
    if g.empty or r.empty:
        return False

    changed = False
    have_cols = [c for c in ("sga_local_median", "sgi_median") if c in g.columns]
    if not have_cols:
        return False

    # Build lookup by system_index
    g["system_index"] = g["system_index"].astype("string")
    look = g.set_index("system_index")
    r["system_index"] = r["system_index"].astype("string")
    for col in have_cols:
        vals = look[col] if col in look.columns else None
        if vals is None:
            continue
        before = r.get(col)
        r[col] = r["system_index"].map(vals)
        if before is None:
            if r[col].notna().any():
                changed = True
        else:
            changed = bool(changed or (before.fillna(0) != r[col].fillna(0)).any())

    # Ensure window_id exists for deterministic sorting
    if "window_id" not in r.columns and "event_id" in r.columns:
        r = r.rename(columns={"event_id": "window_id"})
        changed = True or changed
    if changed:
        _write_sorted(r, rpath, ["window_id", "system_index"])

    # Remove columns from granules
    kept_cols = [c for c in g.columns if c not in ("sga_local_median", "sgi_median")]
    if len(kept_cols) < len(g.columns):
        g = g[kept_cols]
        _write_sorted(g, gpath, ["system_index"], float_fmt="%.4f")
        changed = True

    return changed


def _sanitize_nan_strings(df: pd.DataFrame, text_cols: list[str]) -> tuple[pd.DataFrame, bool]:
    changed = False
    for col in text_cols:
        if col not in df.columns:
            continue
        # Work on a copy of the column coerced to pandas string dtype
        s = df[col].astype("string")
        before = s.copy()
        s = s.replace({"nan": pd.NA, "NaN": pd.NA, "None": pd.NA})
        if not before.equals(s):
            df[col] = s
            changed = True
    return df, changed


def migrate_sanitize_nan_strings() -> bool:
    """Replace literal 'nan'/'NaN'/'None' strings with blanks in CSVs.

    Applies to known textual columns in granules.csv, process_runs.csv,
    windows.csv and structures.csv.
    Returns True if any file was changed.
    """
    changed_any = False
    # process_runs.csv
    pr = _read_csv(cfg.PROCESS_RUNS_CSV).copy()
    if not pr.empty:
        pr2, ch = _sanitize_nan_strings(pr, ["system_index", "git_hash", "last_timestamp"])
        if ch:
            _atomic_write_csv(pr2, cfg.PROCESS_RUNS_CSV)
            changed_any = True
    # granules.csv
    g = _read_csv(cfg.GRANULES_CSV).copy()
    if not g.empty:
        g2, ch = _sanitize_nan_strings(g, ["system_index", "git_hash", "timestamp"])
        if ch:
            _atomic_write_csv(g2, cfg.GRANULES_CSV, float_format="%.4f")
            changed_any = True
    # windows.csv
    w = _read_csv(cfg.WINDOWS_CSV).copy()
    if not w.empty:
        cols = [c for c in ["structure_id", "start", "end"] if c in w.columns]
        w2, ch = _sanitize_nan_strings(w, cols)
        if ch:
            _atomic_write_csv(w2, cfg.WINDOWS_CSV)
            changed_any = True
    # structures.csv
    s = _read_csv(cfg.STRUCTURES_CSV).copy()
    if not s.empty:
        s2, ch = _sanitize_nan_strings(s, ["structure_id", "name", "country"]) if any(
            c in s.columns for c in ("structure_id", "name", "country")
        ) else (s, False)
        if ch:
            _atomic_write_csv(s2, cfg.STRUCTURES_CSV)
            changed_any = True
    return changed_any


def migrate_events_to_structures_and_windows(delete_old: bool = False) -> Tuple[bool, Path, Path]:
    """Split events.csv into structures.csv + windows.csv.

    - structures.csv: structure_id, lon, lat, [name]
    - windows.csv:    id, structure_id, start, end, plus any extra columns from events.csv except lon/lat

    Returns (changed, structures_path, windows_path). If events.csv is missing
    or both target files already exist, returns (False, ...).
    """
    src = cfg.DATA_DIR / "events.csv"
    if not src.is_file():
        return False, cfg.STRUCTURES_CSV, cfg.WINDOWS_CSV
    if cfg.STRUCTURES_CSV.is_file() and cfg.WINDOWS_CSV.is_file():
        return False, cfg.STRUCTURES_CSV, cfg.WINDOWS_CSV

    ev = _read_csv(src).copy()
    if ev.empty:
        # write empty shells
        _write_sorted(pd.DataFrame(columns=["structure_id", "lon", "lat"]), cfg.STRUCTURES_CSV, ["structure_id"])
        _write_sorted(
            pd.DataFrame(columns=["id", "structure_id", "start", "end", "flare_lat", "flare_lon"]),
            cfg.WINDOWS_CSV,
            ["id"],
        )
        if delete_old:
            try:
                os.remove(src)
            except OSError:
                pass
        return True, cfg.STRUCTURES_CSV, cfg.WINDOWS_CSV

    # Normalize required columns
    # Some files may not have id; if missing, use row index as id
    if "id" not in ev.columns:
        ev["id"] = ev.index
    for col in ("lon", "lat"):
        if col not in ev.columns:
            raise ValueError(f"events.csv missing required column: {col}")
    for c in ("id",):
        ev[c] = pd.to_numeric(ev[c], errors="coerce").astype("Int64")
    for c in ("lon", "lat"):
        ev[c] = pd.to_numeric(ev[c], errors="coerce")
    # Keep dates as strings
    for c in ("start", "end"):
        if c in ev.columns:
            ev[c] = ev[c].astype("string")

    # Assign structure_id per unique (lon, lat) as serial strings: x1, x2, ...
    key_cols = ["lon", "lat"]
    # sort unique locations for deterministic IDs
    unique_locs = (
        ev[key_cols]
        .drop_duplicates()
        .sort_values(key_cols)
        .reset_index(drop=True)
    )
    struct_rows = []
    loc_to_struct: dict[tuple, str] = {}
    for idx, row in unique_locs.iterrows():
        lon = row["lon"]
        lat = row["lat"]
        sid = f"x{idx+1}"
        loc_to_struct[(lon, lat)] = sid
        # Optionally carry a name/label if present and consistent (first seen)
        sub = ev[(ev["lon"] == lon) & (ev["lat"] == lat)]
        name = None
        for col in ("name", "label"):
            if col in sub.columns and sub[col].notna().any():
                name = str(sub[col].dropna().iloc[0])
                break
        srow = {"structure_id": sid, "lon": lon, "lat": lat}
        if name is not None:
            srow["name"] = name
        struct_rows.append(srow)

    structures = pd.DataFrame(struct_rows)
    _write_sorted(structures, cfg.STRUCTURES_CSV, ["structure_id"])

    # Windows: move id/start/end and link to structure_id; keep extra columns except lon/lat
    windows_cols = ["id", "start", "end"]
    extra_cols = [c for c in ev.columns if c not in (set(windows_cols) | {"lon", "lat"})]
    win_rows = []
    for _, row in ev.iterrows():
        lon, lat = row["lon"], row["lat"]
        sid = loc_to_struct[(lon, lat)]
        item = {"id": int(row["id"]) if pd.notna(row["id"]) else None, "structure_id": sid}
        for c in ("start", "end"):
            if c in ev.columns:
                item[c] = str(row.get(c, ""))
        for c in extra_cols:
            item[c] = row.get(c)
        win_rows.append(item)

    windows = pd.DataFrame(win_rows)
    # Ensure flare_lat/flare_lon columns exist for downstream code
    for col in ("flare_lat", "flare_lon"):
        if col not in windows.columns:
            windows[col] = pd.NA
    _write_sorted(windows, cfg.WINDOWS_CSV, ["id"]) 

    if delete_old:
        try:
            os.remove(src)
        except OSError:
            pass

    return True, cfg.STRUCTURES_CSV, cfg.WINDOWS_CSV


def migrate_all(delete_old: bool = False) -> None:
    """Run all migrations. Prints a short summary."""
    ch1, dst1 = migrate_event_granule_to_process_runs(delete_old)
    # Ensure process_runs uses window_id only (drop legacy event_id)
    ch1b = migrate_process_runs_event_to_window()
    ch2, dst2 = migrate_granules_add_git_hash()
    ch2b = migrate_move_local_medians_to_runs()
    ch3, dst3a, dst3b = migrate_events_to_structures_and_windows(delete_old)
    ch4 = migrate_sanitize_nan_strings()

    print("Migration summary:")
    print(f"  process_runs.csv: {'created/updated' if ch1 else 'no change'} → {dst1}")
    if ch1b:
        print("  process_runs.csv: normalized column event_id → window_id")
    print(f"  granules.csv:     {'git_hash added' if ch2 else 'no change'} → {dst2}")
    if ch2b:
        print("  moved local medians (SGA/SGI) to process_runs.csv and removed from granules.csv")
    print(
        f"  structures/windows: {'created' if ch3 else 'no change'} → {dst3a}, {dst3b}"
    )
    if ch4:
        print("  sanitized 'nan' string literals to blanks in CSVs")
    # Ensure windows.csv has a proper header row
    if ensure_windows_header():
        print("  windows.csv: inserted header row [id,structure_id,start,end,citation,country]")


def main():
    # Default non-destructive run
    migrate_all(delete_old=False)

    # Optional consolidation step: merge structures within ±0.005°
    try:
        merged, updated = merge_nearby_structures(0.005)
        if merged or updated:
            print(f"Merged {merged} structure group(s); updated {updated} window row(s).")
    except Exception as e:
        print(f"Structure merge skipped due to error: {e}")

    # Promote EEZ (country) from windows.csv to structures.csv
    try:
        changed = add_country_to_structures_from_windows()
        if changed:
            print(f"Added/updated country for {changed} structure(s).")
    except Exception as e:
        print(f"Country promotion skipped due to error: {e}")

    # Ensure windows has header after any merges
    try:
        if ensure_windows_header():
            print("Ensured windows.csv header present.")
    except Exception:
        pass


def ensure_windows_header() -> bool:
    """If windows.csv is missing a header, add one inferred from row layout.

    Assumes current layout: id, structure_id, start, end, citation, country.
    Returns True if a header was added.
    """
    path = cfg.WINDOWS_CSV
    if not path.is_file():
        return False
    # Peek first line
    with open(path, "r", newline="") as f:
        first = f.readline().strip()
        if not first:
            return False
        # If looks like an existing header (contains expected names), skip
        if first.lower().startswith("id,") or "structure_id" in first.lower():
            return False
        rest = f.read()
    header = "id,structure_id,start,end,citation,country\n"
    tmp = path.with_suffix(path.suffix + ".tmpheader")
    with open(tmp, "w", newline="") as out:
        out.write(header)
        out.write(first + "\n")
        out.write(rest)
    os.replace(tmp, path)
    return True


def _sid_numeric_order(s: str) -> tuple:
    s = str(s)
    if s.startswith("x") and s[1:].isdigit():
        return (0, int(s[1:]))
    # Fallback: try int
    try:
        return (1, int(s))
    except Exception:
        return (2, s)


def merge_nearby_structures(tol: float = 0.005) -> tuple[int, int]:
    """Merge structures whose lon and lat differ by ≤ tol (±tol degrees).

    Clustering uses connected components under the L∞ metric (Chebyshev):
      two nodes are adjacent if max(|Δlon|, |Δlat|) ≤ tol.

    - Canonical structure_id = minimum by numeric order (x1 < x2 < x10 < ...).
    - Updates windows.csv structure_id to canonical.
    - Keeps only one row per component in structures.csv (canonical row).

    Returns (merged_components_count, updated_windows_rows).
    """
    s_path = cfg.STRUCTURES_CSV
    w_path = cfg.WINDOWS_CSV
    if not s_path.is_file() or not w_path.is_file():
        return (0, 0)

    sdf = _read_csv(s_path).copy()
    wdf = _read_csv(w_path).copy()
    if sdf.empty:
        return (0, 0)

    # Ensure types
    sdf["structure_id"] = sdf["structure_id"].astype("string")
    sdf["lon"] = pd.to_numeric(sdf["lon"], errors="coerce")
    sdf["lat"] = pd.to_numeric(sdf["lat"], errors="coerce")
    if "structure_id" in wdf.columns:
        wdf["structure_id"] = wdf["structure_id"].astype("string")

    # Build adjacency via naive O(n^2) and union-find components
    ids = sdf["structure_id"].astype(str).tolist()
    lons = sdf["lon"].astype(float).tolist()
    lats = sdf["lat"].astype(float).tolist()
    n = len(ids)

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Union when both lon and lat are within tol
    for i in range(n):
        for j in range(i + 1, n):
            if abs(lons[i] - lons[j]) <= tol and abs(lats[i] - lats[j]) <= tol:
                union(i, j)

    # Collect components
    comp: dict[int, list[int]] = {}
    for i in range(n):
        r = find(i)
        comp.setdefault(r, []).append(i)

    mappings: dict[str, str] = {}
    keep_ids: set[str] = set()
    merged_groups = 0
    for members in comp.values():
        if len(members) <= 1:
            keep_ids.add(ids[members[0]])
            continue
        # choose canonical by smallest x-number order
        member_ids = [ids[k] for k in members]
        member_ids.sort(key=_sid_numeric_order)
        canonical = member_ids[0]
        keep_ids.add(canonical)
        for mid in member_ids[1:]:
            mappings[mid] = canonical
        merged_groups += 1

    if not mappings:
        return (0, 0)

    # Apply mapping to windows
    before = wdf["structure_id"].copy()
    wdf["structure_id"] = wdf["structure_id"].replace(mappings)
    updated_windows = int((before != wdf["structure_id"]).sum())

    # Reduce structures to canonical rows only, preferring canonical rows from original
    sdf = sdf[sdf["structure_id"].isin(list(keep_ids))].copy()
    _write_sorted(sdf, s_path, ["structure_id"])
    _write_sorted(wdf, w_path, ["id"])  # keep id ordering

    return (merged_groups, updated_windows)


def add_country_to_structures_from_windows(
    *, source_col: str = "EEZ", target_col: str = "country"
) -> int:
    """Copy EEZ/country from windows.csv to structures.csv.

    Strategy: for each structure_id, choose the most frequent non-empty value
    from windows[source_col]. On ties, choose lexicographically smallest.
    Writes/updates structures[target_col]. Returns number of structures changed.
    """
    s_path = cfg.STRUCTURES_CSV
    w_path = cfg.WINDOWS_CSV
    if not s_path.is_file() or not w_path.is_file():
        return 0
    sdf = _read_csv(s_path).copy()
    wdf = _read_csv(w_path).copy()
    if sdf.empty or wdf.empty or source_col not in wdf.columns:
        return 0

    # Build mapping structure_id -> country
    w = wdf[["structure_id", source_col]].copy()
    w["structure_id"] = w["structure_id"].astype("string")
    w[source_col] = w[source_col].astype("string").str.strip()
    w = w[w[source_col].notna() & (w[source_col].str.len() > 0)]
    if w.empty:
        return 0

    mapping = {}
    for sid, grp in w.groupby("structure_id"):
        vals = grp[source_col].value_counts()
        if vals.empty:
            continue
        max_count = vals.max()
        candidates = sorted([k for k, v in vals.items() if v == max_count])
        mapping[sid] = candidates[0]

    if not mapping:
        return 0

    sdf["structure_id"] = sdf["structure_id"].astype("string")
    before = sdf.get(target_col)
    sdf[target_col] = [mapping.get(sid, (before[i] if before is not None else None)) for i, sid in enumerate(sdf["structure_id"]) ]

    # Count changes (new or different values)
    changed = 0
    if before is None:
        changed = int(sdf[target_col].notna().sum())
    else:
        b = before.fillna("").astype(str)
        a = sdf[target_col].fillna("").astype(str)
        changed = int((a != b).sum())

    _write_sorted(sdf, s_path, ["structure_id"])
    return changed


if __name__ == "__main__":
    main()
