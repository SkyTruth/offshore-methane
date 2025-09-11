"""
csv_utils.py

Utilities for treating the CSV files in ../data as a small, joined
"virtual database". The join model is evolving towards a clearer split:

  structures.csv  (rows keyed by structure_id with lon/lat)
    ⨝  windows.csv    (time windows keyed by id, referencing structure_id)
         ⇣ (compat) load_events() still returns id/lon/lat/start/end

  process_runs.csv  (many-to-many via window_id ↔ system_index) + run metadata
    ⨝  granules.csv     (rows keyed by system_index, now with git_hash)

Backward-compatibility: if structures/windows are not present, `events.csv`
is used for inputs. All run mappings are stored in `process_runs.csv`.

Core capabilities:
- Load events (from structures/windows or legacy events), runs, and granules.
- Produce joined DataFrames filtered by scene/local quality controls.
- Query by event id (eid) or by granule id (gid, aka system:index).
- Simple upsert helpers to create/update rows in each CSV.

Filtering rules (kwargs):
- scene_cloud_pct: defaults to cfg.MASK_PARAMS["cloud"]["scene_cloud_pct"].
  Applied to column "cloudiness"; rows with missing values are dropped when
  this filter is active.
- scene_sga_range: defaults to cfg.MASK_PARAMS["sunglint"]["scene_sga_range"].
  Applied to column "sga_scene" using strict inequality (min < x < max);
  missing values are dropped when active.
- local_sga_range: default None; when provided, applied to column
  "sga_local_median" using strict inequality; rows with missing values are
  dropped when active.
- local_sgi_range: default None; when provided, applied to column
  "sgi_median" using strict inequality; rows with missing values are dropped
  when active.

Notes
-----
- Column synonyms are normalised on load: sga_local_med → sga_local_median,
  sgi_local_med → sgi_median.
- The join preserves all window rows that appear in process_runs.csv. Empty system_index entries (used to mark
  "no granules") are removed in joined queries by default because they do not
  map to granules.
"""

from __future__ import annotations

import os
import re
import subprocess
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd

import offshore_methane.config as cfg

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# Standard CSV locations relative to DATA_DIR
DATA_DIR: Path = cfg.DATA_DIR
EVENTS_CSV: Path = DATA_DIR / "events.csv"  # legacy input (fallback only)
GRANULES_CSV: Path = DATA_DIR / "granules.csv"
PROCESS_RUNS_CSV: Path = DATA_DIR / "process_runs.csv"
STRUCTURES_CSV: Path = DATA_DIR / "structures.csv"
WINDOWS_CSV: Path = DATA_DIR / "windows.csv"


# -----------------------------------------------------------------------------
# Loader helpers
# -----------------------------------------------------------------------------
def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError as e:
        # Gracefully recover from prior corrupted writes by skipping bad lines
        warnings.warn(
            f"Parsing error in {path}. Skipping malformed lines. Detail: {e}",
            RuntimeWarning,
        )
        return pd.read_csv(path, on_bad_lines="skip")


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def current_git_hash(short: bool = True) -> str:
    """Return the current repository git hash (best-effort).

    Falls back to env var GIT_COMMIT, then "unknown" if not available.
    """
    try:
        args = (
            ["git", "rev-parse", "--short", "HEAD"]
            if short
            else [
                "git",
                "rev-parse",
                "HEAD",
            ]
        )
        out = subprocess.run(args, capture_output=True, text=True, check=True)
        h = out.stdout.strip()
        if h:
            return h
    except Exception:
        pass
    return os.getenv("GIT_COMMIT", "unknown")


# -----------------------------------------------------------------------------
# File locking + atomic write helpers (prevent concurrent write corruption)
# -----------------------------------------------------------------------------
@contextmanager
def _exclusive_lock(lock_path: Path):
    """
    Cross‑platform best‑effort exclusive lock using a sidecar .lock file.

    On POSIX, uses fcntl.flock; on other platforms, falls back to simple
    open/close (still serializes within a single process).
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lockf:
        try:
            import fcntl  # type: ignore

            fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        except Exception:
            pass
        try:
            yield
        finally:
            try:
                import fcntl  # type: ignore

                fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


def _atomic_write_csv(df: pd.DataFrame, path: Path, *, float_format: str | None = None):
    """
    Write CSV atomically (tmp + replace) under an exclusive lock to prevent
    interleaved lines when multiple threads/processes update concurrently.
    """
    lock = path.with_suffix(path.suffix + ".lock")
    with _exclusive_lock(lock):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(tmp, index=False, float_format=float_format)
        os.replace(tmp, path)


def load_events() -> pd.DataFrame:
    """Load events as id/lon/lat/start/end, supporting both legacy and split schema.

    Preference:
      1) If both structures.csv and windows.csv exist, join them.
      2) Else, fall back to events.csv as-is.
    """
    # Normalized split present → build a legacy-like view
    if STRUCTURES_CSV.is_file() and WINDOWS_CSV.is_file():
        s = _read_csv(STRUCTURES_CSV).copy()
        w = _read_csv(WINDOWS_CSV).copy()
        if s.empty or w.empty:
            return pd.DataFrame()
        # Normalise header case/whitespace and common aliases
        s.columns = [str(c).strip().lower() for c in s.columns]
        w.columns = [str(c).strip().lower() for c in w.columns]

        # Coordinate aliases for both tables (token-based to avoid false positives)
        def _apply_coord_aliases(df):
            cols = list(df.columns)
            cols_set = set(cols)

            def tokens(name: str) -> set[str]:
                return set(t for t in re.split(r"[^a-z0-9]+", name) if t)

            if "lon" not in cols_set:
                for c in cols:
                    ts = tokens(c)
                    if ts & {
                        "lon",
                        "long",
                        "longitude",
                        "lon_dd",
                        "long_dd",
                        "longitude_dd",
                        "x",
                        "lon_wgs84",
                        "long_wgs84",
                    }:
                        df.rename(columns={c: "lon"}, inplace=True)
                        cols_set.add("lon")
                        break
            if "lat" not in cols_set:
                for c in cols:
                    ts = tokens(c)
                    if ts & {
                        "lat",
                        "latitude",
                        "lat_dd",
                        "latitude_dd",
                        "y",
                        "lat_wgs84",
                    }:
                        df.rename(columns={c: "lat"}, inplace=True)
                        cols_set.add("lat")
                        break
            return df

        s = _apply_coord_aliases(s)
        w = _apply_coord_aliases(w)
        # Window id/date aliases
        w = w.rename(
            columns={"window_id": "id", "start_date": "start", "end_date": "end"}
        )

        # Headerless detection: if column names look like values (dates/http/x###), re-read with header=None
        def _looks_like_value(name: str) -> bool:
            return bool(
                re.match(r"^\d{4}-\d{2}-\d{2}$", name)
                or name.startswith("http")
                or re.match(r"^x\d+$", name)
            )

        if any(_looks_like_value(c) for c in w.columns):
            w = pd.read_csv(WINDOWS_CSV, header=None)
            # assign best-effort column names by position
            n = w.shape[1]
            cols = ["id", "start", "end"]
            if n >= 4:
                cols += ["country"]
            if n >= 5:
                cols += ["citation"]
            if n > len(cols):
                cols += [f"extra{i}" for i in range(n - len(cols))]
            w.columns = cols[:n]
            # Lowercase for consistency
            w.columns = [str(c).strip().lower() for c in w.columns]

        # Expected minimal schema (with robust aliasing)
        # structures: structure_id, lon, lat, [name]
        # windows: id (or window_id), structure_id (or lon/lat), start/end (or date variants), [note]
        s_cols = set(s.columns)
        w_cols = set(w.columns)
        # Structures may be optional if windows already carry lon/lat
        need_struct_cols = ["structure_id", "lon", "lat"]
        have_struct = all(c in s_cols for c in need_struct_cols)
        # Accept either 'id' or 'window_id' for windows key
        if "id" not in w_cols and "window_id" in w_cols:
            w = w.rename(columns={"window_id": "id"})
            w_cols = set(w.columns)
        # Accept alternative date column names
        if "start" not in w_cols and "start_date" in w_cols:
            w = w.rename(columns={"start_date": "start"})
            w_cols = set(w.columns)
        if "end" not in w_cols and "end_date" in w_cols:
            w = w.rename(columns={"end_date": "end"})
            w_cols = set(w.columns)
        # If id is still missing, generate a stable index-based id
        if "id" not in w_cols:
            w.insert(0, "id", range(len(w)))
            w_cols = set(w.columns)

        # Tolerant start/end detection (handle startdate, date, timestamp)
        if "start" not in w_cols:
            for cand in ("startdate", "start_dt", "starttime", "start_time"):
                if cand in w_cols:
                    w = w.rename(columns={cand: "start"})
                    w_cols = set(w.columns)
                    break
        if "end" not in w_cols:
            for cand in ("enddate", "end_dt", "endtime", "end_time"):
                if cand in w_cols:
                    w = w.rename(columns={cand: "end"})
                    w_cols = set(w.columns)
                    break

        # Fallback: single date/timestamp used for both
        if "start" not in w_cols:
            if "date" in w_cols:
                w["start"] = w["date"].astype("string")
            elif "timestamp" in w_cols:
                w["start"] = w["timestamp"].astype("string")
            else:
                raise ValueError("windows.csv missing start column")
            w_cols = set(w.columns)
        if "end" not in w_cols:
            if "date" in w_cols:
                w["end"] = w["date"].astype("string")
            elif "timestamp" in w_cols:
                w["end"] = w["timestamp"].astype("string")
            else:
                raise ValueError("windows.csv missing end column")
            w_cols = set(w.columns)

        # Validate minimal set now
        for col in ["id", "start", "end"]:
            if col not in w_cols:
                raise ValueError("windows.csv missing column: " + col)

        # If windows has structure_id and we have a valid structures table, join to fetch lon/lat.
        if "structure_id" in w_cols and have_struct:
            # Coerce types (structure_id can be string; do not force numeric)
            s["structure_id"] = s["structure_id"].astype("string")
            s["lon"] = pd.to_numeric(s["lon"], errors="coerce")
            s["lat"] = pd.to_numeric(s["lat"], errors="coerce")
            w["id"] = pd.to_numeric(w["id"], errors="coerce").astype("Int64")
            w["structure_id"] = w["structure_id"].astype("string")
            # Use suffixes to avoid column overlap (e.g., country present in both)
            j = w.merge(s, on="structure_id", how="left", suffixes=("_win", ""))
        else:
            # Fall back to using lon/lat directly from windows.csv
            # Ensure lon/lat exist; else fallback to legacy events.csv if available
            if not all(c in w_cols for c in ("lon", "lat")):
                # Attempt legacy fallback
                legacy = _read_csv(EVENTS_CSV).copy()
                if not legacy.empty:
                    # Normalize legacy events.csv
                    if "id" not in legacy.columns:
                        legacy["id"] = legacy.index
                    for col in ("id",):
                        legacy[col] = pd.to_numeric(
                            legacy[col], errors="coerce"
                        ).astype("Int64")
                    for col in ("lon", "lat"):
                        if col in legacy.columns:
                            legacy[col] = pd.to_numeric(legacy[col], errors="coerce")
                    for col in ("start", "end"):
                        if col in legacy.columns:
                            legacy[col] = legacy[col].astype("string")
                    return legacy[["id", "lon", "lat", "start", "end"]]
                raise ValueError(
                    "windows.csv missing structure_id or lon/lat. Columns present: "
                    + ", ".join(sorted(w_cols))
                )
            w["id"] = pd.to_numeric(w["id"], errors="coerce").astype("Int64")
            w["lon"] = pd.to_numeric(w["lon"], errors="coerce")
            w["lat"] = pd.to_numeric(w["lat"], errors="coerce")
            j = w.copy()
        # Keep same expected columns for downstream
        cols = ["id", "lon", "lat", "start", "end"]
        for c in cols:
            if c not in j.columns:
                j[c] = pd.NA
        # Preserve optional labels
        if "name" in j.columns:
            j = j.rename(columns={"name": "label"})
        return j[cols + (["label"] if "label" in j.columns else [])].copy()

    # Legacy fallback
    df = _read_csv(EVENTS_CSV).copy()
    if df.empty:
        return df
    # Ensure canonical column names
    if "id" not in df.columns:
        # Fall back to index as id if absent
        df["id"] = df.index
    # Types
    for col in ("id",):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ("lon", "lat"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Keep dates as strings to avoid timezone parsing surprises
    for col in ("start", "end"):
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def load_granules() -> pd.DataFrame:
    """Load granules.csv with canonicalised column names and types.

    Normalises synonyms:
      - sga_local_med   → sga_local_median
      - sgi_local_med   → sgi_median
    """
    df = _read_csv(GRANULES_CSV).copy()
    if df.empty:
        return df

    # Normalise synonyms
    if "sga_local_median" not in df.columns and "sga_local_med" in df.columns:
        df = df.rename(columns={"sga_local_med": "sga_local_median"})
    if "sgi_median" not in df.columns and "sgi_local_med" in df.columns:
        df = df.rename(columns={"sgi_local_med": "sgi_median"})

    # Types
    if "system_index" in df.columns:
        df["system_index"] = df["system_index"].astype("string")
    else:
        df["system_index"] = pd.Series(dtype="string")
    for col in ("sga_scene", "cloudiness", "sga_local_median", "sgi_median"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype("string")
    if "git_hash" in df.columns:
        df["git_hash"] = df["git_hash"].astype("string")
    return df


def load_process_runs() -> pd.DataFrame:
    """Load process_runs.csv only.

    Keeps rows with empty system_index (marker: "no granules found").
    Columns recognized: window_id, system_index, git_hash, last_timestamp, and
    optional sga_local_median/sgi_median.
    """
    df = _read_csv(PROCESS_RUNS_CSV).copy()
    if df.empty:
        return df
    # Normalize legacy 'event_id' to 'window_id', then drop any stray event_id
    if "window_id" not in df.columns and "event_id" in df.columns:
        df = df.rename(columns={"event_id": "window_id"})
    if "window_id" not in df.columns:
        df["window_id"] = pd.NA
    df["window_id"] = pd.to_numeric(df["window_id"], errors="coerce").astype("Int64")
    if "event_id" in df.columns:
        df = df.drop(columns=["event_id"])  # fully remove event_id
    if "system_index" not in df.columns:
        df["system_index"] = pd.Series(dtype="string")
    df["system_index"] = df["system_index"].astype("string")
    if "git_hash" in df.columns:
        df["git_hash"] = df["git_hash"].astype("string")
    if "last_timestamp" in df.columns:
        df["last_timestamp"] = df["last_timestamp"].astype("string")
    for col in ("sga_local_median", "sgi_median"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Deduplicate defensively
    df = df.drop_duplicates(subset=["window_id", "system_index"], keep="first")
    return df


# -----------------------------------------------------------------------------
# Join + filter helpers
# -----------------------------------------------------------------------------
@dataclass
class Filters:
    scene_cloud_pct: Optional[float]
    scene_sga_range: Optional[Tuple[float, float]]
    local_sga_range: Optional[Tuple[float, float]]
    local_sgi_range: Optional[Tuple[float, float]]


def _resolve_filters(**kwargs) -> Filters:
    # Defaults from config
    scene_cloud = kwargs.get(
        "scene_cloud_pct", float(cfg.MASK_PARAMS["cloud"]["scene_cloud_pct"])
    )
    scene_sga = kwargs.get(
        "scene_sga_range", tuple(cfg.MASK_PARAMS["sunglint"]["scene_sga_range"])
    )
    local_sga = kwargs.get("local_sga_range")  # default None
    local_sgi = kwargs.get("local_sgi_range")  # default None

    # Normalise tuple-ish inputs
    def _as_pair(x):
        if x is None:
            return None
        if isinstance(x, (list, tuple)) and len(x) == 2:
            return float(x[0]), float(x[1])
        raise ValueError("Range filters must be two-element (min, max)")

    return Filters(
        scene_cloud_pct=None if scene_cloud is None else float(scene_cloud),
        scene_sga_range=_as_pair(scene_sga) if scene_sga is not None else None,
        local_sga_range=_as_pair(local_sga) if local_sga is not None else None,
        local_sgi_range=_as_pair(local_sgi) if local_sgi is not None else None,
    )


def virtual_db(**kwargs) -> pd.DataFrame:
    """Return the full joined DataFrame (events ⨝ mapping ⨝ granules), then
    apply any provided filters.

    Filters accepted as kwargs (see module docstring). If a filter is not
    provided (or is None), it is not applied. The two scene-level filters
    default to config values and are applied by default.
    """
    events = load_events()
    mapping = load_process_runs()
    grans = load_granules()

    # Disambiguate overlapping column names to avoid merge collisions
    if not mapping.empty and "git_hash" in mapping.columns:
        mapping = mapping.rename(columns={"git_hash": "run_git_hash"})
    if not grans.empty:
        ren = {}
        if "git_hash" in grans.columns:
            ren["git_hash"] = "granule_git_hash"
        if "timestamp" in grans.columns:
            ren["timestamp"] = "granule_timestamp"
        if ren:
            grans = grans.rename(columns=ren)

    if mapping.empty or events.empty:
        return pd.DataFrame()

    # Rename for join clarity
    ev = events.rename(columns={"id": "window_id"})

    # Remove mapping rows with empty system_index (marker for no granules)
    map_nonempty = mapping[mapping["system_index"].astype(str).str.len() > 0].copy()

    # event ⨝ mapping
    joined = map_nonempty.merge(ev, on="window_id", how="left", suffixes=("", "_win"))
    # ⨝ granules (metadata columns)
    joined = joined.merge(
        grans,
        on="system_index",
        how="left",
        suffixes=("", "_gran"),
    )

    # Apply filters
    f = _resolve_filters(**kwargs)

    def _between_strict(series: pd.Series, lo: float, hi: float) -> pd.Series:
        return (series > lo) & (series < hi)

    if f.scene_cloud_pct is not None:
        # drop rows with missing cloudiness when filter is active
        joined = joined[joined["cloudiness"].notna()]
        joined = joined[joined["cloudiness"] <= f.scene_cloud_pct]

    if f.scene_sga_range is not None:
        lo, hi = f.scene_sga_range
        joined = joined[joined["sga_scene"].notna()]
        joined = joined[_between_strict(joined["sga_scene"], lo, hi)]

    # Local filters only when provided
    if f.local_sga_range is not None:
        lo, hi = f.local_sga_range
        col = (
            "sga_local_median"
            if "sga_local_median" in joined.columns
            else "sga_local_med"
        )
        if col not in joined.columns:
            # No column available → nothing matches when filter is requested
            return joined.iloc[0:0]
        joined = joined[joined[col].notna()]
        joined = joined[_between_strict(joined[col], lo, hi)]

    if f.local_sgi_range is not None:
        lo, hi = f.local_sgi_range
        col = "sgi_median" if "sgi_median" in joined.columns else "sgi_local_med"
        if col not in joined.columns:
            return joined.iloc[0:0]
        joined = joined[joined[col].notna()]
        joined = joined[_between_strict(joined[col], lo, hi)]

    return joined.reset_index(drop=True)


# -----------------------------------------------------------------------------
# High-level queries
# -----------------------------------------------------------------------------
def df_for_window(wid: int, **kwargs) -> pd.DataFrame:
    """Return fully-joined rows for a single window id.

    Uses unified_db for completeness (includes rows even when runs/granules are missing).
    """
    return unified_db(window_ids=[wid])


def df_for_granule(gid: str, **kwargs) -> pd.DataFrame:
    """Return fully-joined rows for a single granule id (system:index).

    Uses unified_db for completeness.
    """
    return unified_db(system_indexes=[gid])


# -----------------------------------------------------------------------------
# Upsert helpers (create/update rows)
# -----------------------------------------------------------------------------
def upsert_event(row: dict) -> None:
    """Create or update a row in events.csv keyed by id.

    Keys: id (required), lon, lat, start, end, plus any arbitrary columns.
    """
    if "id" not in row:
        raise ValueError("'id' is required to upsert an event row")
    df = load_events()
    if df.empty:
        df = pd.DataFrame(columns=["id", "lon", "lat", "start", "end"])  # base schema
    key = int(row["id"])  # normalize
    # Upsert
    if (df.get("id") == key).any():
        idx = df.index[df["id"] == key][0]
        for k, v in row.items():
            df.loc[idx, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(EVENTS_CSV, index=False)


def upsert_granule(row: dict) -> None:
    """Create or update a row in granules.csv keyed by system_index.

    Keys: system_index (required). Accepts quality columns such as sga_scene,
    cloudiness, timestamp, sga_local_median (or sga_local_med), sgi_median
    (or sgi_local_med).
    """
    if "system_index" not in row:
        raise ValueError("'system_index' is required to upsert a granule row")
    df = load_granules()
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "system_index",
                "sga_scene",
                "cloudiness",
                "timestamp",
                "sga_local_median",
                "sgi_median",
                "git_hash",
            ]
        )

    # Canonicalise synonyms in input
    if "sga_local_med" in row and "sga_local_median" not in row:
        row["sga_local_median"] = row.pop("sga_local_med")
    if "sgi_local_med" in row and "sgi_median" not in row:
        row["sgi_median"] = row.pop("sgi_local_med")

    # Ensure git_hash column exists even if not provided
    if "git_hash" not in df.columns:
        df["git_hash"] = pd.NA
    key = str(row["system_index"])  # normalize
    if (df.get("system_index") == key).any():
        idx = df.index[df["system_index"] == key][0]
        for k, v in row.items():
            df.loc[idx, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    # Persist with reduced float precision (4 decimals) to keep files tidy
    _atomic_write_csv(df, GRANULES_CSV, float_format="%.4f")


def upsert_process_run(
    window_id: int,
    system_index: str,
    *,
    git_hash: Optional[str] = None,
    last_timestamp: Optional[str] = None,
) -> None:
    """Create or update a process run mapping row.

    Keys: (window_id, system_index). Additional metadata columns:
      git_hash (string), last_timestamp (ISO-8601 string, UTC recommended).
    """
    df = load_process_runs()
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "window_id",
                "system_index",
                "git_hash",
                "last_timestamp",
                "sga_local_median",
                "sgi_median",
                "valid_pixel_c",
                "valid_pixel_mbsp",
                "hitl_value",
            ]
        )
    # Ensure columns exist
    for c in [
        "git_hash",
        "last_timestamp",
        "sga_local_median",
        "sgi_median",
        "valid_pixel_c",
        "valid_pixel_mbsp",
        "hitl_value",
    ]:
        if c not in df.columns:
            df[c] = pd.NA
    if "window_id" not in df.columns:
        df["window_id"] = pd.NA
    key_e, key_s = int(window_id), str(system_index)
    mask = (df.get("window_id") == key_e) & (
        df.get("system_index").astype(str) == key_s
    )
    if mask.any():
        idx = df.index[mask][0]
        if git_hash is not None:
            df.loc[idx, "git_hash"] = git_hash
        if last_timestamp is not None:
            df.loc[idx, "last_timestamp"] = last_timestamp
    else:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "window_id": key_e,
                            "system_index": key_s,
                            "git_hash": git_hash,
                            "last_timestamp": last_timestamp,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    _atomic_write_csv(df, PROCESS_RUNS_CSV)


# Backward-compatible alias
# (legacy helpers removed)


# Optional explicit loaders for split schema
def load_structures() -> pd.DataFrame:
    return _read_csv(STRUCTURES_CSV).copy()


def load_windows() -> pd.DataFrame:
    return _read_csv(WINDOWS_CSV).copy()


# -----------------------------------------------------------------------------
# Unified join + query helpers
# -----------------------------------------------------------------------------
def _coerce_iter(x: Optional[Iterable | str | int]) -> Optional[list]:
    if x is None:
        return None
    if isinstance(x, (list, tuple, set, pd.Series)):
        return [str(i) if not isinstance(i, (int, float)) else int(i) for i in x]
    return [str(x) if not isinstance(x, (int, float)) else int(x)]


def unified_db(
    *,
    structure_ids: Optional[Iterable[str | int]] = None,
    window_ids: Optional[Iterable[str | int]] = None,
    system_indexes: Optional[Iterable[str]] = None,
    pairs: Optional[Iterable[Tuple[str | int, str]]] = None,
) -> pd.DataFrame:
    """Return a complete, unified DataFrame spanning structures, windows,
    process_runs and granules, filtered by any of the supplied identifiers.

    - structure_ids: one or more structure_id values
    - window_ids:    one or more window_id values (aka windows.id)
    - system_indexes: one or more granule ids (system:index)
    - pairs:         (window_id, system_index) tuples to select specific runs

    The join is performed in both directions with outer/left joins so that
    missing relations still yield rows with NaNs on the absent side.
    """
    s = load_structures()
    w = load_windows()
    r = load_process_runs()
    g = load_granules()

    # Normalize keys
    # Ensure key columns exist even for empty frames
    if "id" in w.columns:
        w = w.rename(columns={"id": "window_id"})
    if "window_id" not in w.columns:
        w["window_id"] = pd.Series(dtype="Int64")
    if "structure_id" not in w.columns:
        w["structure_id"] = pd.Series(dtype="string")
    if not w.empty:
        w["window_id"] = pd.to_numeric(w["window_id"], errors="coerce").astype("Int64")
        w["structure_id"] = w["structure_id"].astype(str)
    if "structure_id" not in s.columns:
        s["structure_id"] = pd.Series(dtype="string")
    if not s.empty:
        s["structure_id"] = s["structure_id"].astype(str)
    if "window_id" not in r.columns and "event_id" in r.columns:
        r = r.rename(columns={"event_id": "window_id"})
    if "window_id" not in r.columns:
        r["window_id"] = pd.Series(dtype="Int64")
    if "system_index" not in r.columns:
        r["system_index"] = pd.Series(dtype="string")
    r = r.rename(columns={"git_hash": "run_git_hash"})
    if not r.empty:
        r["window_id"] = pd.to_numeric(r["window_id"], errors="coerce").astype("Int64")
        r["system_index"] = r["system_index"].astype(str)
    if "system_index" not in g.columns:
        g["system_index"] = pd.Series(dtype="string")
    g = g.rename(
        columns={"git_hash": "granule_git_hash", "timestamp": "granule_timestamp"}
    )
    if not g.empty:
        g["system_index"] = g["system_index"].astype(str)

    # Windows ⟷ Runs (full outer to include windows with no runs and runs without matching window)
    wr = pd.merge(r, w, on="window_id", how="outer", suffixes=("_run", "_win"))
    # + Structures (left on structure_id; may be NaN if run has no matching window)
    wrs = pd.merge(wr, s, on="structure_id", how="left", suffixes=("", "_str"))
    # + Granules (left on system_index; may be NaN if no granule metadata available)
    full = pd.merge(wrs, g, on="system_index", how="left", suffixes=("", ""))

    # Filtering
    sid_list = _coerce_iter(structure_ids)
    wid_list = _coerce_iter(window_ids)
    sys_list = _coerce_iter(system_indexes)
    pair_list = pairs if pairs is not None else []

    if not any([sid_list, wid_list, sys_list, pair_list]):
        out = full
    else:
        mask = pd.Series([False] * len(full)) if len(full) else pd.Series(dtype=bool)
        if sid_list is not None and not full.empty:
            mask |= full["structure_id"].astype(str).isin([str(x) for x in sid_list])
        if wid_list is not None and not full.empty:
            wid_ser = pd.Series(wid_list)
            # coerce to Int64 to compare against window_id dtype
            wid_ser = pd.to_numeric(wid_ser, errors="coerce").astype("Int64")
            mask |= full["window_id"].astype("Int64").isin(wid_ser)
        if sys_list is not None and not full.empty:
            mask |= full["system_index"].astype(str).isin([str(x) for x in sys_list])
        if pair_list and not full.empty:
            pair_df = pd.DataFrame(pair_list, columns=["window_id", "system_index"])
            pair_df["window_id"] = pd.to_numeric(
                pair_df["window_id"], errors="coerce"
            ).astype("Int64")
            pair_df["system_index"] = pair_df["system_index"].astype(str)
            merged = full.merge(
                pair_df, on=["window_id", "system_index"], how="left", indicator=True
            )
            mask |= merged["_merge"] == "both"
        out = full[mask].copy()

    # Deterministic ordering and column order
    sort_keys = [
        c for c in ["window_id", "system_index", "structure_id"] if c in out.columns
    ]
    if sort_keys:
        out = out.sort_values(sort_keys).reset_index(drop=True)
    return out


def df_for_structure(structure_id: str | int) -> pd.DataFrame:
    return unified_db(structure_ids=[structure_id])


def df_for_run(window_id: int, system_index: str) -> pd.DataFrame:
    return unified_db(pairs=[(window_id, system_index)])


def update_run_metrics(
    window_id: int,
    system_index: str,
    *,
    sga_local_median: Optional[float] = None,
    sgi_median: Optional[float] = None,
    valid_pixel_c: Optional[float] = None,
    valid_pixel_mbsp: Optional[float] = None,
    hitl_value: Optional[int] = None,
    git_hash: Optional[str] = None,
) -> None:
    """Update per-run local metrics in process_runs.csv.

    Keys are (window_id, system_index). Creates the row if missing.
    """
    df = load_process_runs()
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "window_id",
                "system_index",
                "git_hash",
                "last_timestamp",
                "sga_local_median",
                "sgi_median",
                "valid_pixel_c",
                "valid_pixel_mbsp",
                "hitl_value",
            ]
        )
    for c in (
        "sga_local_median",
        "sgi_median",
        "valid_pixel_c",
        "valid_pixel_mbsp",
        "hitl_value",
        "git_hash",
    ):
        if c not in df.columns:
            df[c] = pd.NA
    if "window_id" not in df.columns:
        df["window_id"] = pd.NA

    key_e, key_s = int(window_id), str(system_index)
    mask = (df.get("window_id") == key_e) & (
        df.get("system_index").astype(str) == key_s
    )
    if mask.any():
        idx = df.index[mask][0]
        if sga_local_median is not None:
            df.loc[idx, "sga_local_median"] = sga_local_median
        if sgi_median is not None:
            df.loc[idx, "sgi_median"] = sgi_median
        if git_hash is not None:
            df.loc[idx, "git_hash"] = git_hash
        if valid_pixel_c is not None:
            df.loc[idx, "valid_pixel_c"] = float(valid_pixel_c)
        if valid_pixel_mbsp is not None:
            df.loc[idx, "valid_pixel_mbsp"] = float(valid_pixel_mbsp)
        if hitl_value is not None:
            df.loc[idx, "hitl_value"] = int(hitl_value)
    else:
        row = {
            "window_id": key_e,
            "system_index": key_s,
        }
        if sga_local_median is not None:
            row["sga_local_median"] = sga_local_median
        if sgi_median is not None:
            row["sgi_median"] = sgi_median
        if valid_pixel_c is not None:
            row["valid_pixel_c"] = float(valid_pixel_c)
        if valid_pixel_mbsp is not None:
            row["valid_pixel_mbsp"] = float(valid_pixel_mbsp)
        if hitl_value is not None:
            row["hitl_value"] = int(hitl_value)
        if git_hash is not None:
            row["git_hash"] = git_hash
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    _atomic_write_csv(df, PROCESS_RUNS_CSV)
