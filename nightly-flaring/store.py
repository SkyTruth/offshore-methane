# %%
"""Typed segment exports and a merged, local-night Parquet store (D69)."""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import tempfile
import uuid
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
from pandas.testing import assert_frame_equal

DEFAULT_ROOT = Path(__file__).resolve().parent / "outputs/store"
# `<name>.<pid>.<uuid hex>.tmp`, the names `_Store.atomic` writes.
TEMP_SUFFIX = r"\d+\.[0-9a-f]{32}\.tmp"
# Read once at import, so the digest matches the code this process runs.
_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
# Audit fields can be numeric in one CSV and strings in another. Each child
# retains its original scalar type after the frames have been concatenated.
AUDIT = pa.struct(
    [("text", pa.string()), ("number", pa.float64()), ("integer", pa.int64())]
)
STAMP = pa.timestamp("us", tz="UTC")


def _schema(columns: str, integers: str, *, flags: str = "", audit: str = ""):
    ints, bools, mixed = set(integers.split()), set(flags.split()), set(audit.split())
    return pa.schema(
        [
            (
                col,
                STAMP
                if col == "time_utc"
                else pa.date32()
                if col == "night"
                else pa.string()
                if col in {"satellite", "granule", "status"}
                else pa.int64()
                if col in ints
                else pa.bool_()
                if col in bools
                else AUDIT
                if col in mixed
                else pa.float64(),
            )
            for col in columns.split()
        ]
        + ([("_time_utc_text", pa.string())] if "time_utc" in columns.split() else [])
    )


SCHEMAS = {
    "overpasses": _schema(
        """satellite granule time_utc site_id row col dist_m sza cloud_mask
        cloud_clear_neighbors n_hot_m10 n_fitted n_band_rule_changed n_clusters
        n_sources rh_peak_mw n_rh75 n_rh_cap_excluded n_components_split
        split_component_ids n_copies_merged peak_row peak_col peak_lat peak_lon
        peak_dist_m detected temp_k area_m2 rh_mw bands fit_flag bytes_granule""",
        """site_id row col cloud_mask cloud_clear_neighbors n_hot_m10 n_fitted
        n_band_rule_changed n_clusters n_sources n_rh75 n_rh_cap_excluded
        n_components_split n_copies_merged peak_row peak_col bytes_granule""",
        flags="detected",
        audit="split_component_ids bands fit_flag",
    ),
    "site_nights": _schema(
        """night site_id lat lon status rh_mw bcm_per_year rh_max_mw temp_k
        n_overpasses n_clear n_detected n_detected_cloud n_clear_detclear
        rh_mw_detclear bcm_per_year_detclear""",
        "site_id n_overpasses n_clear n_detected n_detected_cloud n_clear_detclear",
    ),
    "copies": _schema(
        """satellite granule time_utc scan component_id peak_row peak_col
        peak_lat peak_lon footprint_m2 m10_peak prominence_ratio rh75_mw rh_peak_mw
        n_rh75 n_rh_cap_excluded temp_k area_m2 bands fit_flag t_fit_rejected
        cloud_mask_peak clear_neighbors_peak group_id group_size pair_distance_px
        pair_dcol pair_drow pair_rh_ratio group_rh_mw group_max_rh_mw centroid_lat
        centroid_lon assigned_site site_dist_m band_rule_changed n_hot
        n_suppressed_maxima suppressed_max_m10 suppressed_saddle_ratio""",
        """scan component_id peak_row peak_col n_rh75 n_rh_cap_excluded
        cloud_mask_peak clear_neighbors_peak group_id group_size assigned_site
        n_hot n_suppressed_maxima""",
        flags="band_rule_changed",
        audit="bands fit_flag pair_distance_px pair_dcol pair_drow pair_rh_ratio",
    ),
}
ORDER = ["_segment", "_row"]
META = b"legacy_frame"


def frame_metadata(frame: pd.DataFrame, table: str = "frame") -> dict:
    """Describe the CSV frame independently of its physical Arrow types."""
    if not isinstance(frame.index, pd.RangeIndex) or frame.index.name is not None:
        raise ValueError("legacy frames require an unnamed RangeIndex")
    dtypes = {col: str(dtype) for col, dtype in frame.dtypes.items()}
    for col, dtype in dtypes.items():
        if dtype not in {"object", "int64", "float64", "bool"}:
            value = frame[col].iloc[0] if len(frame) else "<empty>"
            raise ValueError(f"{table}.{col}: {value!r}: unsupported dtype {dtype}")
    return dict(
        columns=list(frame.columns),
        dtypes=dtypes,
        index=[frame.index.start, frame.index.stop, frame.index.step],
    )


def _audit_array(series: pd.Series) -> pa.Array:
    kind = pd.api.types.infer_dtype(series, skipna=True)
    empty = pa.nulls(len(series))
    children = [empty.cast(f.type) for f in AUDIT]
    if kind in {"string", "empty"}:
        children[0] = pa.array(series, type=pa.string(), from_pandas=True)
    elif kind == "floating" or (
        kind == "mixed-integer-float" and series.dtype != "object"
    ):
        children[1] = pa.array(series, type=pa.float64(), from_pandas=True)
    elif kind == "integer":
        children[2] = pa.array(series, type=pa.int64(), from_pandas=True)
    else:
        # Mixed numeric/string columns arise when legacy segments concatenate.
        text, numbers, integers = [], [], []
        for value in series:
            if isinstance(value, str):
                text.append(value)
                numbers.append(None)
                integers.append(None)
            elif pd.isna(value):
                text.append(None)
                numbers.append(None)
                integers.append(None)
            elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                text.append(None)
                numbers.append(None)
                integers.append(value)
            elif isinstance(value, (float, np.floating)):
                text.append(None)
                numbers.append(value)
                integers.append(None)
            else:
                raise ValueError(f"unsupported audit value: {value!r}")
        children = [
            pa.array(v, type=f.type) for v, f in zip((text, numbers, integers), AUDIT)
        ]
    return pa.StructArray.from_arrays(children, fields=list(AUDIT))


def _array(series: pd.Series, field: pa.Field) -> pa.Array:
    """Encode one column, rejecting values whose scalar meaning would change."""
    if field.type == AUDIT:
        return _audit_array(series)
    if field.type == STAMP:
        present = series.dropna()
        if len(present) and pd.api.types.infer_dtype(present) != "string":
            raise ValueError("time_utc requires timestamp text")
        times = pd.to_datetime(series, utc=True, format="mixed")
        return pa.array(times, type=pa.timestamp("ns", tz="UTC")).cast(STAMP, safe=True)
    if pa.types.is_date(field.type):
        dates = pd.to_datetime(series, format="%Y-%m-%d", errors="raise")
        present = series.notna()
        if present.any() and not dates[present].dt.strftime("%Y-%m-%d").equals(
            series[present]
        ):
            raise ValueError("night must contain ISO evening dates")
        return pa.array(dates.dt.date, type=field.type, from_pandas=True)
    if pa.types.is_boolean(field.type):
        valid = series.dropna()
        if len(valid) and pd.api.types.infer_dtype(valid) != "boolean":
            raise ValueError("flags require true/false/blank booleans")
    if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
        if pd.api.types.infer_dtype(series, skipna=True) == "boolean":
            raise ValueError("boolean is not a numeric measurement or count")
    if pa.types.is_integer(field.type) and series.dtype == "float64":
        if ((series == 0) & np.signbit(series)).any():
            raise ValueError("negative zero cannot round-trip as integer")
    return pa.array(series, type=field.type, from_pandas=True, safe=True)


def to_arrow(frame: pd.DataFrame, table: str, *, ordered: bool = False) -> pa.Table:
    """Encode a legacy CSV frame with a fixed schema; reject incompatible data."""
    schema = SCHEMAS[table]
    if ordered:
        schema = pa.schema(list(schema) + [(col, pa.int64()) for col in ORDER])
    unknown = set(frame.columns) - set(schema.names) | (
        set(frame.columns) & {"_time_utc_text"}
    )
    if unknown or not frame.columns.is_unique:
        raise ValueError(f"unexpected {table} columns: {sorted(unknown)}")
    metadata = frame_metadata(frame, table)
    arrays = []
    for field in schema:
        if field.name == "_time_utc_text":
            series = frame.get("time_utc", pd.Series(np.nan, index=frame.index))
            array = pa.array(series, type=pa.string(), from_pandas=True)
        elif field.name not in frame:
            array = pa.nulls(len(frame), type=field.type)
        else:
            series = frame[field.name]
            try:
                array = _array(series, field)
            except (ValueError, TypeError, OverflowError) as exc:
                # The slow scalar search runs only on rejected columns.
                value = "<empty>"
                for i, value in enumerate(series):
                    try:
                        _array(series.iloc[i : i + 1], field)
                    except (ValueError, TypeError, OverflowError):
                        break
                raise ValueError(f"{table}.{field.name}: {value!r}: {exc}") from exc
        arrays.append(array)
    return pa.Table.from_arrays(arrays, schema=schema).replace_schema_metadata(
        {META: json.dumps(metadata).encode()}
    )


def from_arrow(arrow: pa.Table, metadata: dict | None = None) -> pd.DataFrame:
    """Restore values, scalar kinds, dtypes and column order of a legacy frame."""
    metadata = metadata or json.loads(arrow.schema.metadata[META])
    values = {}
    for col in metadata["columns"]:
        dtype = metadata["dtypes"][col]
        data = arrow["_time_utc_text" if col == "time_utc" else col]
        if data.type == AUDIT:
            data = data.combine_chunks()
            text, number, integer = [data.field(i) for i in range(3)]
            if dtype == "object":
                result = np.array(text.to_pylist(), dtype=object)
                for child in (number, integer):
                    valid = child.is_valid().to_numpy(zero_copy_only=False)
                    result[valid] = np.asarray(child.to_pylist(), dtype=object)[valid]
            else:
                child = integer if dtype == "int64" else number
                result = child.to_pandas().to_numpy()
                other = number if dtype == "int64" else integer
                valid = other.is_valid().to_numpy(zero_copy_only=False)
                if valid.any():
                    result = result.copy()
                    result[valid] = other.to_pandas().to_numpy()[valid]
        elif pa.types.is_date(data.type):
            result = np.array(
                [v.isoformat() if v else np.nan for v in data.to_pylist()], dtype=object
            )
        else:
            result = data.to_pandas().to_numpy()
        if dtype == "object":
            result = np.asarray(result, dtype=object)
            result[pd.isna(result)] = np.nan
        values[col] = np.asarray(result).astype(dtype, copy=False)
    # CSV frames consolidate columns of the same dtype. pandas concat's empty
    # block resolution depends on that layout as well as the public dtypes.
    frame = pd.DataFrame(values, columns=metadata["columns"])
    # Segment frames have a RangeIndex; store reads deliberately reset it.
    index = metadata.get("index")
    if index is not None:
        frame.index = pd.RangeIndex(*index)
    return frame


def write_frame(frame: pd.DataFrame, path: Path, table: str) -> None:
    """Write a legacy frame as zstd Parquet via an atomic sibling rename."""
    arrow = to_arrow(frame, table)
    tmp = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        pq.write_table(arrow, tmp, compression="zstd", coerce_timestamps="us")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def write_export(frame: pd.DataFrame, path: Path, table: str) -> None:
    """Preserve the CSV parser's inference and float rounding on pipeline output."""
    with tempfile.TemporaryFile(mode="w+") as stream:
        frame.to_csv(stream, index=False)
        stream.seek(0)
        legacy = pd.read_csv(stream, low_memory=False)
    write_frame(legacy, path, table)


def segment_path(folder: Path, tag: str, table: str) -> Path:
    """Prefer Parquet; return the legacy CSV path when Parquet is absent."""
    path = folder / f"{table}_{tag}.parquet"
    return path if path.exists() else path.with_suffix(".csv")


def read_segment(folder: Path, tag: str, table: str, columns=None) -> pd.DataFrame:
    """Read either segment format as the frame returned by the legacy CSV parser."""
    return read_export(segment_path(folder, tag, table), columns)


def read_export(path: Path, columns=None) -> pd.DataFrame:
    """Read an explicitly named export, including validation path overrides."""
    if path.suffix == ".csv":
        return pd.read_csv(path, low_memory=False, usecols=columns)
    with pq.ParquetFile(path) as file:
        metadata = json.loads(file.schema_arrow.metadata[META])
        if columns is not None:
            missing = set(columns) - set(metadata["columns"])
            if missing:
                raise ValueError(f"columns not found: {sorted(missing)}")
            metadata["columns"] = [c for c in metadata["columns"] if c in columns]
        physical = [
            "_time_utc_text" if c == "time_utc" else c for c in metadata["columns"]
        ]
        return from_arrow(file.read(columns=physical), metadata)


def convert(segment: tuple[Path, str], force: bool = False) -> list[dict]:
    """Convert missing Parquet exports; force explicitly replaces existing ones."""
    folder, tag = segment
    result = []
    for table in SCHEMAS:
        source = folder / f"{table}_{tag}.csv"
        target = source.with_suffix(".parquet")
        if not source.exists() or (target.exists() and not force):
            continue
        frame = pd.read_csv(source, low_memory=False)
        # Verify before exposing the converted export to readers.
        tmp = target.with_name(
            target.name + f".{os.getpid()}.{uuid.uuid4().hex}.verify.tmp"
        )
        try:
            write_frame(frame, tmp, table)
            actual = from_arrow(pq.ParquetFile(tmp).read())
            assert_frame_equal(frame, actual, check_exact=True)
            tmp.replace(target)
        finally:
            tmp.unlink(missing_ok=True)
        sizes = dict(
            segment=folder.name,
            table=table,
            csv=source.stat().st_size,
            parquet=target.stat().st_size,
        )
        result.append(sizes)
        print(json.dumps(sizes), flush=True)
    return result


class _Store:
    """All store paths and I/O use the single filesystem resolved from root."""

    def __init__(self, root):
        uri = str(root)
        if "://" not in uri:
            uri = str(Path(uri).expanduser().absolute())
        self.fs, self.root = pafs.FileSystem.from_uri(uri)

    @contextmanager
    def lock(self, *, shared=False):
        """Local writers wait for the root lock; readers share it between writes."""
        if not isinstance(self.fs, pafs.LocalFileSystem):
            yield
            return
        import fcntl

        self.fs.create_dir(self.root, recursive=True)
        # Never unlink the lock: waiting processes must use the same inode.
        with open(self.path(".lock"), "a+b") as stream:
            fcntl.flock(stream, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def path(self, *parts):
        return posixpath.join(self.root, *parts)

    def exists(self, path):
        return self.fs.get_file_info(path).type != pafs.FileType.NotFound

    def manifest(self):
        path = self.path("_segments.json")
        if not self.exists(path):
            return None
        with self.fs.open_input_file(path) as stream:
            return json.loads(stream.read())

    def atomic(self, path, writer):
        if isinstance(self.fs, pafs.LocalFileSystem):
            self.fs.create_dir(posixpath.dirname(path), recursive=True)
        tmp = path + f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
        with self.fs.open_output_stream(tmp) as stream:
            writer(stream)
        self.fs.move(tmp, path)

    def parts(self, table):
        folder = self.path(table)
        if not self.exists(folder):
            return {}
        return {
            posixpath.basename(posixpath.dirname(info.path)).removeprefix(
                "night="
            ): info.path
            for info in self.fs.get_file_info(pafs.FileSelector(folder, recursive=True))
            if info.type == pafs.FileType.File
            and posixpath.basename(info.path) == "part.parquet"
        }

    def arrow(self, path, columns=None):
        with pq.ParquetFile(path, filesystem=self.fs) as file:
            return file.read(columns=columns)


def _fingerprint(segment):
    folder, tag = segment
    files = [segment_path(folder, tag, table) for table in SCHEMAS]
    files.extend([folder / f"jobs_{tag}.csv", folder.parent / "gfw_oil_sites.csv"])
    hashes = {}
    for path in files:
        digest = hashlib.sha256()
        if path.exists():
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            hashes[path.name] = digest.hexdigest()
        else:
            hashes[path.name] = None
    return dict(
        folder=str(folder.absolute()),
        tag=tag,
        files=hashes,
    )


def _key(segment):
    folder, tag = segment
    return tag[:10], tag[11:], folder.name


def _const_text(const) -> str:
    """Process-independent text for a code constant (set order varies by hash seed)."""
    if hasattr(const, "co_code"):
        return _code_digest(const)
    if isinstance(const, frozenset):
        return "frozenset" + repr(sorted(_const_text(item) for item in const))
    if isinstance(const, tuple):
        return "tuple" + repr([_const_text(item) for item in const])
    return repr(const)


def _code_digest(code) -> str:
    """Digest of a loaded function's bytecode, constants and names."""
    digest = hashlib.sha256(code.co_code)
    digest.update(repr((code.co_names, code.co_varnames)).encode())
    for const in code.co_consts:
        digest.update(_const_text(const).encode())
    return digest.hexdigest()


def _code_key() -> str:
    """Digest of the loaded code that decides store contents and encoding.

    A change to the merge rules, the schema or the libraries that infer dtypes
    must rebuild the store even when no segment changed. The digest describes
    what this process executes: this module's source as read at import and the
    bytecode of the three loaders, never the files currently on disk.
    """
    import build_viewer

    # The loaders decode through the store module build_viewer imported, which
    # differs from this one when a notebook re-runs this file's cells.
    parts = [
        _SOURCE_DIGEST,
        getattr(build_viewer.store, "_SOURCE_DIGEST", ""),
        pd.__version__,
        pa.__version__,
        np.__version__,
    ]
    parts.extend(
        _code_digest(function.__code__)
        + repr((function.__defaults__, function.__kwdefaults__))
        for function in (
            build_viewer.granule_owners,
            build_viewer.load_segments,
            build_viewer.load_copies,
        )
    )
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def row_nights(
    frame: pd.DataFrame,
    table: str,
    sites: pd.Series | None = None,
    fallback: pd.Series | None = None,
) -> pd.Series:
    """Evening dates at the site for passes, and at the centroid for copies."""
    if table == "site_nights":
        return frame.night
    lon = frame.centroid_lon if table == "copies" else frame.site_id.map(sites)
    if fallback is not None:
        lon = lon.fillna(fallback)
    utc = pd.to_datetime(frame.time_utc, utc=True, format="mixed").dt.tz_localize(None)
    local = (utc + pd.to_timedelta(lon / 15, "h") - pd.Timedelta(hours=12)).dt.strftime(
        "%Y-%m-%d"
    )
    missing = lon.isna()
    if missing.any():
        local.loc[missing] = utc.loc[missing].dt.strftime("%Y-%m-%d")
    return local


def _same_partition(arrow, old, metadata, night):
    if arrow.equals(old, check_metadata=False):
        return True
    # Global pandas dtype promotion can change an audit child's encoding
    # without changing this night's values. Its file can retain the old child.
    if len(arrow) != len(old):
        return False
    for col in metadata["columns"]:
        if (
            metadata["dtypes"][col] == "object"
            and col in arrow.column_names
            and arrow[col].type == AUDIT
            and not arrow[col].equals(old[col])
        ):
            return False
    metadata = dict(metadata)
    metadata.pop("index", None)
    if "night" in metadata["columns"]:
        dates = pa.array([pd.Timestamp(night).date()] * len(arrow), type=pa.date32())
        arrow, old = [t.append_column("night", dates) for t in (arrow, old)]
    left, right = [from_arrow(t, metadata) for t in (arrow, old)]
    return left.equals(right) and all(
        np.array_equal(
            left[c].to_numpy().view("uint64"), right[c].to_numpy().view("uint64")
        )
        for c in left.select_dtypes("float64")
    )


def _materialize(segments, storage, *, replace):
    # Reuse the canonical ownership/concat rules, including pandas' treatment
    # of empty frames, before comparing and replacing individual night files.
    import build_viewer

    if not segments or segments != sorted(segments, key=_key):
        raise ValueError("segments must be nonempty and in find_segments order")
    records = [_fingerprint(segment) for segment in segments]
    if isinstance(storage.fs, pafs.LocalFileSystem):
        storage.fs.create_dir(storage.root, recursive=True)
    manifest = storage.path("_segments.json")
    if storage.exists(manifest):
        storage.fs.delete_file(manifest)
    # Temp files a crashed store writer left: only this module's own names,
    # in the folders it writes. Local writers hold the root lock; a non-local
    # root has a single writer. Other files under the root are never touched.
    code = _code_key()
    for folder, name in [(storage.root, "_segments.json.")] + [
        (storage.path(table), "part.parquet.") for table in SCHEMAS
    ]:
        for info in storage.fs.get_file_info(
            pafs.FileSelector(
                folder, recursive=folder != storage.root, allow_not_found=True
            )
        ):
            base = posixpath.basename(info.path)
            if info.type == pafs.FileType.File and re.fullmatch(
                re.escape(name) + TEMP_SUFFIX, base
            ):
                storage.fs.delete_file(info.path)
    nights, over = build_viewer.load_segments(segments, _ordered=True)
    sites = pd.Series(dtype="float64")
    # The catalog used by build_data is authoritative if available.
    catalog = segments[0][0].parent / "gfw_oil_sites.csv"
    if catalog.exists():
        sites = pd.read_csv(catalog).set_index("site_id").lon
    descriptions = {}
    for table in ("site_nights", "overpasses", "copies"):
        if table == "site_nights":
            frame = nights
            del nights
        elif table == "overpasses":
            frame = over
            del over
        else:
            frame = build_viewer.load_copies(segments, _ordered=True)
        descriptions[table] = frame_metadata(frame, table)
        existing = storage.parts(table)
        fallback = None
        if table == "overpasses":
            missing = frame.site_id.map(sites).isna()
            if missing.any():
                fallback = pd.Series(np.nan, index=frame.index)
                for i, (folder, tag) in enumerate(segments):
                    rows = missing & frame._segment.eq(i)
                    if rows.any():
                        coords = read_segment(
                            folder, tag, "site_nights", columns=["site_id", "lon"]
                        )
                        coords = (
                            coords.dropna(subset=["lon"])
                            .drop_duplicates("site_id")
                            .set_index("site_id")
                            .lon
                        )
                        fallback.loc[rows] = frame.loc[rows, "site_id"].map(coords)
        partition = row_nights(frame, table, sites, fallback)
        if partition.isna().any():
            raise ValueError(f"{table} rows lack a night")
        for night, positions in partition.groupby(partition, sort=True).indices.items():
            part = frame.iloc[positions].reset_index(drop=True)
            arrow = to_arrow(part, table, ordered=True)
            if "night" in arrow.column_names:
                arrow = arrow.drop(["night"])
            path = storage.path(table, f"night={night}", "part.parquet")
            old = existing.pop(night, None)
            if (
                replace
                or old is None
                or not _same_partition(
                    arrow, storage.arrow(old), descriptions[table], night
                )
            ):
                storage.atomic(
                    path,
                    lambda stream: pq.write_table(
                        arrow, stream, compression="zstd", coerce_timestamps="us"
                    ),
                )
        for path in existing.values():
            storage.fs.delete_file(path)
        del frame
    if records != [_fingerprint(segment) for segment in segments]:
        raise RuntimeError(
            "segment exports changed during materialization; run sync again"
        )
    payload = json.dumps(
        dict(version=1, code=code, segments=records, tables=descriptions)
    ).encode()
    storage.atomic(manifest, lambda stream: stream.write(payload))


def rebuild(segments: list[tuple[Path, str]], root: Path | str = DEFAULT_ROOT) -> None:
    """Materialise the canonical merged tables, replacing every night file."""
    storage = _Store(root)
    with storage.lock():
        _materialize(segments, storage, replace=True)


def ingest(segment: tuple[Path, str], root: Path | str = DEFAULT_ROOT) -> None:
    """Append a segment in discovery order; replace only changed night files."""
    storage = _Store(root)
    with storage.lock():
        _ingest(segment, storage)


def _ingest(segment, storage):
    recorded = storage.manifest()
    previous = recorded["segments"] if recorded else []
    segments = [(Path(s["folder"]), s["tag"]) for s in previous]
    if segments and _key(segment) <= _key(segments[-1]):
        raise ValueError("segment does not sort last; use rebuild")
    if previous != [_fingerprint(s) for s in segments]:
        raise ValueError("ingested exports changed; use rebuild")
    _materialize([*segments, segment], storage, replace=False)


def sync(segments: list[tuple[Path, str]], root: Path | str = DEFAULT_ROOT) -> str:
    """Return 'noop', 'ingest' or 'rebuild' after reconciling segment fingerprints."""
    storage = _Store(root)
    with storage.lock():
        return _sync(segments, storage)


def _sync(segments, storage):
    recorded = storage.manifest()
    current = [_fingerprint(s) for s in segments]
    if (
        not recorded
        or recorded.get("code") != _code_key()
        or recorded["segments"] != current[: len(recorded["segments"])]
    ):
        _materialize(segments, storage, replace=True)
        return "rebuild"
    if recorded["segments"] == current:
        return "noop"
    for segment in segments[len(recorded["segments"]) :]:
        _ingest(segment, storage)
    return "ingest"


def read(
    table: str,
    root: Path | str = DEFAULT_ROOT,
    nights: list[str | date] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read selected partitions/columns in legacy row order with a RangeIndex."""
    storage = _Store(root)
    with storage.lock(shared=True):
        return _read(table, storage, nights, columns)


def load_tables(segments, root=DEFAULT_ROOT):
    """Hold one exclusive lock across sync and all three website reads."""
    storage = _Store(root)
    with storage.lock():
        _sync(segments, storage)
        return tuple(
            _read(table, storage) for table in ("site_nights", "overpasses", "copies")
        )


def _read(table, storage, nights=None, columns=None):
    if nights is not None:
        nights = {pd.Timestamp(night).date().isoformat() for night in nights}
    manifest = storage.manifest()
    if not manifest:
        raise FileNotFoundError("store has no completed manifest; run rebuild")
    metadata = dict(manifest["tables"][table])
    legacy_columns = [c for c in metadata["columns"] if c not in ORDER]
    selected = legacy_columns if columns is None else list(columns)
    unknown = set(selected) - set(legacy_columns)
    if unknown:
        raise KeyError(sorted(unknown))
    wanted = list(dict.fromkeys([*selected, *ORDER]))
    physical = [
        "_time_utc_text" if c == "time_utc" else c for c in wanted if c != "night"
    ]
    metadata["columns"] = wanted
    metadata.pop("index", None)
    frames = []
    for night, path in sorted(storage.parts(table).items()):
        if nights is not None and night not in nights:
            continue
        arrow = storage.arrow(path, physical)
        if "night" in wanted:
            arrow = arrow.append_column(
                "night",
                pa.array([pd.Timestamp(night).date()] * len(arrow), type=pa.date32()),
            )
        frames.append(from_arrow(arrow, metadata))
    if frames:
        frame = pd.concat(frames, ignore_index=True)
        frames.clear()
        frame.sort_values(ORDER, kind="stable", inplace=True)
    else:
        frame = pd.DataFrame(
            {c: pd.Series(dtype=metadata["dtypes"][c]) for c in wanted}
        )
    frame.drop(columns=ORDER, inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def main(
    base: Path = DEFAULT_ROOT.parent, action: str = "convert", root=None, force=False
):
    """Convert all segment CSVs (retained), or rebuild the store under base."""
    import build_viewer

    segments = build_viewer.find_segments(base)
    if action == "convert":
        return [
            sizes for segment in segments for sizes in convert(segment, force=force)
        ]
    if action == "rebuild":
        return rebuild(segments, root if root is not None else base / "store")
    raise ValueError("action must be convert or rebuild")


# %%
if __name__ == "__main__":
    main()
