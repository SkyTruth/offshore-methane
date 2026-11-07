"""Access to NOAA Open Data Dissemination (NODD) VIIRS files on public S3.

Buckets are anonymous and free; they keep a rolling ~91 days. Files are read
over HTTP byte ranges, so h5py only downloads the datasets it touches.
"""

from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import requests

BUCKETS = {
    "SNPP": ("noaa-nesdis-snpp-pds", "npp"),
    "NOAA-20": ("noaa-nesdis-n20-pds", "j01"),
    "NOAA-21": ("noaa-nesdis-n21-pds", "j02"),
}
PRODUCTS = {
    "geo": "VIIRS-MOD-GEO-TC",
    "M10": "VIIRS-M10-SDR",
    "M11": "VIIRS-M11-SDR",
    "M12": "VIIRS-M12-SDR",
    "M13": "VIIRS-M13-SDR",
    "cloud": "VIIRS-JRR-CloudMask",
}
SDR_RE = re.compile(r"_(npp|j01|j02)_d(\d{8})_t(\d{7})_e(\d{7})_")
JRR_RE = re.compile(r"_(npp|j01|j02|n20|n21)_s(\d{8})(\d{7})_e(\d{8})(\d{7})_")
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def bucket_url(bucket: str) -> str:
    return f"https://{bucket}.s3.amazonaws.com"


def list_keys(bucket: str, prefix: str, session: requests.Session) -> list[str]:
    """All object keys under ``prefix`` (S3 ListObjectsV2, paginated)."""
    keys: list[str] = []
    params = {"list-type": "2", "prefix": prefix}
    while True:
        resp = session.get(bucket_url(bucket), params=params, timeout=60)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        keys += [el.text for el in root.iter(f"{S3_NS}Key")]
        token = root.find(f"{S3_NS}NextContinuationToken")
        if token is None:
            return keys
        params["continuation-token"] = token.text


def _hhmmsss(date: str, t: str) -> datetime:
    """'20261101', '0200577' -> datetime (tenths of a second)."""
    base = datetime.strptime(date, "%Y%m%d")
    return base + timedelta(
        hours=int(t[:2]), minutes=int(t[2:4]), seconds=int(t[4:6]) + int(t[6]) / 10
    )


def granule_index(
    satellite: str, day: datetime, session: requests.Session
) -> dict[str, dict]:
    """Map granule start (``dYYYYMMDD_tHHMMSSS``) to times and per-product keys."""
    bucket, _ = BUCKETS[satellite]
    folder = day.strftime("%Y/%m/%d")
    index: dict[str, dict] = {}
    for name, product in PRODUCTS.items():
        for key in list_keys(bucket, f"{product}/{folder}/", session):
            if not key.endswith((".h5", ".nc")):
                continue
            if name == "cloud":
                m = JRR_RE.search(key)
                if not m:
                    continue
                d, t = m.group(2), m.group(3)
            else:
                m = SDR_RE.search(key)
                if not m:
                    continue
                d, t = m.group(2), m.group(3)
            gid = f"d{d}_t{t}"
            if gid not in index:
                start = _hhmmsss(d, t)
                end = _hhmmsss(d, m.group(4) if name != "cloud" else m.group(5))
                if end <= start:
                    end += timedelta(days=1)
                index[gid] = {"start": start, "end": end}
            index[gid][name] = key
    return index


class HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable file over HTTP range requests with a block cache.

    Reads of at least ``block`` bytes go straight to the server; smaller reads
    (HDF5 metadata, small chunks) are served from cached blocks.
    """

    def __init__(self, url: str, session: requests.Session, block: int = 1 << 18):
        super().__init__()
        self.url, self.session, self.block = url, session, block
        resp = session.head(url, timeout=60)
        resp.raise_for_status()
        self.size = int(resp.headers["Content-Length"])
        self.pos = 0
        self.cache: dict[int, bytes] = {}
        self.bytes_fetched = 0
        self.requests = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}
        self.pos = base[whence] + offset
        return self.pos

    def _get(self, lo: int, hi: int) -> bytes:
        resp = self.session.get(
            self.url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=120
        )
        resp.raise_for_status()
        self.bytes_fetched += len(resp.content)
        self.requests += 1
        return resp.content

    def _read(self, start: int, n: int) -> bytes:
        if n >= self.block:
            return self._get(start, start + n - 1)
        first, last = start // self.block, (start + n - 1) // self.block
        parts = []
        for i in range(first, last + 1):
            if i not in self.cache:
                lo = i * self.block
                self.cache[i] = self._get(lo, min(self.size, lo + self.block) - 1)
            parts.append(self.cache[i])
        off = start - first * self.block
        return b"".join(parts)[off : off + n]

    def readinto(self, buf) -> int:
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        buf[:n] = self._read(self.pos, n)
        self.pos += n
        return n
