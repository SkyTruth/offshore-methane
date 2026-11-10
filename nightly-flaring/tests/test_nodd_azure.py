"""Network-free checks of the Azure NODD source (D63)."""

from datetime import datetime

import pytest

import nodd

PAGE1 = b"""<?xml version="1.0" encoding="utf-8"?>
<EnumerationResults ServiceEndpoint="https://jpss.blob.core.windows.net/"
 ContainerName="noaa-20"><Prefix>VIIRS-M12-SDR/2024/03/15/</Prefix><Blobs>
<Blob><Name>VIIRS-M12-SDR/2024/03/15/SVM12_j01_d20240315_t0000455_e0002100_b32754_c1_oeac_ops.h5</Name>
<Properties><Content-Length>4871013</Content-Length></Properties></Blob>
</Blobs><NextMarker>page2</NextMarker></EnumerationResults>"""
PAGE2 = b"""<?xml version="1.0" encoding="utf-8"?>
<EnumerationResults ContainerName="noaa-20"><Blobs>
<Blob><Name>VIIRS-M12-SDR/2024/03/15/SVM12_j01_d20240315_t0002113_e0003340_b32754_c2_oeac_ops.h5</Name>
<Properties><Content-Length>4810773</Content-Length></Properties></Blob>
</Blobs><NextMarker /></EnumerationResults>"""


class FakeResponse:
    def __init__(self, content: bytes):
        self.content, self.status_code = content, 200

    def raise_for_status(self) -> None:
        pass


class FakeSession:
    def __init__(self, pages: list[bytes]):
        self.pages, self.calls = list(pages), []

    def get(self, url, params=None, **kwargs):
        self.calls.append((url, dict(params or {})))
        return FakeResponse(self.pages.pop(0))


def test_bucket_names_and_urls():
    assert nodd.bucket_name("NOAA-20") == "noaa-nesdis-n20-pds"
    assert nodd.bucket_name("NOAA-20", "azure") == "azure:noaa-20"
    assert nodd.bucket_url("noaa-nesdis-n20-pds") == (
        "https://noaa-nesdis-n20-pds.s3.amazonaws.com"
    )
    assert nodd.bucket_url("azure:snpp") == "https://jpss.blob.core.windows.net/snpp"
    with pytest.raises(ValueError):
        nodd.bucket_name("NOAA-20", "gcs")


def test_azure_listing_follows_next_marker():
    session = FakeSession([PAGE1, PAGE2])
    keys = nodd.list_keys("azure:noaa-20", "VIIRS-M12-SDR/2024/03/15/", session)
    assert [k.split("_")[3] for k in keys] == ["t0000455", "t0002113"]
    first, second = session.calls
    assert first[0] == "https://jpss.blob.core.windows.net/noaa-20"
    assert first[1]["comp"] == "list" and "marker" not in first[1]
    assert second[1]["marker"] == "page2"


def test_granule_index_uses_azure_keys(monkeypatch):
    listed = []

    def fake_list(bucket, prefix, session):
        listed.append(bucket)
        product = prefix.split("/")[0]
        if product == "VIIRS-JRR-CloudMask":
            name = "JRR-CloudMask_v3r2_j01_s202403150000455_e202403150002100_c1.nc"
        else:
            band = {"VIIRS-MOD-GEO-TC": "GMTCO"}.get(product, "SV" + product[6:9])
            name = f"{band}_j01_d20240315_t0000455_e0002100_b32754_c1_oeac_ops.h5"
        return [prefix + name]

    monkeypatch.setattr(nodd, "list_keys", fake_list)
    index = nodd.granule_index("NOAA-20", datetime(2024, 3, 15), None, "azure")
    assert set(listed) == {"azure:noaa-20"}
    (g,) = index.values()
    assert set(nodd.PRODUCTS) <= set(g)
    assert g["start"] == datetime(2024, 3, 15, 0, 0, 45, 500000)
