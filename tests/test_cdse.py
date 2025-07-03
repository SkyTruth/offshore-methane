import pytest
from offshore_methane.cdse import parse_sid


def test_parse_sid_valid():
    sid = "20210101T123456_20210102T000000_T10ABC"
    parsed = parse_sid(sid)
    assert parsed == {
        "start": "20210101T123456",
        "proc": "20210102T000000",
        "tile": "10ABC",
        "date": "20210101",
    }


def test_parse_sid_invalid():
    with pytest.raises(ValueError):
        parse_sid("bad_sid")
