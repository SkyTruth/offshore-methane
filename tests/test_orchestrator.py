from offshore_methane.orchestrator import add_days_to_date


def test_add_days():
    assert add_days_to_date("2024-01-01", 7) == "2024-01-08"


def test_add_negative_days():
    assert add_days_to_date("2024-01-10", -2) == "2024-01-08"
