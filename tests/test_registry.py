from app.db import get_connection, init_db
from extract.registry import top_measures, register_measure


def make_conn():
    conn = get_connection(":memory:")
    init_db(conn)
    return conn


def test_register_new_measure_creates_row():
    conn = make_conn()
    register_measure(conn, "revenue_from_operations", "Revenue from Operations", "revenue", "currency_inr")
    measures = top_measures(conn)
    assert len(measures) == 1
    assert measures[0]["measure_key"] == "revenue_from_operations"
    assert "revenue" in measures[0]["aliases"]


def test_register_existing_measure_increments_seen_count_and_adds_alias():
    conn = make_conn()
    register_measure(conn, "real_gdp_growth", "Real GDP Growth", "GDP growth", "percent")
    register_measure(conn, "real_gdp_growth", "Real GDP Growth", "real gross domestic product growth", "percent")
    row = conn.execute("SELECT seen_count, aliases FROM measure_registry WHERE measure_key = ?",
                        ("real_gdp_growth",)).fetchone()
    assert row["seen_count"] == 2
    import json
    aliases = json.loads(row["aliases"])
    assert "GDP growth" in aliases
    assert "real gross domestic product growth" in aliases


def test_register_duplicate_alias_not_added_twice():
    conn = make_conn()
    register_measure(conn, "cad_pct_gdp", "Current Account Deficit (% of GDP)", "CAD", "percent")
    register_measure(conn, "cad_pct_gdp", "Current Account Deficit (% of GDP)", "CAD", "percent")
    row = conn.execute("SELECT aliases FROM measure_registry WHERE measure_key = ?", ("cad_pct_gdp",)).fetchone()
    import json
    assert json.loads(row["aliases"]).count("CAD") == 1


def test_top_measures_orders_by_seen_count_descending():
    conn = make_conn()
    register_measure(conn, "rare_measure", "Rare", "rare", None)
    for _ in range(3):
        register_measure(conn, "common_measure", "Common", "common", None)
    measures = top_measures(conn, n=10)
    assert measures[0]["measure_key"] == "common_measure"


def test_top_measures_respects_limit():
    conn = make_conn()
    for i in range(5):
        register_measure(conn, f"measure_{i}", f"Measure {i}", None, None)
    assert len(top_measures(conn, n=2)) == 2
