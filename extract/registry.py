import json


def top_measures(conn, n: int = 30) -> list[dict]:
    rows = conn.execute(
        "SELECT measure_key, label, aliases, unit_dim FROM measure_registry "
        "ORDER BY seen_count DESC LIMIT ?", (n,),
    ).fetchall()
    return [
        {"measure_key": r["measure_key"], "label": r["label"],
         "aliases": json.loads(r["aliases"] or "[]"), "unit_dim": r["unit_dim"]}
        for r in rows
    ]


def register_measure(conn, measure_key: str, label: str, alias: str | None, unit_dim: str | None) -> None:
    row = conn.execute(
        "SELECT aliases FROM measure_registry WHERE measure_key = ?", (measure_key,)
    ).fetchone()
    if row is None:
        aliases = [alias] if alias else []
        conn.execute(
            "INSERT INTO measure_registry (measure_key, label, aliases, unit_dim, seen_count) "
            "VALUES (?, ?, ?, ?, 1)",
            (measure_key, label, json.dumps(aliases), unit_dim),
        )
    else:
        aliases = json.loads(row["aliases"] or "[]")
        if alias and alias not in aliases:
            aliases.append(alias)
        conn.execute(
            "UPDATE measure_registry SET aliases = ?, seen_count = seen_count + 1 WHERE measure_key = ?",
            (json.dumps(aliases), measure_key),
        )
    conn.commit()
