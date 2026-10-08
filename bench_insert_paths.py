"""Why the graph writers stage rows through a CSV buffer instead of using parameters.

This is the evidence behind `weknow._graph._staged_rows`. It exists because the obvious choices all
look right and all fail on DuckDB 1.5.5, and the failure is not where you would look for it.

Four shapes are timed against a table with the real `graph_relation` primary key, at the row count
NFCorpus actually produces (22,285 merged edges; `RUST_ROWS` can be raised to 69,460 to see the
scaling past the corpus):

  A. `executemany`                      - what the ORM unit of work does underneath, and what
                                          `weknow._insert.insert_documents` uses directly
  B. parametrised `INSERT ... VALUES`   - the safe, idiomatic bulk insert
  C. parametrised `INSERT ... SELECT FROM (VALUES ...)`
  D. literal `INSERT ... SELECT FROM (VALUES ...)` - unsafe by hand, but set-based
  E. pyarrow `register` + `INSERT ... SELECT`      - set-based, but pyarrow is not a dependency
  F. `csv` buffer -> `read_csv` -> `INSERT ... SELECT` - what `_staged_rows` does

Findings:
  * every *parametrised* multi-row insert is quadratic in the row count, so B and C are unusable at
    any useful batch size — the cost is in binding, not in the VALUES form;
  * the per-row path (A) is the ~45 rows/s that made a graph build take 25 minutes;
  * D and E are fast, but D means escaping corpus text by hand and E needs pyarrow;
  * F matches E's order of magnitude with no new dependency and with the stdlib `csv` module
    owning the quoting.

Run:
    .venv/Scripts/python.exe bench_insert_paths.py
"""

from __future__ import annotations

import csv
import io
import pathlib
import tempfile
import time

import duckdb

ROWS = 22_285
COLUMNS = "source_id,target_id,description,weight,strength,observations"
NASTY = ["plain", "it's got a quote", "double \" quote", "back\\slash", "new\nline", "中文 émoji ✅"]


def build_rows(count: int) -> list[tuple]:
    return [
        (f"source-{index:06d}", f"target-{index:06d}", NASTY[index % len(NASTY)], 1.0, 5.0, 2)
        for index in range(count)
    ]


def create_relation_table(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute("DROP TABLE IF EXISTS graph_relation")
    connection.execute(
        """CREATE TABLE graph_relation (
            source_id VARCHAR, target_id VARCHAR, description VARCHAR,
            weight DOUBLE, strength DOUBLE, observations BIGINT,
            PRIMARY KEY (source_id, target_id))"""
    )


def timed(connection: duckdb.DuckDBPyConnection, rows: list[tuple], label: str, run) -> None:
    create_relation_table(connection)
    start = time.perf_counter()
    run()
    elapsed = time.perf_counter() - start
    stored = connection.execute("SELECT count(*) FROM graph_relation").fetchone()[0]
    print(
        f"  {label:<34s} {elapsed:8.3f}s  ({len(rows) / elapsed:11,.0f} rows/s)  {stored:,} rows",
        flush=True,
    )


def scaling(connection: duckdb.DuckDBPyConnection) -> None:
    """B/C: the quadratic wall that rules out parametrised bulk inserts."""
    print("parametrised multi-row insert (quadratic in the row count):")
    connection.execute("DROP TABLE IF EXISTS probe")
    connection.execute("CREATE TABLE probe (a VARCHAR, b VARCHAR, c VARCHAR, d DOUBLE, e DOUBLE, f BIGINT)")
    for count in (200, 1000, 4000):
        placeholders = ",".join(["(?,?,?,?,?,?)"] * count)
        flat = [value for row in build_rows(count) for value in row]
        connection.execute("DELETE FROM probe")
        start = time.perf_counter()
        connection.execute(f"INSERT INTO probe VALUES {placeholders}", flat)  # noqa: S608
        values_elapsed = time.perf_counter() - start
        connection.execute("DELETE FROM probe")
        start = time.perf_counter()
        connection.execute(f"INSERT INTO probe SELECT * FROM (VALUES {placeholders})", flat)  # noqa: S608
        select_elapsed = time.perf_counter() - start
        print(
            f"  {count:6d} rows: INSERT ... VALUES {values_elapsed:7.3f}s"
            f" | INSERT ... SELECT FROM (VALUES ...) {select_elapsed:7.3f}s",
            flush=True,
        )
    print()


def main() -> None:
    path = pathlib.Path(tempfile.gettempdir()) / "bench_insert_paths.db"
    path.unlink(missing_ok=True)
    connection = duckdb.connect(str(path))
    print(f"duckdb {duckdb.__version__}, {ROWS:,} rows, composite primary key on the target\n")
    scaling(connection)

    rows = build_rows(ROWS)
    print("bulk paths:")

    def run_executemany() -> None:
        connection.executemany(
            f"INSERT INTO graph_relation ({COLUMNS}) VALUES (?,?,?,?,?,?)", rows
        )

    timed(connection, rows, "A. executemany (per row)", run_executemany)

    literals = ",".join("(" + ",".join("'" + str(v).replace("'", "''") + "'" for v in row) + ")" for row in rows)

    def run_literal() -> None:
        for offset in range(0, len(rows), 2_000):
            chunk = rows[offset : offset + 2_000]
            values = ",".join(
                "(" + ",".join("'" + str(v).replace("'", "''") + "'" for v in row) + ")"
                for row in chunk
            )
            connection.execute(f"INSERT INTO graph_relation ({COLUMNS}) SELECT * FROM (VALUES {values})")  # noqa: S608

    del literals
    timed(connection, rows, "D. literal VALUES (2k/statement)", run_literal)

    def run_arrow() -> None:
        import pyarrow as pa

        table = pa.table(
            {
                "source_id": pa.array([row[0] for row in rows]),
                "target_id": pa.array([row[1] for row in rows]),
                "description": pa.array([row[2] for row in rows]),
                "weight": pa.array([row[3] for row in rows], type=pa.float64()),
                "strength": pa.array([row[4] for row in rows], type=pa.float64()),
                "observations": pa.array([row[5] for row in rows], type=pa.int64()),
            }
        )
        connection.register("staging", table)
        connection.execute(f"INSERT INTO graph_relation ({COLUMNS}) SELECT {COLUMNS} FROM staging")
        connection.unregister("staging")

    timed(connection, rows, "E. pyarrow register", run_arrow)

    def run_csv() -> None:
        buffer = io.StringIO()
        csv.writer(buffer, quoting=csv.QUOTE_ALL).writerows(rows)
        relation = connection.read_csv(
            io.StringIO(buffer.getvalue()),
            header=False,
            auto_detect=False,
            delimiter=",",
            quotechar='"',
            escapechar='"',
            columns={
                "source_id": "VARCHAR",
                "target_id": "VARCHAR",
                "description": "VARCHAR",
                "weight": "DOUBLE",
                "strength": "DOUBLE",
                "observations": "BIGINT",
            },
            force_not_null=["source_id", "target_id", "description", "weight", "strength", "observations"],
        )
        # The relation object has to outlive the view it backs.
        relation.create_view("staging", replace=True)
        connection.execute(f"INSERT INTO graph_relation ({COLUMNS}) SELECT {COLUMNS} FROM staging")
        connection.execute("DROP VIEW staging")

    timed(connection, rows, "F. csv buffer -> read_csv (used)", run_csv)

    # The quoted-empty field must survive as an empty string, not a NULL.
    connection.execute("DROP TABLE IF EXISTS graph_relation")
    connection.execute("CREATE TABLE graph_relation (a VARCHAR NOT NULL, b VARCHAR NOT NULL)")
    buffer = io.StringIO()
    csv.writer(buffer, quoting=csv.QUOTE_ALL).writerows([("", "it's ok"), ("x", "")])
    relation = connection.read_csv(
        io.StringIO(buffer.getvalue()),
        header=False,
        auto_detect=False,
        delimiter=",",
        quotechar='"',
        escapechar='"',
        columns={"a": "VARCHAR", "b": "VARCHAR"},
        force_not_null=["a", "b"],
    )
    relation.create_view("staging", replace=True)
    connection.execute("INSERT INTO graph_relation SELECT a, b FROM staging")
    stored = connection.execute("SELECT a, b FROM graph_relation ORDER BY b").fetchall()
    print(f"\n  empty fields stay empty strings, not NULL: {stored}")
    connection.close()
    path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
