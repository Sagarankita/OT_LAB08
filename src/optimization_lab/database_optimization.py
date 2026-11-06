"""Deterministic SQLite index-selection experiment for Track H."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class CandidateIndex:
    name: str
    table: str
    columns: tuple[str, ...]

    @property
    def sql(self) -> str:
        return f"CREATE INDEX {self.name} ON {self.table}({', '.join(self.columns)})"


@dataclass(frozen=True)
class QuerySpec:
    name: str
    sql: str
    params: tuple[object, ...]


CANDIDATES = (
    CandidateIndex("idx_orders_customer_status", "orders", ("customer_id", "status")),
    CandidateIndex("idx_orders_status_day", "orders", ("status", "created_day")),
    CandidateIndex("idx_orders_product", "orders", ("product_id",)),
    CandidateIndex("idx_products_category", "products", ("category", "id")),
    CandidateIndex("idx_orders_status_amount_customer", "orders",
                   ("status", "amount_cents", "customer_id")),
    CandidateIndex("idx_orders_day", "orders", ("created_day", "id")),
    CandidateIndex("idx_customers_region", "customers", ("region", "id")),
)

QUERIES = (
    QuerySpec("customer_status", """
        SELECT status, COUNT(*), SUM(amount_cents)
        FROM orders WHERE customer_id BETWEEN ? AND ?
        GROUP BY status ORDER BY status
    """, (1200, 1300)),
    QuerySpec("status_date", """
        SELECT created_day, COUNT(*), SUM(amount_cents)
        FROM orders WHERE status=? AND created_day BETWEEN ? AND ?
        GROUP BY created_day ORDER BY created_day
    """, ("shipped", 200, 299)),
    QuerySpec("category_join", """
        SELECT p.category, COUNT(*), SUM(o.quantity)
        FROM products p JOIN orders o ON o.product_id=p.id
        WHERE p.category IN (?, ?)
        GROUP BY p.category ORDER BY p.category
    """, ("cat03", "cat11")),
    QuerySpec("status_amount", """
        SELECT customer_id, COUNT(*), CAST(AVG(amount_cents) AS INTEGER)
        FROM orders WHERE status=? AND amount_cents>=?
        GROUP BY customer_id ORDER BY customer_id LIMIT 100
    """, ("paid", 45000)),
    QuerySpec("recent_window", """
        SELECT id, customer_id, amount_cents
        FROM orders WHERE created_day BETWEEN ? AND ?
        ORDER BY created_day, id LIMIT 250
    """, (500, 509)),
    QuerySpec("region_join", """
        SELECT c.region, COUNT(*), SUM(o.amount_cents)
        FROM customers c JOIN orders o ON o.customer_id=c.id
        WHERE c.region=? AND o.status=?
        GROUP BY c.region ORDER BY c.region
    """, ("region4", "shipped")),
)


def connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA automatic_index=ON")
    return connection


def create_dataset(path: Path, seed: int, order_count: int,
                   customer_count: int = 5000, product_count: int = 1000) -> dict[str, int]:
    if order_count <= 0 or customer_count <= 0 or product_count <= 0:
        raise ValueError("dataset counts must be positive")
    if path.exists():
        raise FileExistsError(path)
    connection = sqlite3.connect(path)
    connection.executescript("""
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        CREATE TABLE customers(
          id INTEGER PRIMARY KEY, region TEXT NOT NULL, segment TEXT NOT NULL
        );
        CREATE TABLE products(
          id INTEGER PRIMARY KEY, category TEXT NOT NULL, brand TEXT NOT NULL
        );
        CREATE TABLE orders(
          id INTEGER PRIMARY KEY,
          customer_id INTEGER NOT NULL REFERENCES customers(id),
          product_id INTEGER NOT NULL REFERENCES products(id),
          status TEXT NOT NULL,
          created_day INTEGER NOT NULL,
          amount_cents INTEGER NOT NULL,
          quantity INTEGER NOT NULL
        );
    """)
    connection.executemany(
        "INSERT INTO customers VALUES(?,?,?)",
        ((i, f"region{(i * 7 + seed) % 10}", f"segment{(i * 3 + seed) % 5}")
         for i in range(1, customer_count + 1)),
    )
    connection.executemany(
        "INSERT INTO products VALUES(?,?,?)",
        ((i, f"cat{(i * 11 + seed) % 20:02d}", f"brand{(i * 5 + seed) % 40:02d}")
         for i in range(1, product_count + 1)),
    )
    statuses = ("new", "paid", "shipped", "cancelled")
    batch = []
    for i in range(1, order_count + 1):
        batch.append((
            i, (i * 37 + seed) % customer_count + 1,
            (i * 53 + seed * 3) % product_count + 1,
            statuses[(i * 7 + i // 13 + seed) % len(statuses)],
            (i * 17 + seed) % 730,
            (i * 97 + seed * 11) % 50000 + 100,
            i % 5 + 1,
        ))
        if len(batch) == 5000:
            connection.executemany("INSERT INTO orders VALUES(?,?,?,?,?,?,?)", batch)
            batch.clear()
    if batch:
        connection.executemany("INSERT INTO orders VALUES(?,?,?,?,?,?,?)", batch)
    connection.commit()
    connection.execute("ANALYZE")
    connection.commit()
    counts = {table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
              for table in ("customers", "products", "orders")}
    connection.close()
    return counts


def create_indexes(connection: sqlite3.Connection,
                   names: Sequence[str]) -> list[dict[str, object]]:
    mapping = {candidate.name: candidate for candidate in CANDIDATES}
    if len(set(names)) != len(names) or any(name not in mapping for name in names):
        raise ValueError("index names must be unique known candidates")
    details = []
    for name in names:
        connection.execute(mapping[name].sql)
        details.append({"name": name, "sql": mapping[name].sql})
    connection.commit()
    connection.execute("ANALYZE")
    connection.commit()
    return details


def index_storage(connection: sqlite3.Connection, names: Sequence[str]) -> dict[str, int]:
    storage = {}
    for name in names:
        row = connection.execute(
            "SELECT COALESCE(SUM(pgsize),0), COUNT(*) FROM dbstat WHERE name=?", (name,)
        ).fetchone()
        storage[name] = int(row[0])
    return storage


def query_plan(connection: sqlite3.Connection, query: QuerySpec) -> list[list[object]]:
    return [list(row) for row in connection.execute(
        "EXPLAIN QUERY PLAN " + query.sql, query.params
    ).fetchall()]


def execute_query(connection: sqlite3.Connection, query: QuerySpec) -> list[tuple[object, ...]]:
    return connection.execute(query.sql, query.params).fetchall()


def rows_hash(rows: Sequence[Sequence[object]]) -> str:
    # All experiment queries have explicit ORDER BY, so this is an
    # ordering-aware correctness check rather than a multiset comparison.
    payload = json.dumps(list(rows), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def greedy_select(items: Sequence[dict[str, int]], budget_pages: int) -> list[str]:
    if budget_pages < 0:
        raise ValueError("budget_pages must be non-negative")
    ranked = sorted(items, key=lambda item: (
        -(item["benefit_us"] / item["pages"] if item["pages"] else 0),
        -item["benefit_us"], item["name"],
    ))
    selected = []
    used = 0
    for item in ranked:
        if item["benefit_us"] > 0 and used + item["pages"] <= budget_pages:
            selected.append(item["name"])
            used += item["pages"]
    chosen = set(selected)
    return [item["name"] for item in items if item["name"] in chosen]


def dynamic_programming_select(items: Sequence[dict[str, int]],
                               budget_pages: int) -> list[str]:
    """Exact 0/1 knapsack over measured page cost and individual benefit."""
    if budget_pages < 0:
        raise ValueError("budget_pages must be non-negative")
    # value, tuple(names), used pages; lexicographic tie-breaking is deterministic.
    states: list[tuple[int, tuple[str, ...], int] | None] = [None] * (budget_pages + 1)
    states[0] = (0, (), 0)
    for item in items:
        weight, value, name = item["pages"], item["benefit_us"], item["name"]
        for capacity in range(budget_pages, weight - 1, -1):
            previous = states[capacity - weight]
            if previous is None:
                continue
            candidate = (previous[0] + value, tuple(sorted((*previous[1], name))),
                         previous[2] + weight)
            current = states[capacity]
            if current is None or candidate[0] > current[0] or (
                    candidate[0] == current[0] and candidate[1] < current[1]):
                states[capacity] = candidate
    feasible = [state for state in states if state is not None]
    best = min(feasible, key=lambda state: (-state[0], state[1], state[2]))
    chosen = set(best[1])
    return [item["name"] for item in items if item["name"] in chosen]
