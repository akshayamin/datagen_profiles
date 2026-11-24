"""Mine the customer's SQL: join keys, self-joins, filter literals, date ranges, casts and column roles (sqlglot).

Accepts one SQL text or a folder of ``*.sql`` files. Each statement is parsed on its own, so one broken statement is
reported and skipped instead of failing everything. CTE and subquery aliases that select from a single source table
are resolved to that table, so ``WITH f AS (SELECT ... FROM sales_txn) ... JOIN categories c ON f.category_cd = ...``
is still seen as a join on ``sales_txn.category_cd``.
"""

from __future__ import annotations

import glob
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

# wrappers that don't change which column a predicate is about: trim(a.x) = trim(b.y), cast(a.x as int) = ...
_TRANSPARENT = (exp.Paren, exp.Cast, exp.TryCast, exp.Trim, exp.Upper, exp.Lower)
_COMPARISONS = {exp.EQ: "=", exp.NEQ: "<>", exp.GT: ">", exp.GTE: ">=", exp.LT: "<", exp.LTE: "<="}
ROLES = ("join", "self_join", "filter", "partition", "order", "group", "count_distinct")


@dataclass
class QueryFacts:
    """What the SQL tells us about the source tables. Table names are short, lower-case names."""

    queries: list = field(default_factory=list)  # [{name, title, tables}]
    joins: list = field(default_factory=list)  # [{left_table, left_column, right_table, right_column, queries}]
    self_joins: list = field(default_factory=list)  # [{table, left_column, right_column, queries}]
    literals: dict = field(default_factory=lambda: defaultdict(set))  # (table, column) -> {values}
    ranges: list = field(default_factory=list)  # [{table, column, op, value, query}]
    cross_predicates: list = field(default_factory=list)
    casts: dict = field(default_factory=lambda: defaultdict(set))  # (table, column) -> {"DOUBLE", "DATE", ...}
    roles: dict = field(default_factory=lambda: defaultdict(set))  # (table, column) -> {"join", "partition", ...}
    parse_errors: list = field(default_factory=list)  # [{query, error}]
    unresolved_tables: set = field(default_factory=set)

    def critical(self, table: str) -> dict[str, set]:
        """Columns of ``table`` the workload depends on, with their roles."""
        return {c: r for (t, c), r in self.roles.items() if t == table}

    def tables(self) -> set[str]:
        return {t for q in self.queries for t in q["tables"]}

    def to_dict(self) -> dict:
        return {
            "queries": self.queries,
            "joins": self.joins,
            "self_joins": self.self_joins,
            "literals": [{"table": t, "column": c, "values": sorted(map(str, v))}
                         for (t, c), v in sorted(self.literals.items())],
            "ranges": self.ranges,
            "cross_predicates": self.cross_predicates,
            "casts": [{"table": t, "column": c, "as": sorted(v)} for (t, c), v in sorted(self.casts.items())],
            "critical_columns": [{"table": t, "column": c, "roles": sorted(r)}
                                 for (t, c), r in sorted(self.roles.items())],
            "parse_errors": self.parse_errors,
            "unresolved_tables": sorted(self.unresolved_tables),
        }


def read_sql(path: str) -> list[tuple[str, str]]:
    """(name, text) for a single .sql file or every *.sql in a folder and its subfolders (sorted)."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "*.sql"), recursive=True))
    else:
        files = [path]
    return [(os.path.splitext(os.path.basename(f))[0], open(f).read()) for f in files]


def _statements(name: str, text: str):
    """Split a file into statements (sqlglot first, then naive ';' splitting so one bad statement can't hide others)."""
    parts = [p for p in re.split(r";\s*(?:\n|$)", text) if re.sub(r"--[^\n]*", "", p).strip()]
    for i, part in enumerate(parts, 1):
        label = name if len(parts) == 1 else f"{name}#{i}"
        yield label, part


def _unwrap(node):
    while isinstance(node, _TRANSPARENT):
        node = node.this
    return node


def _literal_value(node):
    node = node.this if isinstance(node, exp.Paren) else node
    if isinstance(node, exp.Cast) and isinstance(node.this, exp.Literal):
        return node.this.this  # DATE'2025-01-01' -> '2025-01-01'
    if isinstance(node, exp.Boolean):
        return node.this
    if isinstance(node, exp.Neg) and isinstance(node.this, exp.Literal) and not node.this.is_string:
        return -float(node.this.this)
    if isinstance(node, exp.Literal):
        if node.is_string:
            return node.this
        num = node.this
        return int(num) if re.fullmatch(r"-?\d+", num) else float(num)
    return None


def _title(stmt, text: str) -> str:
    for comment in stmt.comments or []:
        t = comment.strip(" -\n")
        if t:
            return t.splitlines()[0].strip(" -")
    m = re.search(r"--\s*(.+)", text)
    return m.group(1).strip(" -") if m else ""


def analyze_queries(sql, table_columns: dict[str, list[str]], dialect: str = "databricks") -> QueryFacts:
    """Analyze SQL text, or a list of (name, text) pairs from ``read_sql``.

    ``table_columns`` maps each source table's short name to its column names.
    """
    sources = [("sql", sql)] if isinstance(sql, str) else list(sql)
    known = {t.lower(): {c.lower() for c in cols} for t, cols in table_columns.items()}
    facts = QueryFacts()
    joins, self_joins = defaultdict(set), defaultdict(set)

    for name, text in sources:
        for label, part in _statements(name, text):
            try:
                stmt = sqlglot.parse_one(part, read=dialect)
            except Exception as e:  # noqa: BLE001 - report and move on
                facts.parse_errors.append({"query": label, "error": str(e).splitlines()[0][:200]})
                continue
            if stmt is None or not stmt.find(exp.Select):
                continue
            _analyze_statement(stmt, label, part, known, facts, joins, self_joins)

    facts.joins = [{"left_table": a[0], "left_column": a[1], "right_table": b[0], "right_column": b[1],
                    "queries": sorted(q)} for (a, b), q in sorted(joins.items())]
    facts.self_joins = [{"table": a[0], "left_column": a[1], "right_column": b[1], "queries": sorted(q)}
                        for (a, b), q in sorted(self_joins.items())]
    return facts


def _analyze_statement(stmt, label, text, known, facts, joins, self_joins):
    ctes = {c.alias.lower(): c.this for c in stmt.find_all(exp.CTE)}

    def cte_base(name, seen=()):
        """The single source table a CTE selects from (following nested CTEs), if there is exactly one."""
        body = ctes.get(name)
        if body is None or name in seen:
            return None
        bases = set()
        for t in body.find_all(exp.Table):
            n = t.name.lower()
            if n in known:
                bases.add(n)
            elif n in ctes:
                b = cte_base(n, seen + (name,))
                bases |= {b} if b else {None}
        return next(iter(bases)) if len(bases) == 1 and None not in bases else None

    alias, in_scope = {}, set()
    for t in stmt.find_all(exp.Table):
        n = t.name.lower()
        base = n if n in known else cte_base(n)
        if base:
            alias.setdefault(t.alias_or_name.lower(), base)
            alias.setdefault(n, base)
            in_scope.add(base)
        elif n and n not in ctes:
            facts.unresolved_tables.add(".".join(p for p in (t.catalog, t.db, t.name) if p))
    # derived tables: (SELECT ... FROM one_table) alias
    for sq in stmt.find_all(exp.Subquery):
        if sq.alias:
            bases = {t.name.lower() for t in sq.find_all(exp.Table) if t.name.lower() in known}
            if len(bases) == 1:
                alias.setdefault(sq.alias.lower(), next(iter(bases)))
    if not in_scope:
        return
    facts.queries.append({"name": label, "title": _title(stmt, text), "tables": sorted(in_scope)})

    def resolve(node):
        node = _unwrap(node)
        if not isinstance(node, exp.Column):
            return None
        col, qual = node.name.lower(), node.table.lower()
        if qual:
            tbl = alias.get(qual)
            return (tbl, col) if tbl and col in known[tbl] else None
        owners = [t for t in in_scope if col in known[t]]
        return (owners[0], col) if len(owners) == 1 else None

    def role(col, r):
        if col:
            facts.roles[col].add(r)

    for node in stmt.find_all(*_COMPARISONS):
        op = _COMPARISONS[type(node)]
        left, right = resolve(node.this), resolve(node.expression)
        if left and right:
            key = tuple(sorted([left, right]))
            if left[0] == right[0]:
                if op == "=":
                    self_joins[key].add(label)
                    role(left, "self_join")
                    role(right, "self_join")
                continue
            if op == "=":
                joins[key].add(label)
                role(left, "join")
                role(right, "join")
            else:
                facts.cross_predicates.append({"left_table": left[0], "left_column": left[1], "op": op,
                                               "right_table": right[0], "right_column": right[1], "query": label})
            continue
        col, lit = (left, _literal_value(node.expression)) if left else (right, _literal_value(node.this))
        if not col or lit is None:
            continue
        role(col, "filter")
        if op in ("=", "<>"):
            facts.literals[col].add(lit)
        else:
            facts.ranges.append({"table": col[0], "column": col[1], "op": op, "value": lit, "query": label})

    for node in stmt.find_all(exp.In):
        col = resolve(node.this)
        if col:
            role(col, "filter")
            for item in node.expressions:
                val = _literal_value(item)
                if val is not None:
                    facts.literals[col].add(val)

    for node in stmt.find_all(exp.Between):
        col = resolve(node.this)
        lo, hi = _literal_value(node.args["low"]), _literal_value(node.args["high"])
        if col and lo is not None and hi is not None:
            role(col, "filter")
            facts.ranges.append({"table": col[0], "column": col[1], "op": "between", "value": [lo, hi],
                                 "query": label})

    for w in stmt.find_all(exp.Window):
        for p in w.args.get("partition_by") or []:
            role(resolve(p), "partition")
        order = w.args.get("order")
        for o in (order.expressions if order else []):
            role(resolve(o.this), "order")
    for g in stmt.find_all(exp.Group):
        for c in g.find_all(exp.Column):
            role(resolve(c), "group")
    for cnt in stmt.find_all(exp.Count):
        if isinstance(cnt.this, exp.Distinct):
            for e in cnt.this.expressions:
                role(resolve(e), "count_distinct")

    for c in stmt.find_all(exp.TryCast, exp.Cast):
        col = resolve(c.this)
        if col and c.args.get("to") is not None:
            facts.casts[col].add(c.to.sql().upper())
    for fn in stmt.find_all(exp.Func):
        name = fn.sql_name().upper().replace("_", "")
        if name in ("TSORDSTODATE", "TODATE", "STRTODATE", "TSORDSTOTIMESTAMP", "TOTIMESTAMP") and fn.args.get("this"):
            col = resolve(fn.this)
            if col:
                facts.casts[col].add("DATE")


def rewrite_table_names(sql, mapping: dict[str, str], dialect: str = "databricks") -> list[tuple[str, str]]:
    """(name, statement) for every SELECT, with table names replaced (short source name -> fully-qualified target).

    Statements that don't parse are skipped.
    """
    sources = [("sql", sql)] if isinstance(sql, str) else list(sql)
    out = []
    for name, text in sources:
        for label, part in _statements(name, text):
            try:
                stmt = sqlglot.parse_one(part, read=dialect)
            except Exception:  # noqa: BLE001
                continue
            if stmt is None or not stmt.find(exp.Select):
                continue
            stmt = stmt.copy()
            for tbl in stmt.find_all(exp.Table):
                target = mapping.get(tbl.name.lower())
                if target:
                    new = exp.to_table(target, dialect=dialect)
                    parts = target.split(".")
                    tbl.set("this", new.this)
                    tbl.set("db", new.args.get("db") if len(parts) > 1 else None)
                    tbl.set("catalog", new.args.get("catalog") if len(parts) > 2 else None)
            out.append((label, stmt.sql(dialect=dialect)))
    return out
