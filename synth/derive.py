"""Turn profiles + relationships into a reviewable config wrapping a dbldatagen-core ``DataGenPlan``.

Config layout (YAML):

* ``plan``          - a valid ``DataGenPlan`` (editable: weights, ranges, row counts, ...)
* ``post``          - what ``generate`` does around the engine: tables to copy as-is, helper columns to drop, final
                      column order and exact DDL types, partitioning, comments, keys to declare, target names
* ``domains``       - how each key domain is laid out (index ranges, match rates, value function)
* ``provenance``    - per column: the rule chosen and the evidence behind it
* ``relationships`` / ``queries`` / ``warnings`` - the profiling findings, for review

**Key domains.** Every key, ID or high-cardinality code column is generated in two steps: a *domain index* (an
integer drawn from a calibrated distribution, or the row number for unique keys) and a *value function* shared by
every column of the domain - an integer lattice that keeps the source key range and extends it, the source's own value
list, or a format-preserving string. Columns that join in the source share a domain, so matches, partial matches,
overlaps and self-join fan-out are reproduced, and values the SQL filters on are placed at fixed indices.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field

from dbldatagen.core import ColumnSpec, DataGenPlan, TableSpec, generate, resolve_plan
from dbldatagen.core.spec.schema import (
    DataType,
    ExpressionColumn,
    Exponential,
    LogNormal,
    Normal,
    RangeColumn,
    SequenceColumn,
    Uniform,
    ValuesColumn,
    WeightedValues,
    Zipf,
)
from pyspark.sql import functions as F

from .profiler import CATEGORICAL_PROFILE_MAX, ColumnProfile, TableProfile, day_sql
from .relationships import Relationships

NULL_FLOOR = 1e-4  # dbldatagen core's null-mask granularity
CODE_KEY = re.compile(r"(_cd$|_code$|_cd_|_typ$|_type$|_grp$|_ind$)")
LONG_LIST = 64  # longer weighted lists use a binary search: core's WeightedValues emits one CASE branch per value
CAL_MAX_VALUES = 100_000
CAL_MAX_ROWS = 400_000
PRIME = 1_000_000_007
SPREAD_MEAN = 2.5  # columns with this many rows per value or fewer spread rows evenly instead of random draws


def coprime(n: int, start: int = 2_654_435_761) -> int:
    """A multiplier coprime with n, so pmod(i * a, n) visits every residue."""
    a = start % max(n, 1) or 1
    while math.gcd(a, n) != 1:
        a += 1
    return a


@dataclass
class Settings:
    scale: float = 3.0
    date_mode: str = "extend"  # extend: event dates span `scale` x the source window; densify: same window
    top_n: int = 100
    seed: int = 42
    source_prefix: str = ""
    target_prefix: str = "customer_3x_"
    target_catalog: str = ""
    target_schema: str = ""
    copy_max_rows: int = 1000  # code lookups at or below this size are copied unchanged (with their parents)
    copy_tables: list | None = None  # explicit list of source tables to copy; None = automatic
    grow_ids: bool = True  # ID-like columns get `scale` x the distinct values; codes keep theirs
    row_overrides: dict = field(default_factory=dict)

    def target_short(self, source_short: str) -> str:
        rest = source_short[len(self.source_prefix):] if self.source_prefix and \
            source_short.startswith(self.source_prefix) else source_short
        return f"{self.target_prefix}{rest}"

    def target_fqn(self, source_short: str) -> str:
        return ".".join(p for p in (self.target_catalog, self.target_schema, self.target_short(source_short)) if p)


# ------------------------------------------------------------------------------------------------ SQL builders


def sql_str(value) -> str:
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def icdf_sql(u: str, knots: list) -> str:
    """Piecewise-linear inverse CDF mapping a uniform ``u`` in [0,1] onto quantile knots. NULL u stays NULL."""
    pts = []
    for p, x in sorted(knots, key=lambda k: k[0]):
        if pts and p <= pts[-1][0]:
            continue
        pts.append((float(p), float(x)))
    if len(pts) == 1:
        return f"(CASE WHEN {u} IS NULL THEN NULL ELSE {pts[0][1]!r} END)"
    branches = [f"WHEN {u} <= {p1!r} THEN {x0!r} + ({u} - {p0!r}) * {(x1 - x0) / (p1 - p0)!r}"
                for (p0, x0), (p1, x1) in zip(pts, pts[1:])]
    return f"(CASE WHEN {u} IS NULL THEN NULL {' '.join(branches)} ELSE {pts[-1][1]!r} END)"


def lattice_sql(idx: str, base: int, step: int, literals: list[int], wrap: int | None = None) -> str:
    """Integer keys that keep the source range and extend it: literals at indices 0..n-1, then base + m*step.

    Lattice positions equal to a literal are skipped, so every value is produced exactly once.
    """
    n = len(literals)
    on_lattice = sorted({(v - base) // step for v in literals if v >= base and (v - base) % step == 0})
    m = f"({idx} - {n})"
    skip = " + ".join([m] + [f"(CASE WHEN {m} >= {o - k} THEN 1 ELSE 0 END)" for k, o in enumerate(on_lattice)])
    lat = f"({base} + ({skip}) * {step})" if not wrap else f"({base} + pmod(({skip}) * {step}, {wrap}))"
    if not n:
        return lat
    whens = " ".join(f"WHEN {idx} = {i} THEN {v}" for i, v in enumerate(literals))
    return f"(CASE {whens} ELSE {lat} END)"


def int_type_max(type_text: str) -> int | None:
    """Largest value an integer-like column type can hold (None for non-integer types)."""
    t = type_text.lower().replace(" ", "")
    fixed = {"tinyint": 127, "byte": 127, "smallint": 32767, "short": 32767, "int": 2 ** 31 - 1, "integer": 2 ** 31 - 1,
             "bigint": 2 ** 63 - 1, "long": 2 ** 63 - 1}
    if t in fixed:
        return fixed[t]
    m = re.match(r"decimal\((\d+),0\)", t)
    return 10 ** int(m.group(1)) - 1 if m else None


def affine_sql(idx: str, lo: int, span: int) -> str:
    """Bijective scramble of indices into [lo, lo + span): keeps a code column's range and distinct count."""
    a = PRIME
    while math.gcd(a, span) != 1:
        a += 2
    return f"({lo} + pmod(cast({idx} as decimal(38,0)) * {a} + 12345, {span}))"


def format_sql(idx: str, alphabets: list[str], salt: int, b=None, name: str = "") -> tuple[str, int]:
    """Format-preserving strings from an index: per position, a character from that position's alphabet.

    Variable positions form a mixed-radix number of a scrambled index, which is bijective while the index stays below
    the capacity returned; positions beyond that are hashed. Positions with a single character are literal. With a
    TableBuild ``b``, the scrambled index is computed once in a helper column (keeps generated code small).
    """
    var = [(p, a) for p, a in enumerate(alphabets) if len(a) > 1]
    cap, weights = 1, {}
    for p, a in var:
        if cap * len(a) >= 2 ** 62:
            break
        weights[p] = cap
        cap *= len(a)
    mult = PRIME
    while math.gcd(mult, cap) != 1:
        mult += 2
    k = f"cast(pmod(cast({idx} as decimal(38,0)) * {mult} + {salt % max(cap, 1)}, {max(cap, 1)}) as bigint)"
    if b is not None:
        k = b.expr_helper(f"__{name}__k", k)
    parts = []
    for p, alpha in enumerate(alphabets):
        if not alpha:
            continue
        if len(alpha) == 1:
            parts.append(sql_str(alpha))
        elif p in weights:
            parts.append(f"substr({sql_str(alpha)}, 1 + cast(pmod(floor({k} / {weights[p]}), {len(alpha)}) as int), 1)")
        else:
            parts.append(f"substr({sql_str(alpha)}, 1 + cast(pmod(xxhash64({idx}, {salt}, {p}), {len(alpha)}) as int), 1)")
    return ("concat(" + ", ".join(parts) + ")") if parts else "''", cap


def widen(alphabets: list[str], needed: int) -> list[str]:
    """Free up constant digit/letter positions, right to left, until the format can hold `needed` values.

    'VX0000123'-style keys have constant leading zeros in the source; growing the key space needs them to vary
    (VX0001234, ...), like a real sequence would. Letters that are constant (prefixes like 'VX') are kept while digits
    suffice.
    """
    out = list(alphabets)

    def capacity():
        c = 1
        for a in out:
            c *= max(len(a), 1)
        return c

    for kind in ("digit", "letter"):
        for p in range(len(out) - 1, -1, -1):
            if capacity() >= needed:
                return out
            a = out[p]
            if len(a) == 1 and kind == "digit" and a.isdigit():
                out[p] = "0123456789"
            elif len(a) == 1 and kind == "letter" and a.isalpha():
                out[p] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if a.isupper() else "abcdefghijklmnopqrstuvwxyz"
    return out


def alias_table(weights: list[float]) -> tuple[list[float], list[int]]:
    """Walker/Vose alias table: pick slot i uniformly, keep it with probability prob[i], else take alias[i]."""
    n = len(weights)
    total = sum(weights) or 1.0
    scaled = [w * n / total for w in weights]
    prob, alias = [1.0] * n, list(range(n))
    small = [i for i, x in enumerate(scaled) if x < 1.0]
    large = [i for i, x in enumerate(scaled) if x >= 1.0]
    while small and large:
        s_i, l_i = small.pop(), large.pop()
        prob[s_i], alias[s_i] = scaled[s_i], l_i
        scaled[l_i] -= 1.0 - scaled[s_i]
        (small if scaled[l_i] < 1.0 else large).append(l_i)
    return prob, alias


def alphabets_from_values(values: list[str]) -> list[str]:
    longest = max((len(v) for v in values), default=0)
    return ["".join(sorted({v[p] for v in values if len(v) > p})) for p in range(min(longest, 48))]


# ------------------------------------------------------------------------------------------------ calibration

CANDIDATES = (
    [("Uniform", Uniform()), ("Normal()", Normal())]
    + [(f"Zipf({e})", Zipf(exponent=e)) for e in (1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0)]
    + [(f"Exponential({r})", Exponential(rate=r)) for r in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)]
    + [(f"LogNormal(0,{s})", LogNormal(mean=0.0, stddev=s)) for s in (0.5, 1.0, 2.0)]
)
_CAL_CACHE: dict = {}


def calibrate_index(spark, target: dict | None, size: int, draws: int, seed: int):
    """Distribution over [0, size) whose rows-per-value shape best matches ``target`` (a fanout dict).

    Every candidate is generated as a small table at the target proportions (capped) in one plan.
    """
    if not target or size <= 1 or draws <= 1:
        return "Uniform", Uniform(), {}
    s = min(size, CAL_MAX_VALUES)
    n = max(2, min(int(round(s * draws / size)), CAL_MAX_ROWS))
    key = (round(target["top10pct"], 3), round(target["gini"], 3), round(target.get("coverage", 1), 3), s, n)
    if key in _CAL_CACHE:
        return _CAL_CACHE[key]
    tables = [TableSpec(name=f"cal_{i}", rows=n, columns=[
        ColumnSpec(name="k", dtype=DataType.LONG, gen=RangeColumn(min=0, max=s - 1, distribution=dist))])
        for i, (_, dist) in enumerate(CANDIDATES)]
    dfs = generate(spark, DataGenPlan(seed=seed, tables=tables))
    union = None
    for i in range(len(CANDIDATES)):
        part = dfs[f"cal_{i}"].groupBy("k").count().select(F.lit(i).alias("c"), F.col("count").alias("n"))
        union = part if union is None else union.unionByName(part)
    by_c = {}
    for r in union.collect():
        by_c.setdefault(r["c"], []).append(r["n"])
    scored = []
    for i, (label, dist) in enumerate(CANDIDATES):
        m = _fanout_local(by_c.get(i, []), s)
        score = (abs(m["top10pct"] - target["top10pct"]) + abs(m["gini"] - target["gini"])
                 + abs(m["coverage"] - target.get("coverage", 1.0))
                 + 0.1 * abs(m["top1"] - target["top1"]) / max(target["top1"], 1 / s))
        scored.append((score, label, dist, m))
    scored.sort(key=lambda x: x[0])
    out = (scored[0][1], scored[0][2], {"score": round(scored[0][0], 4), "candidate": scored[0][3]})
    _CAL_CACHE[key] = out
    return out


def _fanout_local(counts: list[int], n_values: int) -> dict:
    """Same metrics as profiler.fanout, on collected per-value counts."""
    total, used = sum(counts), len(counts)
    if not total:
        return {"top1": 0.0, "top10pct": 0.0, "gini": 0.0, "coverage": 0.0}
    n_values = max(n_values, used)
    srt = sorted(counts)
    k = max(1, -(-n_values // 10))
    zeros = n_values - used
    s = sum((zeros + r + 1) * x for r, x in enumerate(srt))
    return {"top1": srt[-1] / total, "top10pct": sum(srt[-k:]) / total,
            "gini": (2 * s) / (n_values * total) - (n_values + 1) / n_values, "coverage": used / n_values}


# ------------------------------------------------------------------------------------------------ table builder


def _null(cp: ColumnProfile) -> float:
    return 0.0 if cp.null_fraction < NULL_FLOOR else round(min(cp.null_fraction, 1.0), 4)


def _cast_target(type_text: str) -> str:
    t = type_text.lower()
    return "string" if t.startswith(("char", "varchar")) else type_text


class TableBuild:
    """ColumnSpecs for one table, in passes so expressions only reference earlier columns."""

    def __init__(self, name: str, rows: int):
        self.name, self.rows = name, rows
        self.base, self.exprs, self.dependent = [], [], []
        self.helpers, self.provenance = [], {}
        self._have = set()

    def helper(self, spec: ColumnSpec) -> str:
        if spec.name not in self._have:
            self.base.append(spec)
            self.helpers.append(spec.name)
            self._have.add(spec.name)
        return f"`{spec.name}`"

    def expr_helper(self, name: str, sql: str) -> str:
        """An intermediate expression (dropped before writing), referenced by later expressions by name."""
        if name not in self._have:
            self.exprs.append(ColumnSpec(name=name, gen=ExpressionColumn(expr=sql)))
            self.helpers.append(name)
            self._have.add(name)
        return f"`{name}`"

    def row_index(self) -> str:
        return self.helper(ColumnSpec(name="__i", dtype=DataType.LONG, gen=SequenceColumn(start=0, step=1)))

    def uniform(self, name: str, null_fraction: float = 0.0) -> str:
        return self.helper(ColumnSpec(name=name, dtype=DataType.DOUBLE, null_fraction=null_fraction,
                                      gen=RangeColumn(min=0.0, max=1.0)))

    def index(self, name: str, size: int, dist) -> str:
        return self.helper(ColumnSpec(name=name, dtype=DataType.LONG,
                                      gen=RangeColumn(min=0, max=max(size - 1, 0), distribution=dist)))

    def weighted_pick(self, name: str, values: list[str], counts: list, null_fraction: float = 0.0) -> str:
        """SQL picking one of ``values`` (as strings) with probability proportional to ``counts``.

        Short lists use core's WeightedValues. Long lists (day histograms, full key lists) use Walker's alias method:
        two uniform draws and two array lookups per row, exact weights, and no CASE branch per value (core's
        WeightedValues emits one per value, which breaks Spark code generation for long lists).
        """
        if len(values) <= LONG_LIST:
            return self.helper(ColumnSpec(name=name, dtype=DataType.STRING, null_fraction=null_fraction,
                                          gen=ValuesColumn(values=list(values), distribution=WeightedValues(
                                              weights={v: float(c) for v, c in zip(values, counts)}))))
        prob, alias = alias_table([float(c) for c in counts])
        n = len(values)
        u1 = self.uniform(f"{name}__u1", null_fraction)
        u2 = self.uniform(f"{name}__u2")
        probs = f"cast(split({sql_str(','.join(f'{p:.9g}' for p in prob))}, ',') as array<double>)"
        aliases = f"cast(split({sql_str(','.join(map(str, alias)))}, ',') as array<int>)"
        arr = f"split({sql_str(chr(1).join(values))}, {sql_str(chr(1))})"
        slot = f"least(cast(floor({u1} * {n}) as int), {n - 1})"
        pick = f"CASE WHEN {u2} < element_at({probs}, {slot} + 1) THEN {slot} ELSE element_at({aliases}, {slot} + 1) END"
        return self.expr_helper(f"{name}__pick", f"(CASE WHEN {u1} IS NULL THEN NULL ELSE "
                                                 f"element_at({arr}, ({pick}) + 1) END)")

    def literal_bucket(self, name: str, weights: dict, dtype=DataType.LONG) -> str:
        values = [int(k) for k in weights] if dtype == DataType.LONG else list(weights)
        return self.helper(ColumnSpec(name=name, dtype=dtype, gen=ValuesColumn(
            values=values, distribution=WeightedValues(weights={str(k): w for k, w in weights.items()}))))

    def columns(self):
        return self.base + self.exprs + self.dependent


# ------------------------------------------------------------------------------------------------ derive


def derive_config(spark, profiles: dict[str, TableProfile], rel: Relationships, facts, settings: Settings,
                  calibrate: bool = True, log=None) -> dict:
    say = log or (lambda *_: None)
    warnings = list(rel.warnings)
    target = {t: settings.target_short(t) for t in profiles}

    # ---- tables copied unchanged: small code lookups (keyed by a code such as category_cd, or with no key),
    # provided every table they reference is copied too. Entity tables (customers, merchants) are generated.
    pk_cols = {t: [c.lower() for c in v["columns"]] for t, v in rel.primary_keys.items()}
    if settings.copy_tables is not None:
        copied = {t for t in profiles if t in {x.lower() for x in settings.copy_tables}}
    else:
        copied = {t for t, p in profiles.items() if p.rows <= settings.copy_max_rows
                  and (not pk_cols.get(t) or all(CODE_KEY.search(c) for c in pk_cols[t]))}
    changed = True
    while changed:
        changed = False
        for lk in rel.links:
            if lk["kind"] != "overlap" and lk["child_table"] in copied and lk["parent_table"] not in copied:
                copied.discard(lk["child_table"])
                changed = True
    rows = {t: p.rows if t in copied else int(settings.row_overrides.get(t, max(1, round(p.rows * settings.scale))))
            for t, p in profiles.items()}
    builds = {t: TableBuild(target[t], rows[t]) for t in profiles if t not in copied}
    handled: set = set()  # (table, column) generated by a key domain

    tiles = max(1, round(settings.scale)) if settings.date_mode == "extend" else 1
    # dimensions: tables other tables reference (a domain's anchor or a unique parent). Everything else generated is
    # an event / fact table whose dates are extended in date_mode=extend.
    referenced = {d["anchor"]["table"] for d in rel.domains if any(m["table"] != d["anchor"]["table"]
                                                                   for m in d["members"])}
    referenced |= {lk["parent_table"] for lk in rel.links if lk["kind"] != "overlap"}
    event_tables = set(profiles) - copied - referenced

    domain_plans, spread = [], {}  # spread: (table, column) -> (values, rows) for anchors spread over rows
    for d in rel.domains:
        plan = _plan_domain(spark, d, profiles, settings, rows, copied, builds, handled, calibrate, warnings, say,
                            spread)
        if plan:
            domain_plans.append(plan)

    derived = {(x["table"], x["column"].lower()): x for x in rel.derived_columns}
    fds = {(x["table"], x["dependent"].lower()): x for x in rel.functional_dependencies}
    # composite keys (account_id + txn_source): rows sharing a key get different second-column values
    composite = {}
    for ck in rel.composite_keys:
        info = spread.get((ck["table"], ck["columns"][0]))
        if info and ck["table"] in builds:
            composite[(ck["table"], ck["columns"][1].lower())] = info

    for t, prof in profiles.items():
        if t in copied:
            continue
        b = builds[t]
        say(f"deriving {t}")
        temporal = [c for c in prof.columns if c.kind in ("date", "timestamp") and c.null_fraction < 1
                    and (t, c.name.lower()) not in derived and (t, c.name.lower()) not in handled]
        shift_days = 0
        if tiles > 1 and t in event_tables and temporal:
            spans = [(c.quantiles[-1][1] - c.quantiles[0][1]) / (1 if c.kind == "date" else 86400)
                     for c in temporal if c.quantiles]
            if spans:
                shift_days = max(7, round(max(spans) / 7) * 7)  # whole weeks keep the weekday mix
                b.helper(ColumnSpec(name="__k", dtype=DataType.INT, gen=ValuesColumn(values=list(range(tiles)))))

        for cp in prof.columns:
            key = (t, cp.name.lower())
            if key in handled:
                continue
            if key in composite and cp.top_values:
                _rule_composite(b, cp, composite[key])
            elif key in fds:
                _rule_fd(b, cp, fds[key])
            elif key in derived:
                _rule_derived(b, cp, derived[key], prof)
            elif cp.null_fraction >= 1 or cp.family == "other":
                # complex types (STRUCT/ARRAY/MAP) as a bare NULL: generate casts every column to its DDL type,
                # and the plan validator would read struct field names as column references
                null_expr = "null" if cp.family == "other" else f"cast(null as {_cast_target(cp.data_type)})"
                b.exprs.append(ColumnSpec(name=cp.name, gen=ExpressionColumn(expr=null_expr)))
                b.provenance[cp.name] = {"rule": "always NULL", "details": "all-null in source or unsupported type"}
                if cp.family == "other":
                    warnings.append(f"{t}.{cp.name}: type {cp.data_type} is not generated (left NULL)")
            elif cp.kind in ("date", "timestamp", "time"):
                _rule_temporal(b, cp, shift_days if cp in temporal else 0, tiles)
            elif _is_categorical(cp, settings.top_n):
                _rule_values(b, cp, warnings, t)
            elif cp.quantiles and (_fractional(cp) or (cp.kind in ("integer", "decimal", "number")
                                                         and cp.unique_ratio < 0.95 and not cp.id_like)):
                _rule_icdf(b, cp)
            else:
                unique = cp.unique_ratio >= 0.95 or pk_cols.get(t) == [cp.name.lower()]
                _rule_self_domain(spark, b, cp, prof, settings, rows[t], unique, calibrate, t)

    plan = DataGenPlan(seed=settings.seed, tables=[TableSpec(name=b.name, rows=b.rows, columns=b.columns())
                                                   for b in builds.values()])
    resolve_plan(plan)  # fail here, not at generation time, if a derived rule is invalid

    post = []
    for t, prof in profiles.items():
        fks = [{"column": lk["child_column"], "parent": settings.target_fqn(lk["parent_table"]),
                "parent_column": lk["parent_column"]}
               for lk in rel.links if lk["child_table"] == t and lk["kind"] == "fk"
               and pk_cols.get(lk["parent_table"]) == [lk["parent_column"]]]
        post.append({
            "name": target[t], "source": prof.name, "target": settings.target_fqn(t), "rows": rows[t],
            "source_rows": prof.rows, "copy": t in copied, "comment": prof.comment,
            "columns": [{"name": c.name, "type": c.data_type, "nullable": c.nullable, "comment": c.comment}
                        for c in prof.columns],
            "partition_columns": prof.partition_columns, "clustering_columns": prof.clustering_columns,
            "drop": builds[t].helpers if t in builds else [],
            "primary_key": pk_cols.get(t, []), "foreign_keys": fks,
        })
    order, placed = [], set()
    while len(order) < len(post):
        progressed = False
        for pt in post:
            if pt["name"] in placed:
                continue
            if all(fk["parent"] == pt["target"] or any(o["target"] == fk["parent"] for o in order)
                   for fk in pt["foreign_keys"]):
                order.append(pt)
                placed.add(pt["name"])
                progressed = True
        if not progressed:
            order += [pt for pt in post if pt["name"] not in placed]
            break

    for c in rel.correlations:
        warnings.append(f"{c['table']}: {c['columns'][0]} <-> {c['columns'][1]} {c['measure']}={c['value']} in the "
                        f"source; generated independently")
    for p in rel.cross_predicates:
        warnings.append(f"query {p['query']} compares {p['left_table']}.{p['left_column']} {p['op']} "
                        f"{p['right_table']}.{p['right_column']}; both are generated independently")
    provenance = {target[t]: b.provenance for t, b in builds.items()}
    for t in copied:
        provenance[target[t]] = {"*": {"rule": "copied unchanged",
                                       "details": f"{profiles[t].rows:,} rows (copy_max_rows {settings.copy_max_rows})"}}
    return {
        "version": 2,
        "settings": asdict(settings),
        "plan": plan.model_dump(mode="json", exclude_none=True),
        "post": {"tables": order},
        "domains": domain_plans,
        "provenance": provenance,
        "relationships": rel.to_dict(),
        "queries": facts.to_dict() if hasattr(facts, "to_dict") else facts,
        "warnings": warnings,
    }


# ------------------------------------------------------------------------------------------------ column rules


def _fractional(cp: ColumnProfile) -> bool:
    """Amounts and measures (decimals with a scale, floats, numeric text with decimals) are never IDs."""
    if cp.kind == "float":
        return True
    if cp.kind == "decimal":
        return not cp.data_type.replace(" ", "").endswith(",0)")
    return bool(cp.encoded and cp.encoded["kind"] == "number" and cp.encoded.get("scale"))


def _is_categorical(cp: ColumnProfile, top_n: int) -> bool:
    if not cp.top_values:
        return False
    if cp.family == "boolean":
        return True
    complete = (cp.top_coverage or 0) >= 0.999
    if cp.id_like and not complete:
        return False
    if cp.family == "string" and not cp.encoded:
        return complete or (cp.unique_ratio <= 0.5 and cp.distinct <= top_n)
    return complete and cp.distinct <= min(top_n, 50)


def _rule_values(b: TableBuild, cp: ColumnProfile, warnings, t):
    values = [v for v, _ in cp.top_values]
    nf = _null(cp)
    if cp.family == "boolean":
        bools = [str(v).lower() == "true" for v in values]
        spec = ColumnSpec(name=cp.name, dtype=DataType.BOOLEAN, null_fraction=nf, nullable=nf > 0,
                          gen=ValuesColumn(values=bools, distribution=WeightedValues(
                              weights={str(bv): n for bv, (_, n) in zip(bools, cp.top_values)})))
        b.base.append(spec)
    else:
        pick = b.weighted_pick(f"__{cp.name}__v", [str(v) for v in values], [n for _, n in cp.top_values], nf)
        b.exprs.append(ColumnSpec(name=cp.name, gen=ExpressionColumn(
            expr=f"cast({pick} as {_cast_target(cp.data_type)})")))
    cov = cp.top_coverage or 1.0
    b.provenance[cp.name] = {"rule": "categorical: weighted values",
                             "details": f"{len(values)} of {cp.distinct} distinct values, covering {cov:.1%} of rows"}
    if cov < 0.95:
        warnings.append(f"{t}.{cp.name}: top {len(values)} values cover only {cov:.1%} of source rows")


def _rule_composite(b: TableBuild, cp: ColumnProfile, info: tuple):
    """Second column of a composite key: each row of a key group gets the next value (most frequent first)."""
    size, n = info
    values = [v for v, _ in cp.top_values]
    i = b.row_index()
    group = f"floor({i} * {size} / {n})"
    pos = f"({i} - ceil({group} * {n} / {size}))"
    arr = "array(" + ", ".join(sql_str(v) for v in values) + ")"
    expr = f"cast(element_at({arr}, cast(pmod({pos}, {len(values)}) as int) + 1) as {_cast_target(cp.data_type)})"
    b.exprs.append(ColumnSpec(name=cp.name, gen=ExpressionColumn(expr=expr)))
    b.provenance[cp.name] = {"rule": "second column of a composite key",
                             "details": f"rows sharing a key take {', '.join(values[:5])}... in turn, so the pair "
                                        f"stays unique"}


def _rule_fd(b: TableBuild, cp: ColumnProfile, fd: dict):
    whens = " ".join(f"WHEN {sql_str(a)} THEN {sql_str(v)}" for a, v in fd["mapping"])
    expr = f"cast(CASE cast(`{fd['determinant']}` as string) {whens} END as {_cast_target(cp.data_type)})"
    b.dependent.append(ColumnSpec(name=cp.name, gen=ExpressionColumn(expr=expr)))
    b.provenance[cp.name] = {"rule": f"derived from {fd['determinant']} (functional dependency)",
                             "details": f"{len(fd['mapping'])} value pairs from the source"}


def _rule_derived(b: TableBuild, cp: ColumnProfile, d: dict, prof: TableProfile):
    src = prof.column(d["source"])
    expr = d["rule"].replace("{day}", f"to_date({day_sql(src.name, src)})")
    b.dependent.append(ColumnSpec(name=cp.name, gen=ExpressionColumn(expr=f"cast({expr} as {_cast_target(cp.data_type)})")))
    b.provenance[cp.name] = {"rule": f"derived from {src.name}",
                             "details": d["rule"].replace("{day}", f"day({src.name})")}


def _format_temporal(cp: ColumnProfile, value: str, kind: str) -> str:
    """Turn a DATE / TIMESTAMP / seconds-of-day expression into the column's type (or its string format)."""
    enc = cp.encoded
    if kind == "time":
        h, m, s = (f"lpad(cast({x} as string), 2, '0')" for x in
                   (f"floor({value} / 3600)", f"floor(pmod({value}, 3600) / 60)", f"pmod({value}, 60)"))
        sep = "':'" if enc and ":" in enc["format"] else "''"
        return f"concat({h}, {sep}, {m}, {sep}, {s})"
    if enc:
        return f"date_format({value}, '{enc['format']}')"
    return value


def _rule_temporal(b: TableBuild, cp: ColumnProfile, shift_days: int, tiles: int):
    nf, kind = _null(cp), cp.kind
    u = b.uniform(f"__{cp.name}__u", nf)
    if kind == "time":
        expr = f"cast(round({icdf_sql(u, cp.quantiles)}) as bigint)"
        detail = f"seconds-of-day distribution ({len(cp.quantiles)} knots)"
    elif cp.day_weights:
        days = b.weighted_pick(f"__{cp.name}__day", [d for d, _ in cp.day_weights], [n for _, n in cp.day_weights],
                               nf)
        day = f"to_date({days})"
        if shift_days:
            day = f"date_add({day}, `__k` * {shift_days})"
        if kind == "date":
            expr = day
        else:
            tod = icdf_sql(u, cp.tod_quantiles) if cp.tod_quantiles else f"({u} * 86399)"
            expr = f"timestamp_seconds(unix_seconds(cast({day} as timestamp)) + cast(round({tod}) as bigint))"
        detail = f"exact per-day weights over {len(cp.day_weights)} days" + \
                 (", time of day from the source's intraday distribution" if kind == "timestamp" else "")
    else:
        x = icdf_sql(u, cp.quantiles)
        if kind == "date":
            off = f" + `__k` * {shift_days}" if shift_days else ""
            expr = f"date_from_unix_date(cast(round({x}{off}) as int))"
        else:
            off = f" + `__k` * {shift_days * 86400}" if shift_days else ""
            expr = f"timestamp_seconds(cast(round({x}{off}) as bigint))"
        detail = f"empirical distribution ({len(cp.quantiles)} quantile knots)"
    if shift_days:
        detail += f"; tiled {tiles}x with a {shift_days}-day shift (date_mode=extend)"
    if cp.encoded and cp.encoded.get("format"):
        detail += f"; stored as text '{cp.encoded['format']}'"
    expr = _format_temporal(cp, expr, kind)
    b.exprs.append(ColumnSpec(name=cp.name, gen=ExpressionColumn(expr=f"cast({expr} as {_cast_target(cp.data_type)})")))
    b.provenance[cp.name] = {"rule": f"{kind}: " + ("per-day weights" if cp.day_weights and kind != "time"
                                                     else "empirical distribution"), "details": detail}


def _rule_icdf(b: TableBuild, cp: ColumnProfile):
    knots = [list(k) for k in cp.quantiles]
    if cp.min is not None and not cp.encoded:
        knots[0][1], knots[-1][1] = float(cp.min), float(cp.max)
    u = b.uniform(f"__{cp.name}__u", _null(cp))
    x = icdf_sql(u, knots)
    if cp.encoded:  # number stored as text: same number of decimals as the source
        expr = f"format_string('%.{cp.encoded.get('scale', 0)}f', {x})"
    elif cp.family == "integer":
        expr = f"cast(round({x}) as {cp.data_type})"
    elif cp.family == "decimal":
        scale = int(cp.data_type.split(",")[1].rstrip(")")) if "," in cp.data_type else 0
        expr = f"cast(round({x}, {scale}) as {cp.data_type})"
    else:
        expr = f"cast({x} as {cp.data_type})"
    b.exprs.append(ColumnSpec(name=cp.name, gen=ExpressionColumn(expr=f"cast({expr} as {_cast_target(cp.data_type)})")))
    b.provenance[cp.name] = {"rule": "number: empirical distribution (inverse CDF)",
                             "details": f"{len(knots)} quantile knots, min {knots[0][1]:g}, max {knots[-1][1]:g}"
                                        + (" (stored as text)" if cp.encoded else "")}


# ------------------------------------------------------------------------------------------------ value functions


class ValueFn:
    """A key domain's value function: domain index (SQL) -> value (SQL)."""

    def __init__(self, kind: str, sql, detail: str, literal_index: dict | None = None, capacity: int | None = None):
        self.kind, self.sql, self.detail = kind, sql, detail
        self.literal_index = literal_index or {}  # literal (as string) -> domain index
        self.capacity = capacity


def _int_like(cp: ColumnProfile) -> bool:
    return cp.family == "integer" or (cp.family == "decimal" and cp.data_type.replace(" ", "").endswith(",0)")) or \
        bool(cp.encoded and cp.encoded["kind"] == "number" and not cp.encoded.get("scale"))


def _num(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def value_fn(cp: ColumnProfile, literals: list, salt: int, key_like: bool, needed: int,
             type_max: int | None = None) -> ValueFn:
    """Pick a domain's value function from its anchor column.

    ``type_max`` is the largest value every column of the domain can hold (SMALLINT, DECIMAL(6,0), ...): integer
    keys are spaced to fit it, and wrap within it (with a warning) when even step 1 can't hold the values needed.
    """
    if _int_like(cp):
        lo = _num(cp.encoded.get("min") if cp.encoded else cp.min) or 0
        hi = _num(cp.encoded.get("max") if cp.encoded else cp.max)
        hi = lo + needed if hi is None else hi
        tmax = type_max if type_max is not None else (int_type_max(cp.data_type) or 2 ** 63 - 1)
        room = max(tmax - lo + 1, 1)
        lits = sorted({x for x in (_num(v) for v in literals) if x is not None})
        if key_like:
            step = max(1, (hi - lo) // max(cp.distinct - 1, 1))
            step = max(1, min(step, room // max(needed, 1)))
            wrap = room if needed * step > room else None
            note = f"; wraps within the column type (max {tmax:,}), so some values repeat" if wrap else ""
            return ValueFn("int_lattice", lambda i, b=None, name="": lattice_sql(i, lo, step, lits, wrap),
                           f"source key range kept from {lo} (step {step}) and extended above the source max"
                           + (f"; {len(lits)} filtered value(s) at fixed positions" if lits else "") + note,
                           {str(v): k for k, v in enumerate(lits)}, capacity=room // step if wrap else None)
        span = min(max(hi - lo + 1, needed), room)
        return ValueFn("int_scrambled", lambda i, b=None, name="": affine_sql(i, lo, span),
                       f"values scrambled within [{lo}, {lo + span - 1}] (range and distinct count kept)",
                       capacity=span)
    values = [v for v, _ in (cp.top_values or [])]
    complete = bool(values) and (cp.top_coverage or 0) >= 0.999 and len(values) <= CATEGORICAL_PROFILE_MAX
    alphabets = cp.alphabets or (alphabets_from_values(values) if values else ["ABCDEFGHIJKLMNOPQRSTUVWXYZ"] * 8)
    alphabets = widen(alphabets, needed)
    cap = format_sql("0", alphabets, salt)[1]

    def fmt(i, b=None, name=""):
        return format_sql(i, alphabets, salt, b, name)[0]

    if complete and key_like:
        # one string literal split once (constant-folded) instead of an array of n expressions: small generated code
        arr = f"split({sql_str(chr(1).join(values))}, {sql_str(chr(1))})"
        n = len(values)
        return ValueFn("string_list", lambda i, b=None, name="": f"(CASE WHEN {i} < {n} THEN element_at({arr}, "
                                                                 f"cast({i} as int) + 1) ELSE {fmt(i, b, name)} END)",
                       f"the source's {n} key values, then new values in the same format",
                       {str(v): k for k, v in enumerate(values)}, capacity=cap + n)
    return ValueFn("string_format", fmt, f"format-preserving strings ({len(alphabets)} characters, "
                                         f"{cap:,} possible values)", capacity=cap)


def _literal_weights(cp: ColumnProfile, prof: TableProfile, index_of: dict) -> dict:
    """Bucket weights: -1 = no literal, else the literal's domain index weighted by its source share."""
    rows = max(prof.rows, 1)
    lits = {lit: n for lit, n in (cp.literal_counts or {}).items() if n > 0 and lit in index_of}
    if not lits:
        return {}
    w = {str(index_of[lit]): n / rows for lit, n in lits.items()}
    w["-1"] = max(1e-9, 1 - sum(w.values()))
    return w


# ------------------------------------------------------------------------------------------------ domains


def _plan_domain(spark, d, profiles, settings, rows, copied, builds, handled, calibrate, warnings, say, spread):
    anchor = (d["anchor"]["table"], d["anchor"]["column"])
    members = d["members"]
    if all(m["table"] in copied for m in members):
        return None
    if anchor[0] in copied:
        # a copied lookup fixes the values: members keep their own (complete) value lists, misses included
        out = []
        for m in members:
            if m["table"] in copied:
                continue
            prof = profiles[m["table"]]
            cp = prof.column(m["column"])
            b = builds[m["table"]]
            if cp.top_values and (cp.top_coverage or 0) >= 0.999:
                _rule_values(b, cp, warnings, m["table"])
                b.provenance[cp.name]["rule"] = f"values of copied {anchor[0]}.{anchor[1]} (source weights, misses kept)"
            else:
                _rule_self_domain(spark, b, cp, prof, settings, rows[m["table"]], cp.unique_ratio >= 0.95, calibrate,
                                  m["table"])
                warnings.append(f"{m['table']}.{m['column']}: references copied {anchor[0]}.{anchor[1]} but has more "
                                f"distinct values than are listed; its match rate may differ")
            handled.add((m["table"], m["column"]))
            out.append({"table": m["table"], "column": m["column"], "index": "source values"})
        return {"id": d["id"], "anchor": d["anchor"], "kind": "fixed by copied table", "members": out}

    aprof = profiles[anchor[0]]
    acp = aprof.column(anchor[1])
    a_info = next(m for m in members if (m["table"], m["column"]) == anchor)
    say(f"domain {d['id']}: anchor {anchor[0]}.{anchor[1]}")
    if a_info["unique"]:
        size_t = rows[anchor[0]]
        growth = size_t / max(acp.distinct, 1)
    else:
        growth = settings.scale if settings.grow_ids and acp.id_like else 1.0
        size_t = max(1, round(acp.distinct * growth))
    literals = [v for m in members for v, n in (profiles[m["table"]].column(m["column"]).literal_counts or {}).items()
                if n > 0]
    maxima = [int_type_max(profiles[m["table"]].column(m["column"]).data_type) for m in members]
    maxima = [x for x in maxima if x is not None]
    fn = value_fn(acp, literals, salt=sum(map(ord, d["id"])) * 7919 % 99991, key_like=True,
                  needed=max(size_t, 1) * 4, type_max=min(maxima) if maxima else None)
    row_fn = {}  # unique member -> (row index SQL -> domain index SQL)
    next_free = [size_t]  # private ranges are allocated after the anchor's values

    def private(size):
        start = next_free[0]
        next_free[0] += max(size, 0)
        return start

    ordered = sorted(members, key=lambda m: ((m["table"], m["column"]) != anchor, m["parent"] is not None,
                                             not m["unique"]))
    done, queue = [], list(ordered)
    while queue:  # parents before children
        progressed = False
        for m in list(queue):
            p = m["parent"]
            if p is None or (p["table"], p["column"]) in {(x["table"], x["column"]) for x in done} or \
                    (p["table"], p["column"]) not in {(x["table"], x["column"]) for x in members}:
                done.append(m)
                queue.remove(m)
                progressed = True
        if not progressed:
            done += queue
            break

    infos = []
    for m in done:
        t, c = m["table"], m["column"]
        if t in copied:
            continue
        prof, b = profiles[t], builds[t]
        cp, n_t = prof.column(c), rows[t]
        i = b.row_index()
        nf = _null(cp)
        info = {"table": t, "column": c, "rows": n_t}
        if (t, c) == anchor:
            if a_info["unique"]:
                idx = i
                row_fn[(t, c)] = lambda j: j
                info["index"] = "row number (unique key)"
            else:
                idx = f"floor({i} * {size_t} / {n_t})"
                spread[(t, c)] = (size_t, n_t)
                info["index"] = f"rows spread over {size_t:,} values ({n_t / size_t:.2f} rows each)"
        else:
            parent = m["parent"]
            pfn = row_fn.get((parent["table"], parent["column"])) if parent else None
            # a member that references a unique parent keeps the match rate and overlap of *that* link
            ref = parent if pfn else m
            rate = float(ref.get("match_rate") or 0.0)
            ref_overlap = ref.get("overlap")
            if ref_overlap is None:
                ref_overlap = round((m.get("distinct") or 0) * rate)
            overlap_t = max(0, min(size_t, round((m.get("overlap") or 0) * growth)))
            m_growth = settings.scale if settings.grow_ids and cp.id_like else 1.0
            private_t = max(0, round(max((m.get("distinct") or 0) - ref_overlap, 0) * m_growth))
            if rate >= 0.999:
                private_t = 0
            if m["unique"]:
                if pfn:
                    matched, hit = min(round(rate * n_t), rows[parent["table"]]), pfn
                    source = f"{parent['table']}.{parent['column']}"
                else:
                    matched = min(round(rate * n_t), overlap_t)
                    hit = (lambda j, _o=size_t - overlap_t: f"({_o} + {j})")
                    source = "the anchor"
                start = private(n_t - matched)
                row_fn[(t, c)] = (lambda j, _m=matched, _h=hit, _s=start:
                                  f"(CASE WHEN {j} < {_m} THEN {_h(j)} ELSE {_s} + ({j} - {_m}) END)")
                idx = row_fn[(t, c)](i)
                info["index"] = f"unique: {matched:,} rows match {source}, {n_t - matched:,} new"
            else:
                if pfn:
                    size, tgt, region = rows[parent["table"]], parent.get("fanout"), pfn
                    source = f"{parent['table']}.{parent['column']}"
                else:
                    size, tgt = max(overlap_t, 1), m.get("fanout")
                    region = (lambda j, _o=size_t - max(overlap_t, 1): f"({_o} + {j})")
                    source = "the anchor's values"
                if tgt and tgt.get("mean", 99) <= SPREAD_MEAN:
                    # near-unique: random draws would only cover ~63% of the values, so spread rows evenly
                    matched = round(rate * n_t)
                    hit = region(f"pmod({i} * {coprime(size)}, {size})")
                    if private_t > 0 and matched < n_t:
                        start = private(private_t)
                        miss = f"({start} + pmod(({i} - {matched}) * {coprime(private_t)}, {private_t}))"
                    else:
                        miss = hit
                    idx = f"(CASE WHEN {i} < {matched} THEN {hit} ELSE {miss} END)"
                    if nf > 0:
                        idx = f"(CASE WHEN {b.uniform(f'__{c}__m', nf)} IS NULL THEN NULL ELSE {idx} END)"
                    info["index"] = (f"{rate:.1%} of rows spread evenly over {source} ({tgt['mean']:.2f} rows per "
                                     f"value); the rest over {private_t:,} values that don't match")
                    info["skew"] = "even spread"
                else:
                    label, dist, cal = (calibrate_index(spark, tgt, size, max(1, round(rate * n_t)), settings.seed)
                                        if calibrate else ("Uniform", Uniform(), {}))
                    j = b.index(f"__{c}__j", size, dist)
                    u = b.uniform(f"__{c}__m", nf)
                    if private_t > 0:
                        start = private(private_t)
                        miss = f"({start} + {b.index(f'__{c}__p', private_t, Uniform())})"
                    else:
                        miss = region(j)
                    idx = f"(CASE WHEN {u} IS NULL THEN NULL WHEN {u} < {rate!r} THEN {region(j)} ELSE {miss} END)"
                    info["index"] = (f"{rate:.1%} of rows reference {source} ({label} skew); the rest use "
                                     f"{private_t:,} values that don't match")
                    info["skew"] = label
                    if cal:
                        info["calibration"] = cal
            weights = _literal_weights(cp, prof, fn.literal_index)
            if weights:
                lv = b.literal_bucket(f"__{c}__lit", weights)
                idx = f"(CASE WHEN {lv} >= 0 THEN {lv} ELSE {idx} END)"
                info["literals"] = sorted(lit for lit in (cp.literal_counts or {}) if lit in fn.literal_index)
        idx = b.expr_helper(f"__{c}__idx", idx)
        b.exprs.append(ColumnSpec(name=cp.name, gen=ExpressionColumn(
            expr=f"cast({fn.sql(idx, b, c)} as {_cast_target(cp.data_type)})")))
        b.provenance[cp.name] = {"rule": f"key domain {d['id']} ({fn.kind})",
                                 "details": {"value": fn.detail, "index": info["index"],
                                             "anchor": f"{anchor[0]}.{anchor[1]}",
                                             **{k: info[k] for k in ("literals", "skew") if k in info}}}
        handled.add((t, c))
        infos.append(info)
    if fn.capacity is not None and fn.capacity < next_free[0]:
        warnings.append(f"domain {d['id']}: the value format of {anchor[0]}.{anchor[1]} holds {fn.capacity:,} values "
                        f"but {next_free[0]:,} are needed; some values will repeat")
    return {"id": d["id"], "anchor": d["anchor"], "kind": fn.kind, "value": fn.detail, "anchor_values": size_t,
            "index_space": next_free[0], "members": infos}


def _rule_self_domain(spark, b: TableBuild, cp: ColumnProfile, prof: TableProfile, settings: Settings, n_t: int,
                      unique: bool, calibrate: bool, t: str):
    """A high-cardinality column on its own: the source's distinct count, skew and format, plus literals."""
    i = b.row_index()
    nf = _null(cp)
    growth = settings.scale if settings.grow_ids and (cp.id_like or unique) else 1.0
    size_t = n_t if unique else max(1, round(cp.distinct * growth))
    literals = [v for v, n in (cp.literal_counts or {}).items() if n > 0]
    fn = value_fn(cp, literals, salt=sum(map(ord, f"{t}.{cp.name}")) * 7919 % 99991,
                  key_like=cp.id_like or unique, needed=size_t, type_max=int_type_max(cp.data_type))
    if unique:
        idx, how = i, "row number (unique)"
    elif cp.fanout and cp.fanout.get("mean", 99) <= SPREAD_MEAN:
        idx, how = f"pmod({i} * {coprime(size_t)}, {size_t})", f"{size_t:,} values, rows spread evenly"
    else:
        label, dist, _ = (calibrate_index(spark, cp.fanout, size_t, n_t, settings.seed)
                          if calibrate and cp.fanout else ("Uniform", Uniform(), {}))
        idx, how = b.index(f"__{cp.name}__j", size_t, dist), f"{size_t:,} values, {label} skew"
    if nf > 0:
        u = b.uniform(f"__{cp.name}__m", nf)
        idx = f"(CASE WHEN {u} IS NULL THEN NULL ELSE {idx} END)"
    lit_rows = {lit: n for lit, n in (cp.literal_counts or {}).items() if n > 0}
    weights = _literal_weights(cp, prof, fn.literal_index)
    if weights:
        lv = b.literal_bucket(f"__{cp.name}__lit", weights)
        idx = f"(CASE WHEN {lv} >= 0 THEN {lv} ELSE {idx} END)"
    idx = b.expr_helper(f"__{cp.name}__idx", idx)
    if weights:
        value = fn.sql(idx, b, cp.name)
    elif lit_rows:
        # format-generated values can't place literals by index: emit the literal itself with its source share
        w = {lit: n / max(prof.rows, 1) for lit, n in lit_rows.items()}
        w[""] = max(1e-9, 1 - sum(w.values()))
        lv = b.literal_bucket(f"__{cp.name}__lit", w, dtype=DataType.STRING)
        value = f"(CASE WHEN {lv} != '' THEN {lv} ELSE {fn.sql(idx, b, cp.name)} END)"
    else:
        value = fn.sql(idx, b, cp.name)
    b.exprs.append(ColumnSpec(name=cp.name, gen=ExpressionColumn(expr=f"cast({value} as {_cast_target(cp.data_type)})")))
    b.provenance[cp.name] = {"rule": f"high-cardinality: {fn.kind}",
                             "details": {"value": fn.detail, "index": how,
                                         **({"literals": sorted(lit_rows)} if lit_rows else {})}}
    if fn.capacity is not None and fn.capacity < size_t:
        b.provenance[cp.name]["details"]["note"] = f"format holds {fn.capacity:,} values for {size_t:,} needed"
