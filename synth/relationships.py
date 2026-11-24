"""Primary keys, links between tables (foreign keys, partial foreign keys, overlaps) and within-table dependencies.

Every candidate link - a Unity Catalog FK constraint, an equality join in the SQL, or a column named like another
table's key - is measured on the data:

* the share of each side's rows whose value exists on the other side
* distinct values on each side and in common
* whether each side is unique

and classified as ``fk`` (child values all exist in a unique parent), ``partial_fk`` (unique parent, some child values
missing) or ``overlap`` (neither side unique: a many-to-many join whose match rate matters).

Linked columns are grouped into **key domains** (connected components). Each domain has an anchor - preferably a
unique key - and every member's overlap with the anchor is measured directly, so the generator can give all members
one shared value space with the same match rates.

On large tables keys are measured on a key-hash sample: every row of a sampled key value is kept, with the same
hash rule on both sides of a link, so match rates and rows-per-key stay exact for the sampled keys; counts are scaled
back up. Measures of association (correlations) use the profile's random row sample.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field

from pyspark.sql import functions as F

from .profiler import TableProfile, day_sql, fanout, in_key_sample, key_expr, key_frame

FD_SAMPLE_ROWS = 200_000
FD_MAX_DETERMINANTS = 30
FD_MAX_PAIRS = 600
PK_SHORTLIST_RATIO = 0.90  # distinct/rows (approximate, +-5%) for a column to get the exact uniqueness check
MAX_NAME_LINKS = 30  # candidate links found only by column name (not declared, not joined in the SQL) to measure
KEY_NAME = re.compile(r"(^id$|_id$|_key$|_no$|_num$|_nbr$|_number$|^key$)")
CODE_NAME = re.compile(r"(_cd$|_code$|_id$|_key$|_num$|_no$|_type$)")


@dataclass
class Relationships:
    primary_keys: dict = field(default_factory=dict)  # table -> {columns, evidence}
    composite_keys: list = field(default_factory=list)  # [{table, columns, note}]
    links: list = field(default_factory=list)  # measured links (see module doc)
    domains: list = field(default_factory=list)  # [{id, anchor, members}]
    functional_dependencies: list = field(default_factory=list)
    derived_columns: list = field(default_factory=list)
    correlations: list = field(default_factory=list)
    cross_predicates: list = field(default_factory=list)
    self_joins: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def domain_of(self, table: str, column: str) -> dict | None:
        for d in self.domains:
            for m in d["members"]:
                if m["table"] == table and m["column"] == column.lower():
                    return d
        return None


def read_declared_constraints(fqns: list[str]) -> tuple[dict, list[str]]:
    """PK/FK constraints from Unity Catalog via the Databricks SDK (works in notebooks and with Connect)."""
    out, errors = {}, []
    try:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
    except Exception as e:  # noqa: BLE001
        return {}, [f"constraints not read (no Databricks SDK client): {e}"]
    for fqn in fqns:
        try:
            info = w.tables.get(fqn)
            out[fqn] = [c.as_dict() for c in (info.table_constraints or [])]
        except Exception as e:  # noqa: BLE001
            errors.append(f"constraints not read for {fqn}: {e}")
    return out, errors


def _entity(short_name: str) -> str:
    """'customer_dim_customer' -> 'customer': the noun a child key column would be named after."""
    name = short_name.lower()
    m = re.search(r"(?:^|_)(?:dim|fact|fct|ref|lkp|d|f)_(.+)$", name)
    noun = m.group(1) if m else name.split("_")[-1]
    return noun[:-1] if noun.endswith("s") and not noun.endswith("ss") else noun


_key_expr = key_expr  # kept for callers of the old name


def _row_sample(spark, prof: TableProfile):
    """The table, or the same random row sample its profile was made on."""
    df = spark.table(prof.name)
    return df.sample(fraction=prof.row_fraction, seed=17) if prof.row_fraction < 1 else df


def measure_link(spark, pa: TableProfile, ca, pb: TableProfile, cb) -> dict:
    """Match statistics between two columns (value-level join of their distinct values with row counts).

    Large tables are measured on a key-hash sample shared by both sides (the smaller of the two profiles' key
    fractions): ratios and rows-per-key are measured on it, row and distinct counts are scaled back up.
    """
    f = min(pa.key_fraction, pb.key_fraction)
    a = key_frame(spark.table(pa.name), ca, f).groupBy("k").agg(F.count(F.lit(1)).alias("na"))
    b = key_frame(spark.table(pb.name), cb, f).groupBy("k").agg(F.count(F.lit(1)).alias("nb"))
    j = a.join(b, "k", "full_outer")
    r = j.agg(
        F.sum("na").alias("rows_a"), F.sum("nb").alias("rows_b"),
        F.count("na").alias("distinct_a"), F.count("nb").alias("distinct_b"),
        F.sum(F.when(F.col("nb").isNotNull(), F.col("na"))).alias("rows_a_matched"),
        F.sum(F.when(F.col("na").isNotNull(), F.col("nb"))).alias("rows_b_matched"),
        F.sum(F.when(F.col("na").isNotNull() & F.col("nb").isNotNull(), 1).otherwise(0)).alias("distinct_both"),
        F.max("na").alias("max_a"), F.max("nb").alias("max_b"),
    ).first().asDict()
    r = {k: int(v or 0) for k, v in r.items()}
    r["a_in_b"] = round(r["rows_a_matched"] / r["rows_a"], 6) if r["rows_a"] else 0.0
    r["b_in_a"] = round(r["rows_b_matched"] / r["rows_b"], 6) if r["rows_b"] else 0.0
    # unique: no key repeats, and no NULLs (null rates are exact - they come from the full table)
    r["a_unique"] = r["max_a"] == 1 and (r["rows_a"] == pa.rows if f >= 1 else ca.null_fraction == 0)
    r["b_unique"] = r["max_b"] == 1 and (r["rows_b"] == pb.rows if f >= 1 else cb.null_fraction == 0)
    r["fanout_a"] = fanout(a.select(F.col("na").alias("n")), r["distinct_a"]) if r["distinct_a"] else None
    r["fanout_b"] = fanout(b.select(F.col("nb").alias("n")), r["distinct_b"]) if r["distinct_b"] else None
    if f < 1:
        for k in ("rows_a", "rows_b", "distinct_a", "distinct_b", "rows_a_matched", "rows_b_matched", "distinct_both"):
            r[k] = int(round(r[k] / f))
        r["key_fraction"] = f
    return r


def _is_unique_exact(spark, prof: TableProfile, col: str) -> bool:
    c = prof.column(col)
    if c.null_fraction > 0 or c.unique_ratio < PK_SHORTLIST_RATIO:
        return False
    df = spark.table(prof.name)
    if prof.key_fraction >= 1:
        return df.select(f"`{col}`").distinct().count() == prof.rows
    # every copy of a sampled key is in the sample, so a duplicate shows up as a key with two rows there
    keys = key_frame(df, c, prof.key_fraction)
    return keys.groupBy("k").count().where("count > 1").limit(1).count() == 0


def identify(spark, profiles: dict[str, TableProfile], facts, declared: dict | None = None,
             inclusion_threshold: float = 0.99, top_n: int = 100, check_correlations: bool = True,
             log=None) -> Relationships:
    """Merge every kind of evidence into keys, measured links, key domains and within-table dependencies."""
    rel = Relationships(cross_predicates=list(facts.cross_predicates), self_joins=list(facts.self_joins))
    by_fqn = {p.name.lower(): k for k, p in profiles.items()}
    declared = {k.lower(): v for k, v in (declared or {}).items()}

    def short(name: str) -> str | None:
        key = name.lower()
        return by_fqn.get(key) or (key if key in profiles else by_fqn.get(key.split(".")[-1]))

    # ---------- primary keys: declared, else unique + not null (checked exactly)
    candidates = {}  # (child_t, child_c, parent_t, parent_c) -> evidence
    for t, prof in profiles.items():
        for con in declared.get(prof.name.lower(), []):
            if "primary_key_constraint" in con:
                rel.primary_keys[t] = {"columns": [c.lower() for c in con["primary_key_constraint"]["child_columns"]],
                                       "evidence": ["constraint"]}
            if "foreign_key_constraint" in con:
                fk = con["foreign_key_constraint"]
                parent = short(fk["parent_table"])
                if parent is None:
                    rel.warnings.append(f"{t}: FK {fk.get('name')} points outside the profiled tables; ignored")
                    continue
                for cc, pc in zip(fk["child_columns"], fk["parent_columns"]):
                    candidates.setdefault((t, cc.lower(), parent, pc.lower()), set()).add("constraint")
    joined = {(j["left_table"], j["left_column"]) for j in facts.joins} | \
             {(j["right_table"], j["right_column"]) for j in facts.joins}
    for t, prof in profiles.items():
        if t in rel.primary_keys:
            pk = rel.primary_keys[t]["columns"]
            if len(pk) == 1 and not _is_unique_exact(spark, prof, pk[0]):
                rel.warnings.append(f"{t}: declared PK {pk} is not unique and non-null in the data")
            continue
        ent = _entity(t)
        # a key is joined by the SQL, or named like the table's own key ('id', '<entity>_id', ...)
        own = {"id", "key", f"{ent}_id", f"{ent}id", f"{ent}_key", f"{ent}_cd", f"{ent}_code"}
        own |= {f"{tok}_id" for tok in ent.split("_") if tok}  # card_txn -> txn_id, card_id
        cands = [c for c in prof.columns if c.family in ("integer", "string", "decimal") and c.null_fraction == 0
                 and c.unique_ratio >= PK_SHORTLIST_RATIO and ((t, c.name.lower()) in joined or c.name.lower() in own)]
        cands.sort(key=lambda c: ((t, c.name.lower()) not in joined, not KEY_NAME.search(c.name.lower()),
                                  prof.columns.index(c)))
        for c in cands[:3]:
            if _is_unique_exact(spark, prof, c.name):
                ev = ["unique+not-null"] + (["name"] if KEY_NAME.search(c.name.lower()) else []) + \
                     (["query"] if (t, c.name.lower()) in joined else [])
                rel.primary_keys[t] = {"columns": [c.name.lower()], "evidence": ev}
                break
    pk_col = {t: v["columns"][0] for t, v in rel.primary_keys.items() if len(v["columns"]) == 1}

    # ---------- candidates from the SQL and from names
    for j in facts.joins:
        candidates.setdefault((j["left_table"], j["left_column"], j["right_table"], j["right_column"]),
                              set()).add("query")
    for pt, pcol in pk_col.items():
        pfam = profiles[pt].column(pcol).family
        ent = _entity(pt)
        names = {pcol} | ({f"{ent}_id", f"{ent}id", f"{ent}_key"} if pcol in ("id", "key", f"{ent}_id") else set())
        for ct, cprof in profiles.items():
            if ct == pt:
                continue
            for c in cprof.columns:
                if c.name.lower() in names and c.family == pfam and pk_col.get(ct) != c.name.lower():
                    candidates.setdefault((ct, c.name.lower(), pt, pcol), set()).add("name")

    # ---------- cap the candidates found only by name: each one is two key scans of the data
    name_only = sorted(k for k, ev in candidates.items() if ev == {"name"})
    if len(name_only) > MAX_NAME_LINKS:
        for k in name_only[MAX_NAME_LINKS:]:
            del candidates[k]
        rel.warnings.append(f"{len(name_only)} candidate links matched only by column name; measured the first "
                            f"{MAX_NAME_LINKS} (MAX_NAME_LINKS)")

    # ---------- measure and classify every candidate (merging both directions of the same pair)
    measured = {}
    for (ta, ca, tb, cb), ev in candidates.items():
        key = tuple(sorted([(ta, ca), (tb, cb)]))
        if key in measured:
            measured[key]["evidence"] |= ev
            continue
        (ta, ca), (tb, cb) = key
        if log:
            log(f"measuring {ta}.{ca} ~ {tb}.{cb}")
        stats = measure_link(spark, profiles[ta], profiles[ta].column(ca), profiles[tb], profiles[tb].column(cb))
        measured[key] = {"a": (ta, ca), "b": (tb, cb), "evidence": set(ev), "stats": stats}
    for m in measured.values():
        (ta, ca), (tb, cb), s = m["a"], m["b"], m["stats"]
        ev = m["evidence"]
        a_parent = s["a_unique"] and (not s["b_unique"] or pk_col.get(ta) == ca)
        b_parent = s["b_unique"] and not a_parent
        if a_parent or b_parent:
            (pt, pc), (ct, cc) = ((ta, ca), (tb, cb)) if a_parent else ((tb, cb), (ta, ca))
            inclusion = s["b_in_a"] if a_parent else s["a_in_b"]
            kind = "fk" if inclusion >= inclusion_threshold else "partial_fk"
        else:
            # many-to-many: report the child side's rows found in the parent (b's rows in a)
            (pt, pc), (ct, cc), inclusion, kind = (ta, ca), (tb, cb), s["b_in_a"], "overlap"
        if "constraint" in ev and kind != "fk":
            rel.warnings.append(f"declared FK {ct}.{cc} -> {pt}.{pc} measures as {kind} "
                                f"({inclusion:.1%} of child rows match)")
        if not ({"constraint", "query"} & ev) and kind != "fk":
            continue  # a name match that the data doesn't support is not a link
        rel.links.append({"kind": kind, "parent_table": pt, "parent_column": pc, "child_table": ct,
                          "child_column": cc, "evidence": sorted(ev), "match_rate": inclusion,
                          "child_side": "a" if (ct, cc) == (ta, ca) else "b", "stats": s})

    _domains(spark, profiles, rel, pk_col, log)
    if log:
        log("composite keys")
    _composite_keys(spark, profiles, rel, facts)
    _dependencies(spark, profiles, rel, top_n, log)
    if check_correlations:
        _correlations(spark, profiles, rel, top_n, log=log)
    return rel


def _domains(spark, profiles, rel: Relationships, pk_col: dict, log=None):
    """Group linked columns into connected components and measure every member against the anchor."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for lk in rel.links:
        a, b = (lk["parent_table"], lk["parent_column"]), (lk["child_table"], lk["child_column"])
        parent[find(a)] = find(b)
    comps = {}
    for node in list(parent):
        comps.setdefault(find(node), []).append(node)
    for i, nodes in enumerate(sorted(comps.values(), key=lambda n: sorted(n))):
        degree = {n: sum(1 for lk in rel.links if n in ((lk["parent_table"], lk["parent_column"]),
                                                       (lk["child_table"], lk["child_column"]))) for n in nodes}
        uniq = {n for n in nodes for lk in rel.links if lk["kind"] != "overlap"
                and (lk["parent_table"], lk["parent_column"]) == n}

        def rank(n):
            c = profiles[n[0]].column(n[1])
            return (n not in uniq, pk_col.get(n[0]) != n[1], -degree[n], -c.distinct, n)

        anchor = min(nodes, key=rank)
        members = []
        for n in sorted(nodes, key=rank):
            c = profiles[n[0]].column(n[1])
            m = {"table": n[0], "column": n[1], "rows": profiles[n[0]].rows, "distinct": c.distinct,
                 "unique": n in uniq or (pk_col.get(n[0]) == n[1]), "parent": None}
            if n != anchor:
                if log:
                    log(f"domain {i}: measuring {n[0]}.{n[1]} against anchor {anchor[0]}.{anchor[1]}")
                s = measure_link(spark, profiles[n[0]], c, profiles[anchor[0]], profiles[anchor[0]].column(anchor[1]))
                m.update({"match_rate": s["a_in_b"], "distinct": s["distinct_a"], "overlap": s["distinct_both"],
                          "unique": s["a_unique"], "fanout": s["fanout_a"]})
                # the unique column this member is a child of, if any: its matched rows then reference that
                # column's rows, with the rows-per-parent shape measured on that link
                for lk in rel.links:
                    if (lk["child_table"], lk["child_column"]) == n and lk["kind"] != "overlap":
                        m["parent"] = {"table": lk["parent_table"], "column": lk["parent_column"],
                                       "match_rate": lk["match_rate"], "overlap": lk["stats"]["distinct_both"],
                                       "fanout": lk["stats"][f"fanout_{lk['child_side']}"],
                                       "parent_rows": profiles[lk["parent_table"]].rows}
                        break
            else:
                m.update({"match_rate": 1.0, "overlap": c.distinct, "fanout": c.fanout})
            members.append(m)
        rel.domains.append({"id": f"d{i}", "anchor": {"table": anchor[0], "column": anchor[1]}, "members": members})


def _composite_keys(spark, profiles, rel: Relationships, facts):
    """Non-unique join keys that are unique together with a column the SQL filters on (account_id + txn_source)."""
    filtered = {(t, c) for (t, c) in facts.literals}
    for d in rel.domains:
        for m in d["members"]:
            if m["unique"]:
                continue
            prof = profiles[m["table"]]
            for (t, c) in filtered:
                if t != m["table"] or c == m["column"]:
                    continue
                df, expected = spark.table(prof.name), prof.rows
                if prof.key_fraction < 1:  # all rows of the sampled keys: the pair is unique there or not at all
                    df = df.where(in_key_sample(key_expr(prof.column(m["column"])), prof.key_fraction))
                    expected = df.count()
                n = df.select(f"`{m['column']}`", f"`{c}`").distinct().count()
                if n == expected:
                    rel.composite_keys.append({"table": t, "columns": [m["column"], c],
                                               "note": f"{m['column']} is unique per {c}; the SQL filters {c}"})


def _key_columns(rel: Relationships, t: str) -> set[str]:
    keys = {c.lower() for c in rel.primary_keys.get(t, {}).get("columns", [])}
    for d in rel.domains:
        keys |= {m["column"] for m in d["members"] if m["table"] == t}
    return keys


def categorical_columns(prof: TableProfile, rel: Relationships, top_n: int) -> list:
    keys = _key_columns(rel, prof.short_name)
    return [c for c in prof.columns if c.name.lower() not in keys and c.top_values and c.top_coverage == 1.0
            and c.distinct <= top_n and c.family in ("string", "integer", "boolean", "decimal") and not c.encoded]


def _dependencies(spark, profiles, rel: Relationships, top_n: int, log=None):
    for t, prof in profiles.items():
        if log:
            log(f"dependencies: {t}")
        df = spark.table(prof.name)
        cats = categorical_columns(prof, rel, top_n)
        # determinants: workload-critical or code-named columns; pairs capped so wide tables stay cheap
        dets = [c for c in cats if c.roles or CODE_NAME.search(c.name.lower())][:FD_MAX_DETERMINANTS]
        pairs = [(a, b) for a in dets for b in cats if a is not b and a.distinct >= b.distinct >= 2
                 and a.distinct <= max(prof.rows / 5, 2)][:FD_MAX_PAIRS]
        fds = set()
        if pairs:
            # find candidates on a sample in one pass: B depends on A when every A value has a single B value
            sample = df.limit(FD_SAMPLE_ROWS) if prof.rows > FD_SAMPLE_ROWS else df
            kv = sample.select(F.explode(F.array(*[
                F.struct(F.lit(j).alias("p"), F.col(f"`{a.name}`").cast("string").alias("a"),
                         F.col(f"`{b.name}`").cast("string").alias("b")) for j, (a, b) in enumerate(pairs)])).alias("x")
            ).select("x.p", "x.a", "x.b").where("a is not null")
            bad = kv.groupBy("p", "a").agg(F.countDistinct("b").alias("nb")).where("nb > 1") \
                .select("p").distinct().collect()
            bad = {r["p"] for r in bad}
            fds = {(a.name, b.name) for j, (a, b) in enumerate(pairs) if j not in bad}
        # confirm candidates exactly (approximate counts can tie by accident)
        fds = {(a, b) for a, b in fds
               if df.select(f"`{a}`", f"`{b}`").distinct().count() == df.select(f"`{a}`").distinct().count()}
        order = {c.name: i for i, c in enumerate(prof.columns)}
        derived, bases = {}, set()

        def preferred(x, y):
            return (bool(CODE_NAME.search(x.lower())), -order[x]) > (bool(CODE_NAME.search(y.lower())), -order[y])

        for b in sorted({b for _, b in fds}, key=order.get):
            if b in bases:
                continue
            options = [a for a, bb in fds if bb == b and a not in derived and ((b, a) not in fds or preferred(a, b))]
            if options:
                a = min(options, key=lambda x: (prof.column(x).distinct, order[x]))
                derived[b] = a
                bases.add(a)
        for b, a in derived.items():
            rows = df.select(F.col(f"`{a}`").cast("string").alias("a"), F.col(f"`{b}`").cast("string").alias("b")) \
                .distinct().collect()
            rel.functional_dependencies.append({"table": t, "determinant": a, "dependent": b,
                                                "mapping": [[r["a"], r["b"]] for r in rows if r["a"] is not None]})

        # columns derived from a date/timestamp in the same row: to_date(ts), yyyyMM / yyyyMMdd month or day ids
        temporal = [c for c in prof.columns if c.kind in ("date", "timestamp") and c.null_fraction < 1]

        def could_be(c, lo, hi):
            return c.family == "integer" and c.min is not None and lo <= c.min and c.max <= hi

        checks = []
        for src in temporal:
            day = f"to_date({day_sql(src.name, src)})"
            for tgt in prof.columns:
                if tgt is src or tgt.null_fraction >= 1:
                    continue
                # rules are templates over {day}: the source column's calendar day as a DATE
                if tgt.kind == "date" and not tgt.encoded:
                    checks.append((tgt, src, f"`{tgt.name}` <=> {day}", "{day}"))
                elif tgt.kind == "date" and tgt.encoded:
                    fmt = tgt.encoded["format"]
                    checks.append((tgt, src, f"`{tgt.name}` <=> date_format({day}, '{fmt}')",
                                   f"date_format({{day}}, '{fmt}')"))
                else:
                    for fmt, lo, hi in (("yyyyMM", 190001, 210012), ("yyyyMMdd", 19000101, 21001231)):
                        if could_be(tgt, lo, hi):
                            checks.append((tgt, src, f"cast(`{tgt.name}` as string) <=> date_format({day}, '{fmt}')",
                                           f"cast(date_format({{day}}, '{fmt}') as int)"))
        found = set()
        for i in range(0, len(checks), 40):
            batch = checks[i:i + 40]
            row = df.agg(*[F.min(F.expr(f"CASE WHEN {cond} THEN 1 ELSE 0 END")).alias(f"c{j}")
                           for j, (_, _, cond, _) in enumerate(batch)]).first()
            for j, (tgt, src, _, rule) in enumerate(batch):
                if row[f"c{j}"] == 1 and tgt.name not in found:
                    found.add(tgt.name)
                    rel.derived_columns.append({"table": t, "column": tgt.name, "source": src.name, "rule": rule})


def _correlations(spark, profiles, rel: Relationships, top_n: int, max_levels: int = 50, log=None):
    """Report (not generate) associations between workload-critical columns."""
    derived = {(d["table"], d["dependent"]) for d in rel.functional_dependencies} | \
              {(d["table"], d["column"]) for d in rel.derived_columns}
    for t, prof in profiles.items():
        if log:
            log(f"correlations: {t}")
        crit = [c for c in prof.columns if c.roles]
        keys = _key_columns(rel, t)
        cats = [c for c in categorical_columns(prof, rel, top_n) if c.roles and c.distinct <= max_levels
                and (t, c.name) not in derived][:12]
        nums = [c for c in crit if c.kind in ("integer", "decimal", "float", "number") and c.name.lower() not in keys
                and (c.top_coverage or 0) < 1.0][:8]
        if not cats:
            continue
        df = _row_sample(spark, prof)  # association measures: a random row sample is enough
        num_expr = {n.name: F.expr(f"try_cast(`{n.name}` as double)") for n in nums}
        if nums:
            tot = df.agg(*[F.mean(e).alias(f"m{i}") for i, e in enumerate(num_expr.values())],
                         *[F.var_pop(e).alias(f"v{i}") for i, e in enumerate(num_expr.values())],
                         F.count(F.lit(1)).alias("n")).first()
            for c in cats:
                g = df.groupBy(f"`{c.name}`").agg(F.count(F.lit(1)).alias("n"),
                                                    *[F.mean(e).alias(f"m{i}") for i, e in enumerate(num_expr.values())]
                                                    ).collect()
                for i, n in enumerate(nums):
                    if not tot[f"v{i}"]:
                        continue
                    ss = sum(r["n"] * (float(r[f"m{i}"]) - float(tot[f"m{i}"])) ** 2 for r in g if r[f"m{i}"] is not None)
                    eta2 = ss / (tot["n"] * float(tot[f"v{i}"]))
                    if eta2 >= 0.06:
                        rel.correlations.append({"table": t, "columns": [c.name, n.name], "measure": "eta_squared",
                                                 "value": round(eta2, 4), "generated": False})
        for i, a in enumerate(cats):
            for b in cats[i + 1:]:
                rows = df.groupBy(f"`{a.name}`", f"`{b.name}`").count().collect()
                n = sum(r["count"] for r in rows)
                ra, rb, cell = {}, {}, {}
                for r in rows:
                    ra[r[0]] = ra.get(r[0], 0) + r["count"]
                    rb[r[1]] = rb.get(r[1], 0) + r["count"]
                    cell[(r[0], r[1])] = r["count"]
                chi2 = sum((cell.get((x, y), 0) - ra[x] * rb[y] / n) ** 2 / (ra[x] * rb[y] / n) for x in ra for y in rb)
                k = min(len(ra), len(rb)) - 1
                v = math.sqrt(chi2 / (n * k)) if n and k > 0 else 0.0
                if v >= 0.3:
                    rel.correlations.append({"table": t, "columns": [a.name, b.name], "measure": "cramers_v",
                                             "value": round(v, 4), "generated": False})

