"""Compare generated tables with their sources, without re-scanning the source.

The source side comes from the profiles ``01_profile_and_derive`` saved next to the config; only the targets are
profiled (same passes, same workload columns). Optionally, chosen queries run against both.
"""

from __future__ import annotations

import datetime as _dt
import json
import math

from pyspark.sql import functions as F

from . import profiler
from .profiler import ColumnProfile, TableProfile
from .sqlparse_util import rewrite_table_names

PROBES = [0.05, 0.25, 0.5, 0.75, 0.95]


def save_profiles(profiles: dict[str, TableProfile], path: str) -> None:
    with open(path, "w") as fh:
        json.dump({k: p.to_dict() for k, p in profiles.items()}, fh)


def profile_from_dict(raw: dict) -> TableProfile:
    raw = dict(raw)
    cols = [ColumnProfile(**c) for c in raw.pop("columns")]
    return TableProfile(**raw, columns=cols)


def load_profiles(path: str) -> dict[str, TableProfile]:
    with open(path) as fh:
        return {k: profile_from_dict(p) for k, p in json.load(fh).items()}


def _workload(cfg: dict):
    """Query literals and critical columns per source short name, from the config's ``queries`` section."""
    literals, critical = {}, {}
    for lit in cfg.get("queries", {}).get("literals", []):
        literals.setdefault(lit["table"], {})[lit["column"]] = set(lit["values"])
    for cc in cfg.get("queries", {}).get("critical_columns", []):
        critical.setdefault(cc["table"], {})[cc["column"]] = set(cc["roles"])
    return literals, critical


def profile_target(spark, cfg: dict, name: str) -> TableProfile:
    """Profile one generated table (by target name) with the same workload columns and literals as its source."""
    pt = next(pt for pt in cfg["post"]["tables"] if pt["name"] == name)
    short = pt["source"].split(".")[-1].lower()
    literals, critical = _workload(cfg)
    sizes = cfg.get("profiling") or {}  # profile the targets with the source's sample sizes
    return profiler.profile_table(
        spark, pt["target"], top_n=cfg["settings"]["top_n"], query_literals=literals.get(short),
        critical=critical.get(short),
        full_value_columns={c for c, r in critical.get(short, {}).items() if "join" in r},
        sample_rows=sizes.get("sample_rows", profiler.SAMPLE_ROWS),
        key_sample_rows=sizes.get("key_sample_rows", profiler.KEY_SAMPLE_ROWS))


def _status(value: float, ok: float, warn: float) -> str:
    return "PASS" if value <= ok else "WARN" if value <= warn else "FAIL"


def _q_at(knots, p):
    for (p0, x0), (p1, x1) in zip(knots, knots[1:]):
        if p0 <= p <= p1:
            return x0 if p1 == p0 else x0 + (p - p0) / (p1 - p0) * (x1 - x0)
    return knots[-1][1]


def _tvd(a: dict, b: dict) -> float:
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))


def _tvd_noise(freq: dict, n_src: int, n_tgt: int) -> float:
    """Expected TVD between two samples of the same distribution: sampling noise alone."""
    n_eff = 1 / (1 / max(n_src, 1) + 1 / max(n_tgt, 1))
    return 0.5 * sum(math.sqrt(2 * p * (1 - p) / (math.pi * n_eff)) for p in freq.values())


def _tvd_status(tvd: float, noise: float) -> str:
    return _status(tvd, max(0.05, 1.5 * noise), max(0.10, 2.5 * noise))


def _freq(values) -> dict:
    total = sum(n for _, n in values) or 1
    return {str(v): n / total for v, n in values}


def _share(weights, key) -> dict:
    out = {}
    for d, n in weights:
        out[key(d)] = out.get(key(d), 0) + n
    tot = sum(out.values()) or 1
    return {k: v / tot for k, v in out.items()}


def _match_rate(spark, child: str, ccol: str, parent: str, pcol: str) -> float:
    c = spark.table(child).select(F.col(f"`{ccol}`").cast("string").alias("k")).where("k is not null")
    p = spark.table(parent).select(F.col(f"`{pcol}`").cast("string").alias("k")).distinct()
    total = c.count()
    return 1 - c.join(p, "k", "left_anti").count() / total if total else 0.0


def _self_join_factor(spark, table: str, col: str) -> float:
    """Rows a self-join on `col` produces per input row: sum(count^2) / rows."""
    r = spark.table(table).where(F.col(f"`{col}`").isNotNull()).groupBy(f"`{col}`").count() \
        .agg(F.sum(F.col("count") * F.col("count")).alias("s"), F.sum("count").alias("n")).first()
    return float(r["s"] or 0) / max(r["n"] or 1, 1)


def validate(spark, cfg: dict, source_profiles: dict[str, TableProfile], sql_sources=None,
             run_queries: str = "none", dialect: str = "databricks", log=None, target_profiles=None):
    """Return ``(checks, query_results)`` as lists of dicts.

    ``run_queries``: 'none', 'all', or a comma list of query names (e.g. 'q021,q062') to run on source and target.
    ``target_profiles`` (source short name -> profile), when the targets were profiled beforehand (the job's for-each
    step); otherwise every target is profiled here.
    """
    say = log or (lambda *_: None)
    checks = []
    rel, settings = cfg["relationships"], cfg["settings"]
    tables = cfg["post"]["tables"]
    by_source = {pt["source"].split(".")[-1].lower(): pt for pt in tables}
    tgt_of = {k: pt["target"] for k, pt in by_source.items()}
    literals, critical = _workload(cfg)

    def add(area, table, column, check, source, target, status, note=""):
        checks.append({"area": area, "table": table, "column": column or "", "check": check,
                       "source": str(source), "target": str(target), "status": status, "note": note})

    def tiled(name: str, prov: dict) -> bool:
        """Whether a date column was extended; derived columns (to_date(ts), month ids) follow their source."""
        rule = prov.get(name, {}).get("rule", "")
        if rule.startswith("derived from "):
            name = rule[len("derived from "):].split(" ")[0]
        return "tiled" in str(prov.get(name, {}).get("details", ""))

    fixed_domains = {(m["table"], m["column"]) for d in cfg.get("domains", []) if d["kind"] == "fixed by copied table"
                     for m in d["members"]}

    def growth_for(sc: ColumnProfile, pt: dict, prov: dict) -> float:
        """Expected growth of distinct values, following the rule the generator actually used."""
        scale = settings["scale"] if settings.get("grow_ids", True) else 1.0
        rule = prov.get(sc.name, {}).get("rule", "")
        short = pt["source"].split(".")[-1].lower()
        if pt.get("copy") or (short, sc.name.lower()) in fixed_domains:
            return 1.0
        if sc.kind in ("date", "timestamp") or rule.startswith("derived from"):
            # dates, and values derived from a date (month ids), grow when the date window was extended
            return settings["scale"] if tiled(sc.name, prov) else 1.0
        if rule.startswith("key domain"):
            return scale  # keys grow with the tables they reference
        if rule.startswith("high-cardinality"):
            return scale if (sc.id_like or sc.unique_ratio >= 0.95) else 1.0
        return 1.0  # categorical codes, derived values, numbers

    # ---- profile every target (same workload columns and literals as the source), unless done already
    if target_profiles is None:
        target_profiles = {}
        for short, pt in by_source.items():
            say(f"profiling {pt['target']}")
            target_profiles[short] = profile_target(spark, cfg, pt["name"])

    # ---- structure and links
    for short, pt in by_source.items():
        sp, tp = source_profiles[short], target_profiles[short]
        note = "copied" if pt.get("copy") else f"{tp.rows / max(sp.rows, 1):.2f}x"
        add("structure", pt["name"], None, "row count", f"{sp.rows:,}", f"{tp.rows:,}",
            "PASS" if tp.rows == pt["rows"] else "FAIL", f"expected {pt['rows']:,} ({note})")
        for k in pt["primary_key"]:
            dupes = tp.rows - spark.table(pt["target"]).select(f"`{k}`").distinct().count()
            nulls = tp.column(k).null_fraction
            add("structure", pt["name"], k, "key unique & not null", "", f"{dupes} dupes, {nulls:.1%} null",
                "PASS" if dupes == 0 and nulls == 0 else "FAIL")
    for lk in rel["links"]:
        if lk["child_table"] not in tgt_of or lk["parent_table"] not in tgt_of:
            continue
        say(f"link {lk['child_table']}.{lk['child_column']} -> {lk['parent_table']}.{lk['parent_column']}")
        rate = _match_rate(spark, tgt_of[lk["child_table"]], lk["child_column"], tgt_of[lk["parent_table"]],
                           lk["parent_column"])
        want = lk["match_rate"]
        status = ("PASS" if rate >= 0.999 else "FAIL") if lk["kind"] == "fk" else _status(abs(rate - want), 0.05, 0.12)
        add("relationship", by_source[lk["child_table"]]["name"], lk["child_column"],
            f"{lk['kind']} -> {lk['parent_table']}.{lk['parent_column']} match rate", f"{want:.1%}", f"{rate:.1%}",
            status)

    # ---- key domains: distinct counts and skew of every member
    for d in cfg.get("domains", []):
        for m in d.get("members", []):
            short, col = m["table"], m["column"]
            if short not in target_profiles:
                continue
            pt = by_source[short]
            sc, tc = source_profiles[short].column(col), target_profiles[short].column(col)
            expected = sc.distinct * growth_for(sc, pt, cfg["provenance"].get(pt["name"], {}))
            add("domain", pt["name"], col, f"distinct values ({d['id']})", f"{sc.distinct:,}", f"{tc.distinct:,}",
                _status(abs(tc.distinct / max(expected, 1) - 1), 0.15, 0.35), f"expected ~{expected:,.0f}")
            if sc.fanout and tc.fanout:
                diff = max(abs(sc.fanout[k] - tc.fanout[k]) for k in ("top10pct", "gini"))
                add("domain", pt["name"], col, "rows-per-value shape",
                    {k: round(sc.fanout[k], 3) for k in ("top1", "top10pct", "gini")},
                    {k: round(tc.fanout[k], 3) for k in ("top1", "top10pct", "gini")},
                    _status(diff, 0.08, 0.15), f"largest difference {diff:.3f} (top-10% share, Gini)")

    # ---- dependencies
    for fd in rel["functional_dependencies"]:
        tgt = tgt_of.get(fd["table"])
        if not tgt:
            continue
        df = spark.table(tgt)
        pairs = df.select(fd["determinant"], fd["dependent"]).distinct().count()
        singles = df.select(fd["determinant"]).distinct().count()
        add("dependency", by_source[fd["table"]]["name"], fd["dependent"], f"determined by {fd['determinant']}", "",
            f"{pairs} pairs / {singles} values", "PASS" if pairs == singles else "FAIL")
    for dc in rel["derived_columns"]:
        tgt = tgt_of.get(dc["table"])
        if not tgt or by_source[dc["table"]].get("copy"):
            continue
        src = source_profiles[dc["table"]].column(dc["source"])
        expr = dc["rule"].replace("{day}", f"to_date({profiler.day_sql(src.name, src)})")
        bad = spark.table(tgt).where(f"not (cast(`{dc['column']}` as string) <=> cast({expr} as string))").count()
        add("dependency", by_source[dc["table"]]["name"], dc["column"], f"derived from {dc['source']}", "",
            f"{bad} rows differ", "PASS" if bad == 0 else "FAIL")
    for ck in rel.get("composite_keys", []):
        tgt = tgt_of.get(ck["table"])
        if tgt:
            df = spark.table(tgt)
            dupes = df.count() - df.select(*ck["columns"]).distinct().count()
            add("dependency", by_source[ck["table"]]["name"], "+".join(ck["columns"]), "composite key unique", "",
                f"{dupes} dupes", "PASS" if dupes == 0 else "FAIL")

    # ---- values the SQL filters on
    for short, cols in literals.items():
        if short not in target_profiles:
            continue
        for col, vals in cols.items():
            sc, tc = source_profiles[short].column(col), target_profiles[short].column(col)
            for v in sorted(vals, key=str):
                s_n, t_n = (sc.literal_counts or {}).get(str(v), 0), (tc.literal_counts or {}).get(str(v), 0)
                if s_n:
                    add("literal", by_source[short]["name"], col, f"rows with {v!r}", s_n, t_n,
                        "FAIL" if t_n == 0 else "PASS", f"{t_n / s_n:.2f}x")

    # ---- workload-critical columns
    self_cols = {(sj["table"], sj["left_column"]) for sj in rel.get("self_joins", [])}
    for short, tp in target_profiles.items():
        pt, sp = by_source[short], source_profiles[short]
        if pt.get("copy"):
            continue
        prov = cfg["provenance"].get(pt["name"], {})
        for sc in sp.columns:
            if not sc.roles:
                continue
            tc = tp.column(sc.name)
            rule = prov.get(sc.name, {}).get("rule", "")
            add("column", pt["name"], sc.name, "null rate", f"{sc.null_fraction:.2%}", f"{tc.null_fraction:.2%}",
                _status(abs(sc.null_fraction - tc.null_fraction), 0.01, 0.05))
            if rule.startswith("categorical") or rule.startswith("values of copied"):
                s_f, t_f = _freq(sc.top_values or []), _freq(tc.top_values or [])
                tvd, noise = _tvd(s_f, t_f), _tvd_noise(s_f, sp.rows, tp.rows)
                add("column", pt["name"], sc.name, "value mix", f"{sc.distinct} values", f"{tc.distinct} values",
                    _tvd_status(tvd, noise), f"TVD {tvd:.3f} (noise ~{noise:.3f})")
            elif sc.kind in ("date", "timestamp") and sc.quantiles and tc.quantiles:
                unit = 1 if sc.kind == "date" else 86400
                span_s = (sc.quantiles[-1][1] - sc.quantiles[0][1]) / unit
                span_t = (tc.quantiles[-1][1] - tc.quantiles[0][1]) / unit
                want = growth_for(sc, pt, prov)
                add("column", pt["name"], sc.name, "time span (days)", round(span_s), round(span_t),
                    _status(abs(span_t / max(span_s, 1) / want - 1), 0.05, 0.15), f"expected {want:.0f}x the source")
                if sc.day_weights and tc.day_weights:
                    mixes = [("day-of-week mix", lambda d: _dt.date.fromisoformat(d).weekday())]
                    if span_s >= 364:  # a shorter source window can't show its month-of-year pattern
                        mixes.insert(0, ("month-of-year mix", lambda d: d[5:7]))
                    for label, key in mixes:
                        s_f, t_f = _share(sc.day_weights, key), _share(tc.day_weights, key)
                        tvd = _tvd(s_f, t_f)
                        add("column", pt["name"], sc.name, label, "", "",
                            _tvd_status(tvd, _tvd_noise(s_f, sp.rows, tp.rows)), f"TVD {tvd:.3f}")
            elif sc.quantiles and tc.quantiles and rule.startswith("number"):
                width = (_q_at(sc.quantiles, 0.95) - _q_at(sc.quantiles, 0.05)) or 1.0
                err = max(abs(_q_at(sc.quantiles, p) - _q_at(tc.quantiles, p)) for p in PROBES) / width
                add("column", pt["name"], sc.name, "quantiles p5/p50/p95",
                    [round(_q_at(sc.quantiles, p), 2) for p in (0.05, 0.5, 0.95)],
                    [round(_q_at(tc.quantiles, p), 2) for p in (0.05, 0.5, 0.95)], _status(err, 0.05, 0.10),
                    f"max error {err:.1%} of the p5-p95 width")
            if {"count_distinct", "partition", "group"} & set(sc.roles):
                expected = sc.distinct * growth_for(sc, pt, prov)
                add("column", pt["name"], sc.name, "distinct values", f"{sc.distinct:,}", f"{tc.distinct:,}",
                    _status(abs(tc.distinct / max(expected, 1) - 1), 0.15, 0.35), f"expected ~{expected:,.0f}")
            if (short, sc.name.lower()) in self_cols:
                s_f = _self_join_factor(spark, pt["source"], sc.name)
                t_f = _self_join_factor(spark, pt["target"], sc.name)
                add("column", pt["name"], sc.name, "self-join rows per input row", f"{s_f:,.1f}", f"{t_f:,.1f}",
                    "PASS", f"{t_f / max(s_f, 1e-9):.2f}x (3x is expected when the key's distinct count is fixed)")

    # ---- queries (optional: the workload's queries are heavy)
    query_results = []
    if sql_sources and run_queries and run_queries.lower() != "none":
        wanted = None if run_queries.lower() == "all" else {q.strip() for q in run_queries.split(",") if q.strip()}
        src_sql = dict(rewrite_table_names(sql_sources, {k: pt["source"] for k, pt in by_source.items()}, dialect))
        tgt_sql = dict(rewrite_table_names(sql_sources, tgt_of, dialect))
        for name, qs in src_sql.items():
            if wanted and name not in wanted:
                continue
            say(f"running {name}")
            try:
                n_s, n_t = spark.sql(qs).count(), spark.sql(tgt_sql[name]).count()
                query_results.append({"query": name, "source_rows": n_s, "target_rows": n_t})
                add("query", "", "", name, f"{n_s} rows", f"{n_t} rows", "FAIL" if n_s and not n_t else "PASS")
            except Exception as e:  # noqa: BLE001
                add("query", "", "", name, "", "", "WARN", f"did not run: {str(e).splitlines()[0][:150]}")
    return checks, query_results


def summary(checks: list[dict]) -> str:
    counts = {s: sum(1 for c in checks if c["status"] == s) for s in ("PASS", "WARN", "FAIL")}
    lines = [f"**{counts['PASS']} pass, {counts['WARN']} warn, {counts['FAIL']} fail**"]
    for c in checks:
        if c["status"] != "PASS":
            lines.append(f"- {c['status']} {c['table']}.{c['column']} {c['check']}: source {c['source']} / "
                         f"target {c['target']} {c['note']}")
    return "\n".join(lines)
