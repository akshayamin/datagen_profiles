"""Small helpers shared by the notebooks (widgets, display) that also work outside Databricks."""

from __future__ import annotations

import os


def param(dbutils, name: str, default: str, label: str | None = None) -> str:
    """Notebook widget value; outside a notebook, the environment variable NAME (upper case) or the default."""
    if dbutils is not None and hasattr(dbutils, "widgets"):
        dbutils.widgets.text(name, default, label or name)
        return dbutils.widgets.get(name).strip()
    return os.environ.get(name.upper(), default).strip()


def set_task_value(dbutils, key: str, value) -> None:
    """Pass a value to later job tasks (``{{tasks.<task>.values.<key>}}``); a no-op outside a job."""
    jobs = getattr(dbutils, "jobs", None) if dbutils is not None else None
    if jobs is not None:
        jobs.taskValues.set(key=key, value=value)


def flag(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def show(spark, rows: list[dict], display_fn=None, limit: int = 1000):
    """``display`` a list of dicts as a table in a notebook, or print it elsewhere."""
    if not rows:
        print("(none)")
        return
    cols = list(rows[0].keys())
    clean = [{c: ("" if r.get(c) is None else str(r.get(c))) for c in cols} for r in rows[:limit]]
    if display_fn is not None:
        display_fn(spark.createDataFrame(clean))
        return
    widths = {c: min(60, max(len(c), *(len(r[c]) for r in clean))) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in clean:
        print("  ".join(r[c][:60].ljust(widths[c]) for c in cols))


def resolve_tables(spark, catalog: str, schema: str, spec: str, exclude_prefix: str = "") -> list[str]:
    """Fully-qualified tables from a comma list (``a,b``) or a name pattern (``customer_*``).

    Tables starting with ``exclude_prefix`` (the target prefix) are skipped, so re-running after generation
    doesn't profile the synthetic tables as if they were source tables.
    """
    names = [s.strip() for s in spec.split(",") if s.strip()]
    if len(names) == 1 and "*" in names[0]:
        rows = spark.sql(f"SHOW TABLES IN {catalog}.{schema} LIKE '{names[0]}'").collect()
        names = sorted(r["tableName"] for r in rows if not r["isTemporary"])
    names = [n for n in names if not (exclude_prefix and n.lower().startswith(exclude_prefix.lower()))]
    return [n if n.count(".") == 2 else f"{catalog}.{schema}.{n}" for n in names]


def show_relationships(spark, rel, display_fn=None):
    """Links, key domains and within-table rules that ``relationships.identify`` found."""
    show(spark, [{"kind": lk["kind"], "child": f"{lk['child_table']}.{lk['child_column']}",
                  "parent": f"{lk['parent_table']}.{lk['parent_column']}", "match_rate": f"{lk['match_rate']:.1%}",
                  "evidence": ", ".join(lk["evidence"])} for lk in rel.links], display_fn)
    show(spark, [{"domain": d["id"], "anchor": f"{d['anchor']['table']}.{d['anchor']['column']}",
                  "members": ", ".join(f"{m['table']}.{m['column']}" for m in d["members"])} for d in rel.domains],
         display_fn)
    show(spark, [{"table": c["table"], "rule": c["note"]} for c in rel.composite_keys]
         + [{"table": d["table"], "rule": f"{d['dependent']} determined by {d['determinant']}"}
            for d in rel.functional_dependencies]
         + [{"table": d["table"], "rule": f"{d['column']} = {d['rule'].replace('{day}', 'day(' + d['source'] + ')')}"}
            for d in rel.derived_columns], display_fn)


def show_tables(spark, cfg: dict, display_fn=None):
    show(spark, [{"source": pt["source"], "target": pt["target"], "source_rows": f"{pt['source_rows']:,}",
                  "target_rows": f"{pt['rows']:,}", "how": "copied" if pt["copy"] else "generated"}
                 for pt in cfg["post"]["tables"]], display_fn)


def show_provenance(spark, cfg: dict, display_fn=None):
    """How each column will be generated, and the warnings to review before generating."""
    show(spark, [{"table": t, "column": c, "rule": p["rule"], "details": p["details"]}
                 for t, cols in cfg["provenance"].items() for c, p in cols.items()], display_fn)
    print("warnings:" if cfg["warnings"] else "no warnings")
    for w in cfg["warnings"]:
        print("  -", w)
