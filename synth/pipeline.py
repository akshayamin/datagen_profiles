"""The steps of the pipeline, shared by the interactive notebooks (01/02/03) and the job notebooks (jobs/).

The jobs run each step as its own task and hand results over through a work folder next to the config:

    <config>_work/plan.json           tables to profile + the settings of this run (readable)
    <config>_work/facts.pkl           the parsed SQL (QueryFacts)
    <config>_work/profiles/<t>.json   one source profile per table (job 1, for-each)
    <config>_work/targets/<t>.json    one target profile per table (job 2, for-each)

Job 1 (profile & derive) ends with the YAML config; job 2 (generate & validate) starts from it, so the config can be
reviewed and edited in between.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import shutil

from pyspark.sql import functions as F

from . import derive, profiler, relationships, sqlparse_util, validate
from .nb import resolve_tables
from .profiler import TableProfile

SETTINGS = ("scale", "date_mode", "top_n", "seed", "source_prefix", "target_prefix", "target_catalog", "target_schema",
            "copy_max_rows", "grow_ids")


def work_dir(config_path: str) -> str:
    return config_path.rsplit(".", 1)[0] + "_work"


def profiles_path(config_path: str) -> str:
    return config_path.rsplit(".", 1)[0] + "_profiles.json"


def short_name(fqn: str) -> str:
    return fqn.split(".")[-1].lower()


# ---- job 1: plan -> profile (per table) -> derive

def plan_tables(spark, source_catalog: str, source_schema: str, source_tables: str, queries_path: str,
                target_prefix: str, log=print):
    """Parse the SQL and choose the source tables. Returns ``(tables, facts)``."""
    sql_sources = sqlparse_util.read_sql(queries_path) if os.path.exists(queries_path) else []
    log(f"{len(sql_sources)} SQL file(s) from {queries_path}")
    by_queries = source_tables.strip() == "@queries"
    candidates = resolve_tables(spark, source_catalog, source_schema, "*" if by_queries else source_tables,
                                exclude_prefix=target_prefix)
    if by_queries and sql_sources:
        # only tables whose name appears in the SQL text: the rest of the schema is never touched (a superset of
        # the tables the SQL uses; the parser below narrows it down)
        words = set(re.findall(r"[a-z0-9_]+", " ".join(text for _, text in sql_sources).lower()))
        unused = len(candidates)
        candidates = [t for t in candidates if short_name(t) in words]
        log(f"{unused:,} tables in {source_catalog}.{source_schema}; {len(candidates)} named in the SQL")
    columns, unreadable = {}, []
    for t in candidates:  # column names only (no data); a table that can't be read is reported, not fatal
        try:
            columns[short_name(t)] = [f.name for f in spark.table(t).schema.fields]
        except Exception as e:  # noqa: BLE001 - broken views, missing files, no permission
            unreadable.append((t, (str(e).strip().splitlines() or [type(e).__name__])[0][:200]))
    facts = sqlparse_util.analyze_queries(sql_sources, columns) if sql_sources else sqlparse_util.QueryFacts()
    tables = [t for t in candidates if short_name(t) in columns and (not by_queries or short_name(t) in facts.tables())]
    log(f"{len(facts.queries)} statements parsed, {len(facts.parse_errors)} skipped; {len(tables)} tables selected")
    for t, err in unreadable:
        print(f"  WARNING: skipped {t}, which can't be read: {err}")
    if facts.unresolved_tables:
        print(f"  WARNING: the SQL uses {len(facts.unresolved_tables)} table(s) not found (or not readable) in "
              f"{source_catalog}.{source_schema}, so they are not profiled: {', '.join(sorted(facts.unresolved_tables))}")
        print("    if they live in another schema or catalog, list every table in source_tables with fully-qualified "
              "names (catalog.schema.table, comma-separated)")
    for e in facts.parse_errors:
        print("  could not parse", e["query"], "-", e["error"])
    assert tables, (f"no source tables selected from {source_catalog}.{source_schema}: check source_catalog, "
                    f"source_schema and the table names in the SQL (warnings above)")
    return tables, facts


def save_plan(config_path: str, tables: list[str], facts, settings: dict, read_constraints: bool = True) -> None:
    """Start a fresh work folder (profiles from an earlier run are removed) holding the plan and the parsed SQL."""
    wd = work_dir(config_path)
    shutil.rmtree(wd, ignore_errors=True)
    os.makedirs(os.path.join(wd, "profiles"))
    with open(os.path.join(wd, "plan.json"), "w") as fh:
        json.dump({"tables": tables, "settings": settings, "read_constraints": read_constraints}, fh, indent=1)
    with open(os.path.join(wd, "facts.pkl"), "wb") as fh:
        pickle.dump(facts, fh)


def load_plan(config_path: str):
    """``(tables, settings, read_constraints, facts)`` saved by :func:`save_plan`."""
    wd = work_dir(config_path)
    with open(os.path.join(wd, "plan.json")) as fh:
        plan = json.load(fh)
    with open(os.path.join(wd, "facts.pkl"), "rb") as fh:
        facts = pickle.load(fh)
    return plan["tables"], plan["settings"], plan["read_constraints"], facts


def profile_source(spark, fqn: str, facts, top_n: int, log=None, sample_rows: int | None = profiler.SAMPLE_ROWS,
                   key_sample_rows: int | None = profiler.KEY_SAMPLE_ROWS) -> TableProfile:
    """Profile one source table with the workload's critical columns, literals and casts."""
    short = short_name(fqn)
    crit = facts.critical(short)
    profile = profiler.profile_table(
        spark, fqn, top_n=top_n, query_literals=profiler.literal_columns(facts.literals, short), critical=crit,
        casts=profiler.table_items(facts.casts, short), full_value_columns={c for c, r in crit.items() if "join" in r},
        log=log, sample_rows=sample_rows, key_sample_rows=key_sample_rows)
    profile.fingerprint = fingerprint(fqn, facts, top_n, sample_rows, key_sample_rows, profile.rows)
    return profile


def fingerprint(fqn: str, facts, top_n: int, sample_rows, key_sample_rows, rows: int) -> str:
    """What a profile depends on: the settings, this table's SQL facts and its row count."""
    short = short_name(fqn)

    def norm(d):
        return {str(k): sorted(map(str, v)) if isinstance(v, (set, list, tuple)) else str(v)
                for k, v in sorted((d or {}).items())}

    basis = {"version": profiler.PROFILE_VERSION, "table": fqn.lower(), "top_n": top_n, "sample_rows": sample_rows or 0,
             "key_sample_rows": key_sample_rows or 0, "rows": rows, "critical": norm(facts.critical(short)),
             "literals": norm(profiler.literal_columns(facts.literals, short)),
             "casts": norm(profiler.table_items(facts.casts, short))}
    return hashlib.sha256(json.dumps(basis, sort_keys=True).encode()).hexdigest()[:16]


def profile_or_load(spark, fqn: str, facts, top_n: int, config_path: str, resume: bool = True,
                    sample_rows: int | None = profiler.SAMPLE_ROWS,
                    key_sample_rows: int | None = profiler.KEY_SAMPLE_ROWS, log=None) -> tuple[TableProfile, bool]:
    """``(profile, loaded)``: the profile saved by an earlier run when it was made the same way, else a new one.

    Each new profile is saved as soon as it is done, so a failure later (another table, the derive step) loses
    nothing. Tables don't depend on each other here; everything across tables happens in :func:`derive_all`.
    """
    path = source_profile_path(config_path, fqn)
    if resume and os.path.exists(path):
        saved = load_profile(path)
        expected = fingerprint(fqn, facts, top_n, sample_rows, key_sample_rows, spark.table(fqn).count())
        if saved.fingerprint == expected:
            return saved, True
        if log:
            log(f"{fqn}: saved profile was made with other settings, SQL or row count; profiling again")
    profile = profile_source(spark, fqn, facts, top_n, log=log, sample_rows=sample_rows,
                             key_sample_rows=key_sample_rows)
    save_profile(profile, path)
    return profile, False


def save_profile(profile: TableProfile, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(profile.to_dict(), fh)


def load_profile(path: str) -> TableProfile:
    with open(path) as fh:
        return validate.profile_from_dict(json.load(fh))


def source_profile_path(config_path: str, fqn: str) -> str:
    return os.path.join(work_dir(config_path), "profiles", f"{short_name(fqn)}.json")


def load_source_profiles(config_path: str, tables: list[str]) -> dict[str, TableProfile]:
    missing = [t for t in tables if not os.path.exists(source_profile_path(config_path, t))]
    assert not missing, f"no profile yet for {missing}: run the profile step for them first"
    return {short_name(t): load_profile(source_profile_path(config_path, t)) for t in tables}


def derive_all(spark, tables: list[str], profiles: dict[str, TableProfile], facts, settings: dict,
               read_constraints: bool, config_path: str, log=print, profiling: dict | None = None):
    """Relationships + config. Writes the config and the merged source profiles; returns ``(cfg, rel)``.

    Tables copied unchanged (small lookups) carry their rows in the config (``copy_rows``), so generating from
    the config never needs the source. ``profiling`` (sample sizes) is recorded so targets are profiled alike.
    """
    declared, notes = relationships.read_declared_constraints(tables) if read_constraints else ({}, [])
    for n in notes:
        print("note:", n)
    rel = relationships.identify(spark, profiles, facts, declared=declared, top_n=settings["top_n"],
                                 log=lambda m: log(f"  {m}"))
    log("relationships done")
    cfg = derive.derive_config(spark, profiles, rel, facts, derive.Settings(**settings), log=lambda m: log(f"  {m}"))
    embed_copied_rows(spark, cfg, log=log)
    if profiling:
        cfg["profiling"] = dict(profiling)
    os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
    validate.save_profiles(profiles, profiles_path(config_path))
    cfg["profiles_path"] = profiles_path(config_path)
    from .generate import save_config
    save_config(cfg, config_path)
    log(f"config written to {config_path}")
    return cfg, rel


def embed_copied_rows(spark, cfg: dict, log=print) -> None:
    """Put the rows of every ``copy: true`` table into the config, as text (NULL as None), in column order."""
    for pt in cfg["post"]["tables"]:
        if not pt.get("copy"):
            continue
        names = [c["name"] for c in pt["columns"]]
        rows = spark.table(pt["source"]).select(*[F.col(f"`{n}`").cast("string") for n in names]).collect()
        pt["copy_rows"] = [[r[i] for i in range(len(names))] for r in rows]
        log(f"{pt['source']}: {len(rows):,} rows embedded in the config (copied unchanged)")


# ---- job 2: check config -> generate (per table) -> constraints, profile targets (per table) -> validate

def clear_target_profiles(config_path: str) -> None:
    """Remove target profiles from an earlier generate run, so validation can't compare against stale ones."""
    shutil.rmtree(os.path.join(work_dir(config_path), "targets"), ignore_errors=True)


def target_profile_path(config_path: str, name: str) -> str:
    return os.path.join(work_dir(config_path), "targets", f"{name.lower()}.json")


def load_target_profiles(config_path: str, cfg: dict) -> dict[str, TableProfile]:
    """Target profiles saved by the for-each step, keyed by source short name (as ``validate`` expects)."""
    out, missing = {}, []
    for pt in cfg["post"]["tables"]:
        path = target_profile_path(config_path, pt["name"])
        if os.path.exists(path):
            out[short_name(pt["source"])] = load_profile(path)
        else:
            missing.append(pt["name"])
    assert not missing, f"no target profile yet for {missing}: run the profile_target step for them first"
    return out
