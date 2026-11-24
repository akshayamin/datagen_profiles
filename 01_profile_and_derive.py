# Databricks notebook source
# MAGIC %md
# MAGIC # 1 · Profile source tables and derive dbldatagen configs
# MAGIC
# MAGIC Reads the source tables, their DDL and the SQL workload, then writes a **YAML config** that `02_generate` turns
# MAGIC into scaled synthetic tables, plus the source profiles `03_validate` compares against. **Nothing is written to any
# MAGIC table**; the only outputs are files under `config_path` (default `/Workspace/Users/<you>/customer_synth_runs/`).
# MAGIC
# MAGIC 1. **SQL**: every statement in `queries_path` (a `.sql` file or a folder of them) is parsed on its own; joins,
# MAGIC    filter literals, casts and the columns the workload partitions / orders / groups / joins / counts on are
# MAGIC    collected. Broken statements are reported and skipped.
# MAGIC 2. **Profile**, one table at a time; each profile is saved as soon as it's done, and a rerun with `resume = yes`
# MAGIC    reuses the ones made with the same settings and SQL. What is read:
# MAGIC    * the **full table**, aggregates only: row count, null rates, distinct counts, min / max, filter-literal counts;
# MAGIC    * a **random row sample** of about `sample_rows` rows: quantiles, value frequencies, per-day counts, string
# MAGIC      lengths and formats;
# MAGIC    * a **key-hash sample** of about `key_sample_rows` rows: rows per key. Every row of a sampled key is kept, with
# MAGIC      the same rule on every table, so join match rates and skew stay exact for those keys.
# MAGIC
# MAGIC    Tables smaller than the sample sizes are read in full; `0` reads every table in full.
# MAGIC 3. **Relationships**: Unity Catalog constraints, SQL joins, names and data → measured links (FK, partial FK,
# MAGIC    overlap) grouped into key domains, plus composite keys, code → description pairs and derived date columns.
# MAGIC 4. **Derive**: a rule per column (see the provenance table at the end).
# MAGIC
# MAGIC **What the output files contain** (review before sharing): per column, null rate, distinct count, min / max,
# MAGIC quantiles and the most frequent values (up to `top_n`); per-position character sets of mostly-unique strings
# MAGIC (never their values); counts of the SQL's filter literals; code → description pairs; the SQL text; and the
# MAGIC **full rows of small lookup tables** (≤ `copy_max_rows` rows, listed in the last cell; `copy_max_rows = 0` keeps
# MAGIC them out, and they are then generated instead).
# MAGIC
# MAGIC `source_tables = @queries` profiles exactly the tables the SQL uses.

# COMMAND ----------

# MAGIC %md
# MAGIC The library wheel ships in `wheels/` next to this notebook. If it lives elsewhere, change the path below.

# COMMAND ----------

# MAGIC %pip install ./wheels/dbldatagen-0.4.0.post2-py3-none-any.whl "pydantic>=2.0" sqlglot pyyaml

# COMMAND ----------

try:
    dbutils.library.restartPython()  # noqa: F821 - notebook only; reload Python so the new packages are importable
except NameError:
    pass

# COMMAND ----------

import os
import sys
import time

BASE = os.getcwd()  # the notebook's folder; the `synth` package sits next to it
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from synth import pipeline, profiler  # noqa: E402
from synth.nb import flag, param, show, show_provenance, show_relationships, show_tables  # noqa: E402

_dbutils = globals().get("dbutils")
_display = globals().get("display")
_t0 = time.time()


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} (+{time.time() - _t0:,.0f}s) {msg}", flush=True)


def get_spark():
    try:
        return spark  # noqa: F821 - provided by the Databricks runtime
    except NameError:
        from databricks.connect import DatabricksSession
        return DatabricksSession.builder.serverless(True).getOrCreate()


session = get_spark()


def default_config_path(prefix: str) -> str:
    """On Databricks: the user's customer_synth_runs folder (kept apart from the code); locally: configs/ here."""
    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        user = session.sql("SELECT current_user()").first()[0]
        return f"/Workspace/Users/{user}/customer_synth_runs/{prefix}config.yaml"
    return os.path.join(BASE, "configs", f"{prefix}config.yaml")


SOURCE_CATALOG = param(_dbutils, "source_catalog", "customer_tests")
SOURCE_SCHEMA = param(_dbutils, "source_schema", "datagen")
SOURCE_TABLES = param(_dbutils, "source_tables", "@queries", "source_tables (@queries, comma list or pattern)")
QUERIES_PATH = param(_dbutils, "queries_path", os.path.join(BASE, "queries"), "queries_path (.sql file or folder)")
SCALE = float(param(_dbutils, "scale", "3"))
DATE_MODE = param(_dbutils, "date_mode", "extend", "date_mode (extend | densify)")
TOP_N = int(param(_dbutils, "top_n", "100"))
SEED = int(param(_dbutils, "seed", "42"))
SOURCE_PREFIX = param(_dbutils, "source_prefix", "")
TARGET_PREFIX = param(_dbutils, "target_prefix", "customer_3x_")
TARGET_CATALOG = param(_dbutils, "target_catalog", SOURCE_CATALOG) or SOURCE_CATALOG
TARGET_SCHEMA = param(_dbutils, "target_schema", SOURCE_SCHEMA) or SOURCE_SCHEMA
COPY_MAX_ROWS = int(param(_dbutils, "copy_max_rows", "1000", "copy_max_rows (lookups embedded as-is; 0 = none)"))
GROW_IDS = flag(param(_dbutils, "grow_ids", "yes", "grow_ids (IDs scale with rows; codes stay fixed)"))
READ_CONSTRAINTS = flag(param(_dbutils, "read_constraints", "yes", "read_constraints (UC PK/FK)"))
SAMPLE_ROWS = int(param(_dbutils, "sample_rows", str(profiler.SAMPLE_ROWS), "sample_rows (distributions; 0 = all)"))
KEY_SAMPLE_ROWS = int(param(_dbutils, "key_sample_rows", str(profiler.KEY_SAMPLE_ROWS),
                            "key_sample_rows (key skew and joins; 0 = all)"))
RESUME = flag(param(_dbutils, "resume", "yes", "resume (reuse profiles saved by an earlier run)"))
CONFIG_PATH = param(_dbutils, "config_path", default_config_path(TARGET_PREFIX))
assert DATE_MODE in ("extend", "densify"), "date_mode must be extend or densify"
assert os.path.exists(QUERIES_PATH), (
    f"queries_path {QUERIES_PATH} doesn't exist: put the workload's .sql files in a `queries` folder next to this "
    f"notebook, or set queries_path to a .sql file or folder")
log(f"outputs go to {CONFIG_PATH} and {pipeline.work_dir(CONFIG_PATH)}/")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. SQL and source tables

# COMMAND ----------

tables, facts = pipeline.plan_tables(session, SOURCE_CATALOG, SOURCE_SCHEMA, SOURCE_TABLES, QUERIES_PATH, TARGET_PREFIX,
                                     log=log)
show(session, [{"table": t, "columns": len(session.table(t).columns),
                "critical_columns": len(facts.critical(t.split('.')[-1].lower()))} for t in tables], _display)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Profile

# COMMAND ----------

profiles = {}
for i, fqn in enumerate(tables, 1):
    short = pipeline.short_name(fqn)
    t1 = time.time()
    profiles[short], loaded = pipeline.profile_or_load(
        session, fqn, facts, TOP_N, CONFIG_PATH, resume=RESUME, sample_rows=SAMPLE_ROWS,
        key_sample_rows=KEY_SAMPLE_ROWS, log=lambda m: log(f"  {m}"))
    p = profiles[short]
    how = "loaded from an earlier run" if loaded else f"profiled in {time.time() - t1:,.0f}s"
    log(f"[{i}/{len(tables)}] {fqn}: {p.rows:,} rows, {len(p.columns)} columns, {how}"
        + (f" (row sample {p.row_fraction:.4%}, key sample {p.key_fraction:.4%})" if p.sampled or p.key_fraction < 1
           else ""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Relationships, then the config

# COMMAND ----------

settings = dict(scale=SCALE, date_mode=DATE_MODE, top_n=TOP_N, seed=SEED, source_prefix=SOURCE_PREFIX,
                target_prefix=TARGET_PREFIX, target_catalog=TARGET_CATALOG, target_schema=TARGET_SCHEMA,
                copy_max_rows=COPY_MAX_ROWS, grow_ids=GROW_IDS)
cfg, rel = pipeline.derive_all(session, tables, profiles, facts, settings, READ_CONSTRAINTS, CONFIG_PATH, log=log,
                               profiling={"sample_rows": SAMPLE_ROWS, "key_sample_rows": KEY_SAMPLE_ROWS})
show_relationships(session, rel, _display)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. The derived config
# MAGIC Written above together with the relationships (key skew is calibrated by generating small candidate tables at
# MAGIC the target proportions, a minute or two).

# COMMAND ----------

show_tables(session, cfg, _display)

# COMMAND ----------

# MAGIC %md
# MAGIC ### How each column will be generated, and warnings to review

# COMMAND ----------

show_provenance(session, cfg, _display)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Small lookup tables embedded in the config, row for row
# MAGIC These rows leave with the config so `02_generate` can recreate the lookups exactly. Set `copy_max_rows = 0` to
# MAGIC keep them out (they are then generated like every other table).

# COMMAND ----------

embedded = [pt for pt in cfg["post"]["tables"] if pt.get("copy")]
show(session, [{"table": pt["source"], "rows": len(pt.get("copy_rows") or []), "columns": len(pt["columns"])}
               for pt in embedded], _display)
for pt in embedded:
    print(pt["source"])
    rows = [dict(zip([c["name"] for c in pt["columns"]], r)) for r in pt.get("copy_rows") or []]
    show(session, rows, _display)
