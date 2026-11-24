"""Per-table and per-column statistics, computed in Spark. Only aggregates are collected - never raw rows.

The profile is built in a few passes, each reading only the columns it needs (columnar storage keeps them cheap):

1. counts, null rates, distinct counts, min/max
2. string columns that hold dates, timestamps, times or numbers (``'2022-01-31'``, ``'20221216'``, ``'12.50'``)
3. quantiles (numbers, dates, timestamps - including string-encoded ones) and time-of-day quantiles
4. one combined frequency pass: value frequencies, per-day histograms and string lengths
5. how often each query literal occurs
6. per-position character sets of high-cardinality strings (their "format"), never the values themselves
7. rows-per-value shape (skew) of high-cardinality key and workload columns

What is read where, so large (multi-TB) tables stay affordable:

* **full table**, aggregates only (no shuffle): row count, null rates, approximate distinct counts, min/max, query
  literal counts, and the parsed min/max of string-encoded columns;
* **random row sample** (``sample_rows``): quantiles, value frequencies, per-day histograms, lengths, string encodings
  and formats; counts are scaled back up. Join columns with few values keep exact full-table frequencies;
* **key-hash sample** (``key_sample_rows``): rows-per-key skew. Every row of a sampled key is kept (``xxhash64`` of
  the key below a threshold), so per-key counts stay exact; relationships use the same rule on both sides of a join.

``0`` for either setting reads the full table. Strings whose values are mostly unique (names, card numbers) are only
ever summarised as formats.
"""

from __future__ import annotations

import datetime as _dt
import decimal as _decimal
import re
from dataclasses import asdict, dataclass, field

from pyspark.sql import functions as F
from pyspark.sql.window import Window

PROFILE_VERSION = 3  # bump when what a profile holds changes: profiles saved by older code are then not reused
QUANTILE_PROBS = [0.0, 0.001, 0.005] + [i / 100 for i in range(1, 100)] + [0.995, 0.999, 1.0]
CATEGORICAL_PROFILE_MAX = 10_000  # value frequencies are only collected below this many distinct values
MAX_DAYS = 10_000  # temporal columns spanning fewer distinct days get an exact per-day histogram
SAMPLE_ROWS = 20_000_000  # tables larger than this profile their distributions on a random row sample
KEY_SAMPLE_ROWS = 200_000_000  # tables larger than this measure key skew (and links) on a key-hash sample
KEY_HASH_BUCKETS = 1_000_000
# relative error of the approximate distinct counts (HyperLogLog). 0.05 (Spark's default) keeps each sketch at 512
# registers; 0.01 needs 16,384 per column, and 40 of those in one aggregate made pass 1 about 25x slower (minutes on
# wide tables). Low-cardinality counts are made exact by the frequency pass, and join keys are measured exactly in
# relationships, so this only affects the sizing of other high-cardinality columns (+-5%).
DISTINCT_RSD = 0.05
UNIQUE_CHECK_RATIO = 0.85  # columns estimated at least this unique get an exact uniqueness check (pass 1b)
FORMAT_SAMPLE_ROWS = 1_000_000
MAX_POSITIONS = 48

_INTEGER = ("tinyint", "smallint", "int", "bigint", "byte", "short", "integer", "long")
# string encodings, most specific first: (kind, Spark format, regex the values must match)
ENCODINGS = [
    ("timestamp", "yyyy-MM-dd HH:mm:ss.SSS", r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}$"),
    ("timestamp", "yyyy-MM-dd HH:mm:ss", r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"),
    ("timestamp", "yyyy-MM-dd'T'HH:mm:ss", r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$"),
    ("date", "yyyy-MM-dd", r"^\d{4}-\d{2}-\d{2}$"),
    ("date", "yyyyMMdd", r"^(?!\d{4}02(30|31))(?!\d{4}(0[469]|11)31)(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])$"),
    ("date", "yyyy/MM/dd", r"^\d{4}/\d{2}/\d{2}$"),
    ("date", "MM/dd/yyyy", r"^\d{2}/\d{2}/\d{4}$"),
    ("time", "HHmmss", r"^([01]\d|2[0-3])[0-5]\d[0-5]\d$"),
    ("time", "HH:mm:ss", r"^([01]\d|2[0-3]):[0-5]\d:[0-5]\d$"),
    ("number", None, r"^-?\d+(\.\d+)?$"),
]
TIME_NAME = re.compile(r"(tm|time|hhmmss)")
ID_NAME = re.compile(r"(^id$|_id$|[a-z]{3}id$|_ids$|_key$|_no$|_num$|_nbr$|_number$|ods$|acct|^key$|hash)")


def type_family(data_type: str) -> str:
    t = data_type.lower()
    if t in _INTEGER:
        return "integer"
    if t.startswith("decimal") or t.startswith("numeric"):
        return "decimal"
    if t in ("float", "double", "real"):
        return "float"
    if t == "string" or t.startswith("char") or t.startswith("varchar"):
        return "string"
    if t == "boolean":
        return "boolean"
    if t == "date":
        return "date"
    if t.startswith("timestamp"):
        return "timestamp"
    return "other"


def jsonable(v):
    if isinstance(v, _decimal.Decimal):
        return int(v) if v == v.to_integral_value() else float(v)
    if isinstance(v, (_dt.datetime, _dt.date)):
        return v.isoformat()
    return v


def is_id_like(name: str, unique_ratio: float) -> bool:
    """IDs (accounts, merchants, keys, hashes) grow with the data; codes don't."""
    return bool(ID_NAME.search(name.lower())) or unique_ratio >= 0.5


@dataclass
class ColumnProfile:
    name: str
    data_type: str
    family: str
    nullable: bool
    comment: str | None
    null_fraction: float
    distinct: int
    unique_ratio: float
    min: object = None
    max: object = None
    roles: list = field(default_factory=list)  # workload roles from the SQL (join, partition, ...)
    encoded: dict | None = None  # string holding another type: {kind, format, scale, min, max}
    top_values: list | None = None  # [[value as string, count], ...]
    top_coverage: float | None = None
    literal_counts: dict | None = None  # query literal -> rows in the source
    quantiles: list | None = None  # [[p, x]]: numbers; dates as unix days; timestamps as unix seconds
    tod_quantiles: list | None = None  # timestamps: seconds within the day
    day_weights: list | None = None  # [[yyyy-mm-dd, rows]] when the column spans few distinct days
    lengths: list | None = None  # [[length, share]] for strings
    alphabets: list | None = None  # per position: the characters seen there (strings that aren't codes)
    fanout: dict | None = None  # rows-per-value shape for high-cardinality repeating columns

    @property
    def kind(self) -> str:
        """family, with string-encoded types resolved: 'date', 'timestamp', 'time', 'number', ..."""
        return self.encoded["kind"] if self.encoded else self.family

    @property
    def id_like(self) -> bool:
        return is_id_like(self.name, self.unique_ratio)


@dataclass
class TableProfile:
    name: str
    short_name: str
    rows: int
    comment: str | None
    columns: list = field(default_factory=list)
    partition_columns: list = field(default_factory=list)
    clustering_columns: list = field(default_factory=list)
    sampled: bool = False  # distributions come from a random row sample
    row_fraction: float = 1.0  # share of rows in that sample
    key_fraction: float = 1.0  # share of key values in the key-hash sample (skew, links)
    fingerprint: str | None = None  # settings + SQL facts the profile was made with (resume)

    def column(self, name: str) -> ColumnProfile:
        return next(c for c in self.columns if c.name.lower() == name.lower())

    def has(self, name: str) -> bool:
        return any(c.name.lower() == name.lower() for c in self.columns)

    def to_dict(self) -> dict:
        return asdict(self)


AGG_BATCH = 40  # columns per aggregate query: one huge aggregate over hundreds of columns is slow to compile


def batched_agg(df, groups: list[list]) -> dict:
    """Run aggregate expressions grouped per column in batches of AGG_BATCH columns; merged row as a dict."""
    out = {}
    for i in range(0, len(groups), AGG_BATCH):
        exprs = [e for g in groups[i:i + AGG_BATCH] for e in g]
        if exprs:
            out.update(df.agg(*exprs).first().asDict())
    return out


def batched_select(df, groups: list[list]) -> dict:
    """Like batched_agg for SQL select-expression strings (each must aggregate)."""
    out = {}
    for i in range(0, len(groups), AGG_BATCH):
        exprs = [e for g in groups[i:i + AGG_BATCH] for e in g]
        if exprs:
            out.update(df.selectExpr(*exprs).first().asDict())
    return out


def describe_table(spark, fqn: str):
    """(columns [(name, type, comment)], table comment, partition columns, clustering columns)."""
    cols, parts, section = [], [], "columns"
    for row in spark.sql(f"DESCRIBE TABLE {fqn}").collect():
        name = (row["col_name"] or "").strip()
        if name.startswith("#"):
            section = "partitions" if "Partition" in name else ("other" if section != "partitions" else section)
            continue
        if not name:
            section = "other" if section == "columns" else section
            continue
        if section == "columns":
            cols.append((name, row["data_type"], row["comment"] or None))
        elif section == "partitions":
            parts.append(name)
    comment, provider = None, ""
    for row in spark.sql(f"DESCRIBE TABLE EXTENDED {fqn}").collect():
        key = (row["col_name"] or "").strip()
        if key == "Comment":
            comment = row["data_type"] or None
        elif key == "Provider":
            provider = (row["data_type"] or "").lower()
    clustering = []
    if provider == "delta":
        try:
            detail = spark.sql(f"DESCRIBE DETAIL {fqn}").first().asDict()
            clustering = list(detail.get("clusteringColumns") or [])
            parts = parts or list(detail.get("partitionColumns") or [])
        except Exception:  # noqa: BLE001
            pass
    return cols, comment, parts, clustering


def parsed_sql(col: str, cp: ColumnProfile) -> str | None:
    """SQL turning the column into a number: unix days, unix seconds, seconds of day, or the numeric value."""
    c = f"`{col}`"
    kind, enc = cp.kind, cp.encoded
    if enc:
        # try_*: a string column is called a date / number when 99% of a sample parses; the odd bad value in the
        # full table ('00000000', '', 'N/A') becomes NULL instead of failing the query under ANSI mode
        fmt = enc.get("format")
        if kind == "date":
            return f"unix_date(to_date(try_to_timestamp({c}, '{fmt}')))"
        if kind == "timestamp":
            return f"unix_seconds(try_to_timestamp({c}, '{fmt}'))"
        if kind == "time":
            d = f"regexp_replace({c}, ':', '')"
            return (f"(try_cast(substr({d}, 1, 2) as int) * 3600 + try_cast(substr({d}, 3, 2) as int) * 60 + "
                    f"try_cast(substr({d}, 5, 2) as int))")
        if kind == "number":
            return f"try_cast({c} as double)"
    if kind in ("integer", "decimal", "float"):
        return f"cast({c} as double)"
    if kind == "date":
        return f"unix_date({c})"
    if kind == "timestamp":
        return f"unix_seconds(cast({c} as timestamp))"
    return None


def day_sql(col: str, cp: ColumnProfile) -> str | None:
    """SQL giving the calendar day (yyyy-mm-dd) of a date / timestamp / string-encoded column."""
    kind, enc, c = cp.kind, cp.encoded, f"`{col}`"
    if kind == "date":
        return (f"cast(to_date(try_to_timestamp({c}, '{enc['format']}')) as string)" if enc
                else f"cast({c} as string)")
    if kind == "timestamp":
        return (f"cast(to_date(try_to_timestamp({c}, '{enc['format']}')) as string)" if enc
                else f"cast(to_date({c}) as string)")
    return None


def key_expr(cp, alias: str = "k"):
    """Comparable form of a key column: integers/decimals with no scale as decimal(38,0), otherwise as string.

    Linked columns of different types (an INT id joined to a STRING id) compare - and hash - the same way.
    """
    name = f"`{cp.name}`"
    if cp.family == "integer" or (cp.family == "decimal" and re.search(r",\s*0\)", cp.data_type)):
        return F.expr(f"try_cast({name} as decimal(38,0))").alias(alias)
    if cp.encoded and cp.encoded["kind"] == "number" and not cp.encoded.get("scale"):
        return F.expr(f"try_cast({name} as decimal(38,0))").alias(alias)
    return F.col(name).cast("string").alias(alias)


def key_fraction(rows: int, key_sample_rows: int | None) -> float:
    """Share of key values to keep so about ``key_sample_rows`` rows remain (1.0: every key)."""
    if not key_sample_rows or rows <= key_sample_rows:
        return 1.0
    return max(key_sample_rows / rows, 1 / KEY_HASH_BUCKETS)


def in_key_sample(k, fraction: float):
    """True for the keys inside a key-hash sample of ``fraction``; the same keys on every table (nested by size)."""
    return F.pmod(F.xxhash64(k), F.lit(KEY_HASH_BUCKETS)) < F.lit(max(1, int(round(fraction * KEY_HASH_BUCKETS))))


def key_frame(df, cp, fraction: float = 1.0, alias: str = "k"):
    """The non-null keys of ``cp`` (normalised by :func:`key_expr`), restricted to a key-hash sample."""
    kdf = df.select(key_expr(cp, alias)).where(F.col(alias).isNotNull())
    return kdf.where(in_key_sample(F.col(alias), fraction)) if fraction < 1 else kdf


def fanout_from_histogram(hist, n_values: int) -> dict:
    """:func:`fanout` from ``[(rows per value, number of values)]``: the same metrics, computed locally."""
    hist = sorted((int(n), int(m)) for n, m in hist if n and m)
    total, used = sum(n * m for n, m in hist), sum(m for _, m in hist)
    if not total:
        return {"top1": 0.0, "top10pct": 0.0, "gini": 0.0, "coverage": 0.0, "mean": 0.0}
    n_values = max(n_values, used)
    k = max(1, -(-n_values // 10))
    top10, left = 0, k
    for n, m in reversed(hist):  # the busiest k values
        take = min(m, left)
        top10, left = top10 + n * take, left - take
        if not left:
            break
    s, rank = 0, n_values - used  # ascending ranks; values that never occur take the lowest ranks
    for n, m in hist:  # m values with n rows each occupy ranks rank+1 .. rank+m
        s += n * (m * rank + m * (m + 1) // 2)
        rank += m
    gini = (2 * s) / (n_values * total) - (n_values + 1) / n_values
    return {"top1": round(hist[-1][0] / total, 6), "top10pct": round(top10 / total, 6), "gini": round(gini, 6),
            "coverage": round(used / n_values, 6), "mean": round(total / max(used, 1), 4)}


def fanout(counts_df, n_values: int) -> dict:
    """Shape of rows-per-value over ``n_values`` possible values (counts_df has column ``n``).

    top1 / top10pct: share of rows on the busiest value / the busiest 10% of values; gini over all values
    (zeros included); coverage: share of values that occur at all; mean: rows per occurring value.

    One small aggregate - how many values have n rows - is collected, so there is no global sort and no
    single-partition window however many distinct values there are.
    """
    hist = counts_df.groupBy("n").count().collect()
    return fanout_from_histogram([(r["n"], r["count"]) for r in hist], n_values)


def profile_table(spark, fqn: str, top_n: int = 100, query_literals: dict | None = None,
                  critical: dict | None = None, casts: dict | None = None, sample_rows: int | None = SAMPLE_ROWS,
                  full_value_columns: set | None = None, log=None,
                  key_sample_rows: int | None = KEY_SAMPLE_ROWS) -> TableProfile:
    """Profile every column of ``fqn``.

    ``critical`` maps workload-critical columns to their SQL roles (they get skew stats). ``sample_rows`` /
    ``key_sample_rows`` bound the row sample and the key-hash sample (see the module doc; 0 = full table).
    ``query_literals`` (column -> values) are always counted. ``casts`` (column -> {'DOUBLE', 'DATE'}) are the SQL's
    hints about what a string column really holds. ``full_value_columns`` keep their complete value list (up to
    CATEGORICAL_PROFILE_MAX) instead of the top-N, e.g. columns that must reference a copied lookup table.
    """
    def low(d):
        return {k.lower(): v for k, v in (d or {}).items()}

    query_literals, critical, casts = low(query_literals), low(critical), low(casts)
    full_value_columns = {c.lower() for c in (full_value_columns or set())}
    described, comment, part_cols, cluster_cols = describe_table(spark, fqn)
    df = spark.table(fqn)
    nullability = {f.name.lower(): f.nullable for f in df.schema.fields}
    # DESCRIBE truncates long complex types ('... 2 more fields'); PySpark's schema gives the full type text
    full_types = {f.name.lower(): f.dataType.simpleString() for f in df.schema.fields}
    described = [(n, full_types.get(n.lower(), t) if type_family(t) == "other" else t, c) for n, t, c in described]
    rows = df.count()
    sampled = bool(sample_rows) and rows > sample_rows
    row_fraction = sample_rows / rows if sampled else 1.0
    sdf = df.sample(fraction=row_fraction, seed=17) if sampled else df
    sample_factor = rows / max(sdf.count(), 1) if sampled else 1.0
    kf = key_fraction(rows, key_sample_rows)
    # string encodings and formats: a random ~1M rows (a plain limit() would read only the first files)
    fs = df.sample(fraction=FORMAT_SAMPLE_ROWS / rows, seed=19) if rows > FORMAT_SAMPLE_ROWS else df
    profile = TableProfile(name=fqn, short_name=fqn.split(".")[-1].lower(), rows=rows, comment=comment,
                           partition_columns=part_cols, clustering_columns=cluster_cols, sampled=sampled,
                           row_fraction=round(row_fraction, 9), key_fraction=round(kf, 9))
    if log and (sampled or kf < 1):
        log(f"{fqn}: {rows:,} rows; distributions on a {row_fraction:.4%} row sample, key skew on a {kf:.4%} key "
            f"sample")
    if not rows:
        for name, dtype, cc in described:
            profile.columns.append(ColumnProfile(name, dtype, type_family(dtype), True, cc, 0.0, 0, 0.0))
        return profile

    if log:
        log(f"{fqn}: pass 1")
    # ---- pass 1: counts, distinct, min/max
    fams = {n: type_family(t) for n, t, _ in described}
    groups = []
    for name, _, _ in described:
        c = F.col(f"`{name}`")
        g = [F.count(c).alias(f"nn__{name}")]
        if fams[name] != "other":
            # approximate (HLL, DISTINCT_RSD); exact counts would replicate every row once per column. Made exact
            # below wherever the frequency pass sees every value, and keys are re-checked exactly in relationships.
            g.append(F.approx_count_distinct(c, DISTINCT_RSD).alias(f"nd__{name}"))
        if fams[name] not in ("other", "boolean"):
            g += [F.min(c).alias(f"mn__{name}"), F.max(c).alias(f"mx__{name}")]
        groups.append(g)
    stats = batched_agg(df, groups)
    for name, dtype, cc in described:
        non_null = stats[f"nn__{name}"]
        distinct = int(stats.get(f"nd__{name}") or 0)
        profile.columns.append(ColumnProfile(
            name=name, data_type=dtype, family=fams[name], nullable=nullability.get(name.lower(), True),
            comment=cc, null_fraction=round(1 - non_null / rows, 6), distinct=distinct,
            unique_ratio=round(distinct / non_null, 6) if non_null else 0.0,
            min=jsonable(stats.get(f"mn__{name}")), max=jsonable(stats.get(f"mx__{name}")),
            roles=sorted(critical.get(name.lower(), []))))
    cols = {c.name: c for c in profile.columns}

    # ---- pass 1b: are the near-unique columns really unique? The distinct count is approximate (DISTINCT_RSD), and
    # "unique" decides between row-number keys and repeating values, so check it with one group-by for all of them.
    # Large tables are checked on two samples: the key-hash sample (every copy of a sampled value is in it, so values
    # that repeat a little show up) and the random row sample (a few values repeated many times, such as -1 or 0
    # placeholders, show up there).
    near = [c for c in profile.columns if c.family in ("integer", "decimal", "string") and c.null_fraction < 1
            and c.unique_ratio >= UNIQUE_CHECK_RATIO]
    if near:
        if log:
            log(f"{fqn}: pass 1b ({len(near)} near-unique columns)")

        def values(frame, src: int):
            return frame.select(F.explode(F.array(*[
                F.struct(F.lit(src).alias("s"), F.lit(i).alias("i"), F.col(f"`{c.name}`").cast("string").alias("v"))
                for i, c in enumerate(near)])).alias("x")).select("x.s", "x.i", "x.v").where("v is not null")

        if kf < 1:
            kv = values(df, 0).where(in_key_sample(F.col("v"), kf)).unionByName(values(sdf, 1))
        else:
            kv = values(df, 0)
        repeated = {r["i"] for r in kv.groupBy("s", "i", "v").count().where("count > 1").select("i").distinct()
                    .collect()}
        for i, c in enumerate(near):
            if i not in repeated:
                c.distinct = int(stats[f"nn__{c.name}"])  # unique: one value per non-null row, exactly
                c.unique_ratio = 1.0

    if log:
        log(f"{fqn}: pass 2")
    # ---- pass 2: what string columns really hold (on a sample of non-null values)
    # each value is tagged with the first (most specific) encoding it matches; one small group-by for all columns
    strings = [c for c in profile.columns if c.family == "string" and c.null_fraction < 1]
    if strings:
        tag = F.lit(-1)
        for i, (_, _, rx) in reversed(list(enumerate(ENCODINGS))):  # Column API: SQL literals would eat the \d
            tag = F.when(F.col("v").rlike(rx), i).otherwise(tag)
        kv = fs.select(F.explode(F.array(*[F.struct(F.lit(c.name).alias("k"), F.col(f"`{c.name}`").alias("v"))
                                           for c in strings])).alias("kv")).select("kv.k", "kv.v").where("v is not null")
        shares = {}
        for r in kv.select("k", tag.alias("e")).groupBy("k", "e").count().collect():
            shares.setdefault(r["k"], {})[r["e"]] = r["count"]
        numeric_like = {i for i, (kind, fmt, _) in enumerate(ENCODINGS)
                        if kind == "number" or fmt in ("yyyyMMdd", "HHmmss")}
        for c in strings:
            got = shares.get(c.name, {})
            total = sum(got.values()) or 1
            hints = {h.upper() for h in casts.get(c.name.lower(), set())}
            for i, (kind, fmt, _) in enumerate(ENCODINGS):
                share = got.get(i, 0) / total
                if kind == "number":
                    share = sum(got.get(j, 0) for j in numeric_like) / total
                if share < 0.99 or (kind == "time" and not TIME_NAME.search(c.name.lower())):
                    continue
                if kind == "number":
                    # numeric text that is really a code ('05', '6011') stays a code unless the SQL casts it
                    casted = any(h.startswith(("DOUBLE", "DECIMAL", "FLOAT", "INT", "BIGINT")) for h in hints)
                    if not casted and (c.unique_ratio <= 0.05 or c.distinct <= top_n):
                        break
                    scale = fs.agg(F.max(F.length(F.regexp_extract(f"`{c.name}`", r"\.(\d+)$", 1)))).first()[0]
                    c.encoded = {"kind": "number", "format": None, "scale": int(scale or 0)}
                else:
                    c.encoded = {"kind": kind, "format": fmt}
                break

    if log:
        log(f"{fqn}: pass 3")
    # ---- pass 3: quantiles, on the row sample (their end points are made exact in pass 5)
    probs = ", ".join(repr(p) for p in QUANTILE_PROBS)
    q_exprs = {c.name: parsed_sql(c.name, c) for c in profile.columns if c.null_fraction < 1}
    q_exprs = {k: v for k, v in q_exprs.items() if v}
    if q_exprs:
        groups = []
        for n, e in q_exprs.items():
            g = [f"percentile_approx({e}, array({probs}), 10000) as `q__{n}`"]
            if cols[n].kind == "timestamp":
                g.append(f"percentile_approx(pmod({e}, 86400), array({probs}), 10000) as `t__{n}`")
            groups.append(g)
        qrow = batched_select(sdf, groups)
        for n in q_exprs:
            cp = cols[n]
            if qrow.get(f"q__{n}"):
                cp.quantiles = [[p, float(x)] for p, x in zip(QUANTILE_PROBS, qrow[f"q__{n}"]) if x is not None]
            if qrow.get(f"t__{n}"):
                cp.tod_quantiles = [[p, float(x)] for p, x in zip(QUANTILE_PROBS, qrow[f"t__{n}"]) if x is not None]

    if log:
        log(f"{fqn}: pass 4")
    # ---- pass 4: frequencies, per-day histograms, string lengths, on the row sample - except join columns with few
    # values, whose complete value list (e.g. every code a lookup has) is counted on the full table
    def freq_items(c: ColumnProfile):
        items = []
        if c.family != "other" and c.distinct <= CATEGORICAL_PROFILE_MAX and c.null_fraction < 1 and \
                not (c.family == "string" and c.unique_ratio > 0.5 and not c.encoded):
            items.append((f"v__{c.name}", F.col(f"`{c.name}`").cast("string")))
        day = day_sql(c.name, c)
        if day and c.null_fraction < 1:
            items.append((f"d__{c.name}", F.expr(day)))
        if c.family == "string" and c.null_fraction < 1:
            items.append((f"l__{c.name}", F.length(F.col(f"`{c.name}`")).cast("string")))
        return items

    freq = {}
    exact = [c for c in profile.columns if not sampled or (c.name.lower() in full_value_columns
                                                           and c.distinct <= CATEGORICAL_PROFILE_MAX)]
    exact_names = {c.name for c in exact}
    groups = [(df, 1.0, exact), (sdf, sample_factor, [c for c in profile.columns if c.name not in exact_names])]
    for frame, factor, members in groups:
        items = [it for c in members for it in freq_items(c)]
        if not items:
            continue
        kv = frame.select(F.explode(F.array(*[F.struct(F.lit(k).alias("k"), e.alias("v")) for k, e in items]))
                          .alias("kv")).select("kv.k", "kv.v").where("v is not null")
        ranked = kv.groupBy("k", "v").count().withColumn(
            "r", F.row_number().over(Window.partitionBy("k").orderBy(F.desc("count"), "v")))
        keep = CATEGORICAL_PROFILE_MAX if full_value_columns else top_n
        for r in ranked.where((F.col("r") <= keep) | ~F.col("k").startswith("v__")).collect():
            freq.setdefault(r["k"], []).append([r["v"], int(round(r["count"] * factor))])
    for c in profile.columns:
        non_null = rows * (1 - c.null_fraction)
        vals = freq.get(f"v__{c.name}")
        if vals and len(vals) < (CATEGORICAL_PROFILE_MAX if full_value_columns else top_n):
            # every value was seen: exact on the full table; a sample can miss rare values, so there the values seen
            # are only a lower bound for the (approximate) count
            c.distinct = len(vals) if c.name in exact_names else max(c.distinct, len(vals))
            c.unique_ratio = round(c.distinct / non_null, 6) if non_null else 0.0
        if vals:
            limit = CATEGORICAL_PROFILE_MAX if c.name.lower() in full_value_columns else top_n
            vals.sort(key=lambda x: (-x[1], x[0]))
            c.top_values = vals[:limit]
            c.top_coverage = round(min(1.0, sum(n for _, n in c.top_values) / non_null), 6) if non_null else None
        days = freq.get(f"d__{c.name}")
        if days and len(days) <= MAX_DAYS:
            c.day_weights = sorted(days)
        lens = freq.get(f"l__{c.name}")
        if lens:
            tot = sum(n for _, n in lens) or 1
            c.lengths = sorted([[int(v), round(n / tot, 6)] for v, n in lens])

    if log:
        log(f"{fqn}: pass 5")
    # ---- pass 5, full table: query literals, and the exact ends of every quantile range (parsed min / max)
    lit_aggs, lit_keys = [], []
    for c in profile.columns:
        for lit in sorted(query_literals.get(c.name.lower(), set()), key=str):
            lit_aggs.append(F.sum(F.when(F.col(f"`{c.name}`").cast("string") == str(lit), 1).otherwise(0))
                            .alias(f"lit{len(lit_keys)}"))
            lit_keys.append((c, str(lit)))
    ends = {n: e for n, e in q_exprs.items() if cols[n].encoded or (sampled and cols[n].quantiles)}
    end_aggs = [[F.min(F.expr(e)).alias(f"emn__{n}"), F.max(F.expr(e)).alias(f"emx__{n}")] for n, e in ends.items()]
    if lit_aggs or end_aggs:
        lrow = batched_agg(df, [[a] for a in lit_aggs] + end_aggs)
        for i, (c, lit) in enumerate(lit_keys):
            c.literal_counts = {**(c.literal_counts or {}), lit: int(lrow[f"lit{i}"] or 0)}
        for n in ends:
            cp, lo, hi = cols[n], lrow.get(f"emn__{n}"), lrow.get(f"emx__{n}")
            if lo is None:
                continue
            lo, hi = float(lo), float(hi)
            if cp.encoded:
                cp.encoded["min"], cp.encoded["max"] = lo, hi
            if cp.quantiles:
                cp.quantiles[0], cp.quantiles[-1] = [cp.quantiles[0][0], lo], [cp.quantiles[-1][0], hi]

    if log:
        log(f"{fqn}: pass 6")
    # ---- pass 6: formats of high-cardinality strings (per-position character sets), one explode + group-by
    fmt_cols = [c for c in profile.columns if c.family == "string" and not c.encoded and c.null_fraction < 1
                and (c.top_coverage or 0) < 0.999 and c.lengths]
    if fmt_cols:
        kv = fs.select(F.explode(F.array(*[F.struct(F.lit(c.name).alias("k"), F.col(f"`{c.name}`").alias("v"))
                                           for c in fmt_cols])).alias("kv")).select("kv.k", "kv.v").where("v is not null")
        chars = kv.select("k", F.posexplode(F.split(F.substring("v", 1, MAX_POSITIONS), "")).alias("p", "ch")) \
            .where("ch != ''").distinct().groupBy("k", "p").agg(F.array_join(F.array_sort(F.collect_set("ch")), "").alias("a"))
        found = {}
        for r in chars.collect():
            found.setdefault(r["k"], {})[r["p"]] = r["a"]
        for c in fmt_cols:
            longest = min(max(n for n, _ in c.lengths), MAX_POSITIONS)
            c.alphabets = [found.get(c.name, {}).get(p, "") for p in range(longest)]

    if log:
        log(f"{fqn}: pass 7")
    # ---- pass 7: skew of high-cardinality repeating workload columns, on the key-hash sample (all rows of a sampled
    # key are kept, so rows-per-key is exact for those keys)
    for c in profile.columns:
        if c.roles and c.distinct > top_n and c.unique_ratio < 0.999 and c.null_fraction < 1:
            counts = key_frame(df, c, kf).groupBy("k").count().select(F.col("count").alias("n"))
            c.fanout = fanout(counts, max(1, round(c.distinct * kf)))
    return profile


def literal_columns(literals: dict, table: str) -> dict[str, set]:
    """Column -> query literal values for one table, from ``QueryFacts.literals``."""
    return {c: v for (t, c), v in literals.items() if t == table}


def table_items(mapping: dict, table: str) -> dict:
    """Column -> value for one table, from a dict keyed by (table, column)."""
    return {c: v for (t, c), v in mapping.items() if t == table}
