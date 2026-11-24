"""Profile existing tables, derive dbldatagen-core plans from them, and generate scaled synthetic copies.

Pipeline (one module per step):

* ``profiler``       - per-table / per-column statistics (aggregates only, no raw rows are collected)
* ``sqlparse_util``  - join keys, filter literals and date ranges mined from the customer's SQL
* ``relationships``  - PK/FK identification (UC constraints, queries, names + data) and within-table dependencies
* ``derive``         - turns profiles + relationships into a reviewable YAML config wrapping a ``DataGenPlan``
* ``generate``       - materialises the config into target tables
* ``validate``       - compares source and target tables, and re-runs the SQL against both
"""

__all__ = ["profiler", "sqlparse_util", "relationships", "derive", "generate", "validate"]
