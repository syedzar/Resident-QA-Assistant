import os
from pathlib import Path

import pytest

# Local Spark needs Java 17+. Use JAVA_HOME if set, otherwise the default Temurin install path on Windows.
_TEMURIN = Path(r"C:\Program Files\Eclipse Adoptium")
if "JAVA_HOME" not in os.environ and _TEMURIN.exists():
    jdks = sorted(_TEMURIN.glob("jdk-17*"))
    if jdks:
        os.environ["JAVA_HOME"] = str(jdks[-1])


@pytest.fixture(scope="session")
def spark():
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    s = (SparkSession.builder.master("local[1]").appName("rcqa-tests")
         .config("spark.sql.shuffle.partitions", "1")
         .config("spark.ui.enabled", "false")
         .config("spark.sql.session.timeZone", "UTC")
         # Build test DataFrames through Arrow. On Windows, local PySpark's Python worker
         # crashes when parallelizing plain Python lists; the Arrow path needs no worker.
         .config("spark.sql.execution.arrow.pyspark.enabled", "true")
         .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "false")
         .getOrCreate())
    yield s
    s.stop()


@pytest.fixture
def make_df(spark):
    """make_df(rows, columns) -> Spark DataFrame of strings, like a bronze table."""
    import pandas as pd
    from pyspark.sql.types import StringType, StructField, StructType

    def _make(rows, columns):
        schema = StructType([StructField(c, StringType()) for c in columns])
        return spark.createDataFrame(pd.DataFrame(rows, columns=columns, dtype=object), schema=schema)

    return _make
