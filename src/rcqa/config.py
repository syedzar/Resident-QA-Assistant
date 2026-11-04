"""Names shared by every notebook, so a table is never spelled two ways."""
CATALOG = "workspace"   # Free Edition default catalog
SCHEMA = "rcqa"
VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/raw"

# Salt for the resident_id ordering. Fine in code for synthetic data; with real data
# this would live in a Databricks secret scope so ids can't be recomputed from UUIDs.
RESIDENT_ID_SALT = "rcqa-v1"

SYNTHEA_TABLES = ["patients", "encounters", "conditions", "medications", "observations",
                  "procedures", "allergies", "careplans", "immunizations"]


def table(name: str) -> str:
    """table('bronze_patients') -> 'workspace.rcqa.bronze_patients'"""
    return f"{CATALOG}.{SCHEMA}.{name}"
