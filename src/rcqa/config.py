"""Names shared by every notebook, so a table is never spelled two ways."""
CATALOG = "workspace"   # Free Edition default catalog
SCHEMA = "rcqa"
VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/raw"

SYNTHEA_TABLES = ["patients", "encounters", "conditions", "medications", "observations",
                  "procedures", "allergies", "careplans", "immunizations"]


def table(name: str) -> str:
    """table('bronze_patients') -> 'workspace.rcqa.bronze_patients'"""
    return f"{CATALOG}.{SCHEMA}.{name}"
