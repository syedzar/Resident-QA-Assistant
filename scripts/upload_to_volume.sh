#!/usr/bin/env bash
# Upload raw inputs to the Unity Catalog Volume that the bronze notebook reads.
#   /Volumes/workspace/rcqa/raw/synthea/*.csv.gz   9 clinical tables (gzipped)
#   /Volumes/workspace/rcqa/raw/docs/               guideline PDFs + e-Laws .txt
#   /Volumes/workspace/rcqa/raw/drug_labels/        openFDA label JSON
#   /Volumes/workspace/rcqa/raw/sources/sources.csv citation metadata
#
# Needs the Databricks CLI, logged in once with:
#   databricks auth login --host https://<workspace>.cloud.databricks.com --profile rcqa
set -euo pipefail
cd "$(dirname "$0")/.."

PROFILE="${DATABRICKS_PROFILE:-rcqa}"
DBX="${DBX:-databricks}"
VOL="dbfs:/Volumes/workspace/rcqa/raw"
TABLES="patients conditions medications encounters observations procedures allergies careplans immunizations"

# Synthea writes ~2.9 GB, mostly billing tables we don't use. Gzip only the clinical ones.
mkdir -p data/raw/upload/synthea
for t in $TABLES; do
  src="data/raw/synthea/csv/$t.csv"; dst="data/raw/upload/synthea/$t.csv.gz"
  [ "$dst" -nt "$src" ] || gzip -c "$src" > "$dst"
done

"$DBX" fs cp -r --overwrite -p "$PROFILE" data/raw/upload/synthea "$VOL/synthea"
mkdir -p data/raw/upload/docs
cp data/raw/docs/*.pdf data/raw/docs/*.txt data/raw/upload/docs/   # .doc originals stay local
"$DBX" fs cp -r --overwrite -p "$PROFILE" data/raw/upload/docs "$VOL/docs"
"$DBX" fs cp -r --overwrite -p "$PROFILE" data/raw/drug_labels "$VOL/drug_labels"
"$DBX" fs mkdir -p "$PROFILE" "$VOL/sources"
"$DBX" fs cp --overwrite -p "$PROFILE" data/sources/sources.csv "$VOL/sources/sources.csv"
"$DBX" fs ls -p "$PROFILE" "$VOL"
