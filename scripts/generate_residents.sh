#!/usr/bin/env bash
# Generate ~1,000 synthetic older-adult residents with Synthea (CSV only).
# Fixed seeds make the population reproducible: same command -> same patients.
#
# Requires Java 17+ and tools/synthea-with-dependencies.jar, from:
#   https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar
set -euo pipefail
cd "$(dirname "$0")/.."

JAVA="${JAVA:-java}"
OUT=data/raw/synthea

"$JAVA" -jar tools/synthea-with-dependencies.jar \
  -p 1000 -a 65-95 -s 42 -cs 42 \
  --exporter.baseDirectory="$OUT" \
  --exporter.csv.export=true \
  --exporter.fhir.export=false \
  --exporter.hospital.fhir.export=false \
  --exporter.practitioner.fhir.export=false \
  --exporter.years_of_history=10 \
  Massachusetts
