#!/usr/bin/env bash
# Create or update the Databricks job from databricks/jobs/rcqa_pipeline.json (idempotent).
#   DATABRICKS_PROFILE=rcqa scripts/deploy_job.sh          # laptop (CLI OAuth login)
#   scripts/deploy_job.sh                                   # CI (DATABRICKS_HOST + DATABRICKS_TOKEN env)
set -euo pipefail
export MSYS_NO_PATHCONV=1
DBX=${DBX:-databricks}
PROFILE_ARGS=()
[[ -n "${DATABRICKS_PROFILE:-}" ]] && PROFILE_ARGS=(-p "$DATABRICKS_PROFILE")
SPEC=databricks/jobs/rcqa_pipeline.json
NAME=$(python -c "import json;print(json.load(open('$SPEC'))['name'])")
JOB_ID=$("$DBX" jobs list "${PROFILE_ARGS[@]}" -o json | python -c "
import sys, json
d = json.load(sys.stdin); d = d if isinstance(d, list) else d.get('jobs', [])
print(next((str(j['job_id']) for j in d if j['settings']['name'] == '$NAME'), ''))")
if [[ -z "$JOB_ID" ]]; then
  "$DBX" jobs create "${PROFILE_ARGS[@]}" --json @"$SPEC"
else
  python -c "import json;print(json.dumps({'job_id': $JOB_ID, 'new_settings': json.load(open('$SPEC'))}))" > /tmp/rcqa_reset.json
  "$DBX" jobs reset "${PROFILE_ARGS[@]}" --json @/tmp/rcqa_reset.json
  echo "updated job $JOB_ID"
fi
