#!/bin/bash

mkdir -p results

overall_result="0"

while IFS= read -r pipeline_run_file; do
  pipeline_run_name="$(dirname "$pipeline_run_file")/$(basename -s .yml.gz "$(basename -s .yaml.gz "$pipeline_run_file")")"
  primitive_name="$(basename "$(dirname "$pipeline_run_file")")"

  if [[ -L "$pipeline_run_file" ]]; then
    echo ">>> Skipping '$pipeline_run_file'."
    continue
  else
    mkdir -p "results/$pipeline_run_name"
  fi

  pipelines_path="pipelines/$primitive_name"

  if [[ ! -d "$pipelines_path" ]]; then
    echo ">>> ERROR: Could not find pipelines for '$pipeline_run_file'."
    overall_result="1"
    continue
  fi

  echo ">>> Running '$pipeline_run_file'."
  python3 -m d3m --pipelines-path "$pipelines_path" \
    runtime \
    --datasets /data/datasets --volumes /data/static_files \
    fit-score --input-run "$pipeline_run_file" \
    --output "results/$pipeline_run_name/predictions.csv" \
    --scores "results/$pipeline_run_name/scores.csv" \
    --output-run "results/$pipeline_run_name/pipeline_runs.yaml"
  result="$?"

  if [[ "$result" -eq 0 ]]; then
    echo ">>> SUCCESS ($pipeline_run_file)"
  else
    echo ">>> ERROR ($pipeline_run_file)"
    overall_result="1"
  fi
done < <(find pipeline_runs -name '*.yml.gz' -or -name '*.yaml.gz')

exit "$overall_result"
