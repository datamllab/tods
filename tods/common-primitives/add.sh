#!/bin/bash -e

# Assumption is that this repository is cloned into "common-primitives" directory
# which is a sibling of "d3m-primitives" directory with D3M public primitives.

D3M_VERSION="$(python3 -c 'import d3m; print(d3m.__version__)')"

for PRIMITIVE_SUFFIX in $(./list_primitives.py --suffix); do
  echo "$PRIMITIVE_SUFFIX"
  python3 -m d3m index describe -i 4 "d3m.primitives.$PRIMITIVE_SUFFIX" > primitive.json
  pushd ../d3m-primitives > /dev/null
  ./add.py ../common-primitives/primitive.json
  popd > /dev/null
  if [[ -e "pipelines/$PRIMITIVE_SUFFIX" ]]; then
    PRIMITIVE_PATH="$(echo ../d3m-primitives/v$D3M_VERSION/common-primitives/d3m.primitives.$PRIMITIVE_SUFFIX/*)"
    mkdir -p "$PRIMITIVE_PATH/pipelines"
    find pipelines/$PRIMITIVE_SUFFIX/ \( -name '*.json' -or -name '*.yaml' -or -name '*.yml' -or -name '*.json.gz' -or -name '*.yaml.gz' -or -name '*.yml.gz' \) -exec cp '{}' "$PRIMITIVE_PATH/pipelines" ';'
  fi
  if [[ -e "pipeline_runs/$PRIMITIVE_SUFFIX" ]]; then
    PRIMITIVE_PATH="$(echo ../d3m-primitives/v$D3M_VERSION/common-primitives/d3m.primitives.$PRIMITIVE_SUFFIX/*)"
    mkdir -p "$PRIMITIVE_PATH/pipeline_runs"
    find pipeline_runs/$PRIMITIVE_SUFFIX/ \( -name '*.yml.gz' -or -name '*.yaml.gz' \) -exec cp '{}' "$PRIMITIVE_PATH/pipeline_runs" ';'
  fi
done
