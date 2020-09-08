#!/bin/bash

alias python="python3"

# check if we are on a deployment container or not.
if [ -d "/user_dev" ]; then
  cd  /user_dev
  echo "Running on deployment"
else
  echo "Running on testing"
fi


# check output_dir
if [[ -z "$D3MOUTPUTDIR" ]]; then
  D3MOUTPUTDIR="$(pwd)/output_dir"
  mkdir -p "$D3MOUTPUTDIR"
else
  D3MOUTPUTDIR="$D3MOUTPUTDIR"
fi

# check if time is set, otherwise we use 1 min
if [[ -z "$D3MTIMEOUT" ]]; then
    D3MTIMEOUT="60" # 10 gb
  else
    D3MTIMEOUT="$D3MTIMEOUT"
fi

# execute d3m server.
case $D3MRUN in
 "standalone")
    echo "Executing TAMU TA2 Standalone"
    echo "No standalone supported yet"
    ;;
  *)
    echo "Executing TAMU TA2"
    python3 -m axolotl.d3m_grpc.server
    ;;
esac
