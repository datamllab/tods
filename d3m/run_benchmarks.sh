#!/bin/bash -e

if ! git remote get-url upstream > /dev/null 2>&1 ; then
  git remote add upstream https://gitlab.com/datadrivendiscovery/d3m.git
fi
git fetch upstream

asv machine --yes --config tests/asv.conf.json

ASV_OUTPUT=$(asv continuous upstream/devel HEAD -s -f 1.1 -e --config tests/asv.conf.json)
echo "$ASV_OUTPUT"

if echo "$ASV_OUTPUT" | egrep -q "(SOME BENCHMARKS HAVE CHANGED SIGNIFICANTLY)|( failed$)" ; then
  echo "Benchmarks have errors."
  exit 1
else
  echo "Benchmarks ran without errors."
fi
