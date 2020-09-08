#!/bin/bash -e

if git rev-list --objects --all \
| git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
| sed -n 's/^blob //p' \
| awk '$2 >= 100*(2^10)' \
| awk '{print $3}' \
| egrep -v '(^|/).gitattributes$' ; then
  echo "Repository contains committed objects larger than 100 KB."
  exit 1
fi

if git lfs ls-files --name-only | xargs -r stat -c '%s %n' | awk '$1 < 100*(2^10)' | awk '{print $2}' | grep . ; then
  echo "Repository contains LFS objects smaller than 100 KB."
  exit 1
fi

if git lfs ls-files --name-only | xargs -r stat -c '%s %n' | awk '$1 >= 2*(2^30)' | awk '{print $2}' | grep . ; then
  echo "Repository contains LFS objects not smaller than 2 GB."
  exit 1
fi
