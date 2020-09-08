#!/bin/bash -e

# This requires git LFS 2.9.0 or newer.

find * -type f -size +100k -exec git lfs track --filename '{}' +
