# How to release a new version

*A cheat sheet.*

* On `devel` branch:
  * `git pull` to make sure everything is in sync with remote origin.
  * Change a version in `d3m/__init__.py` to the new version, e.g., `2019.2.12`.
  * Change `vNEXT` in `HISTORY.md` to the to-be-released version, with `v` prefix.
  * Commit with message `Bumping version for release.`
  * `git push`
  * Wait for CI to run tests successfully.
* On `master` branch:
  * `git pull` to make sure everything is in sync with remote origin.
  * Merge `devel` into `master` branch: `git merge devel`
  * `git push`
  * Wait for CI to run tests successfully.
  * Release a package to PyPi:
    * `rm -rf dist/`
    * `python setup.py sdist`
    * `twine upload dist/*`
  * Tag with version prefixed with `v`, e.g., for version `2017.9.20`: `git tag v2017.9.20`
  * `git push` & `git push --tags`
* On `devel` branch:
  * `git merge master` to make sure `devel` is always on top of `master`.
  * Change a version in `d3m/__init__.py` to `devel`.
  * Add a new empty `vNEXT` version on top of `HISTORY.md`.
  * Commit with message `Version bump for development.`
  * `git push`
* After a release:
  * Create a new [`core` and `primitives` Docker images](https://gitlab.com/datadrivendiscovery/images) for the release.
  * Add new release to the [primitives index repository](https://gitlab.com/datadrivendiscovery/primitives/blob/master/HOW_TO_MANAGE.md).

If there is a need for a patch version to fix a released version on the same day,
use `.postX` prefix, like `2017.9.20.post0`. If more than a day has passed, just
use the new day's version.
