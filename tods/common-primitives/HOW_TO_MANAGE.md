# How to publish primitive annotations

As contributors add or update their primitives they might want to publish
primitive annotations for added primitives. When doing this it is important
to republish also all other primitive annotations already published from this
package. This is because only one version of the package can be installed at
a time and all primitive annotations have to point to the same package in
their `installation` metadata.

Steps to publish primitive annotations:
* Operate in a virtual env with the following installed:
  * Target core package installed.
  * [Test primitives](https://gitlab.com/datadrivendiscovery/tests-data/tree/master/primitives)
    with the same version of primitives which are currently published in `primitives`
    repository. Remember to install them in `-e` editable mode.
* Update `HISTORY.md` for `vNEXT` release with information about primitives
  added or updated. If there was no package release since they were updated last,
  do not duplicate entries but just update any existing entries for those primitives
  instead, so that once released it is clear what has changed in a release as a whole.
* Make sure tests for primitives being published (primitives added, updated,
  and primitives previously published which should be now republished) pass.
* Update `entry_points.ini` and add new primitives. Leave active
  only those entries for primitives being (re)published and comment out all others.
  * If this is the first time primitives are published after a release of a new `d3m`
    core package, leave active only those which were updated to work with
    the new `d3m` core package. Leave to others to update, verify, and publish
    other common primitives.
* In clone of `primitives` repository prepare a branch of the up-to-date `master` branch
  to add/update primitive annotations. If existing annotations for common primitives
  are already there the best is to first remove them to make sure annotations for
  removed primitives do not stay around. We will re-add all primitives in the next step.
* Run `add.sh` in root of this package, which will add primitive annotations
  to `primitives`. See instructions in the script for more information.
* Verify changes in the `primitives`, add and commit files to git.
* Publish a branch in `primitives` and make a merge request.

# How to release a new version

A new version is always released from `master` branch against a stable release
of `d3m` core package. A new version should be released when there are major
changes to the package (many new primitives added, larger breaking changes).
Sync up with other developers of the repo to suggest a release, or do a release.

* On `master` branch:
  * Make sure `HISTORY.md` file is updated with all changes since the last release.
  * Change a version in `common_primitives/__init__.py` to the to-be-released version, without `v` prefix.
  * Change `vNEXT` in `HISTORY.md` to the to-be-released version, with `v` prefix.
  * Commit with message `Bumping version for release.`
  * `git push`
  * Wait for CI to run tests successfully.
  * Tag with version prefixed with `v`, e.g., for version `0.2.0`: `git tag v0.2.0`
  * `git push` & `git push --tags`
  * Change a version in `common_primitives/__init__.py` back to `devel` string.
  * Add a new empty `vNEXT` version on top of `HISTORY.md`.
  * Commit with message `Version bump for development.`
  * `git push`  
* On `devel` branch:
  * Merge `master` into `devel` branch: `git merge master`
  * Update the branch according to the section below.
  * `git push`

# How to update `master` branch after a release of new `d3m` core package

Hopefully, `devel` branch already contains code which works against the released
`d3m` core package. So merge `devel` branch into `master` branch and update
files according to the following section.

# Keeping `master` and `devel` branches in sync

Because `master` and `devel` branches mostly contain the same code,
just made against different version of `d3m` core package, it is common
to merge branches into each other as needed to keep them in sync.
When doing so, the following are files which are specific to branches:

* `.gitlab-ci.yml` has a `DEPENDENCY_REF` environment variable which
  has to point to `master` on `master` branch of this repository,
  and `devel` on `devel` branch of this repository.

# How to add an example pipeline

Every common primitive (except those used in non-standard pipelines, like splitting primitives)
should have at least one example pipeline and associated pipeline run.

Add example pipelines into a corresponding sub-directory based on primitive's suffix into `pipelines`
directory in the repository. If a pipeline uses multiple common primitives, add it for only one
primitive and create symbolic links for other primitives.

Create a `fit-score` pipeline run as [described in primitives index repository](https://gitlab.com/datadrivendiscovery/primitives#adding-a-primitive).
Compress it with `gzip` and store it under `pipeline_runs` directory in the repository.
Similarly, add it only for one primitive and create symbolic links for others, if pipeline run
corresponds to a pipeline with multiple common primitives.

Use `git-add.sh` script to assure all files larger than 100 KB are added as git LFS files to
the repository.
