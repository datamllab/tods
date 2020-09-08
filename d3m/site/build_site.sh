#!/bin/bash -e

# Builds sites for schemas. For each tag and `devel` branch a separate site is built.

deploy () {
  if [ ! -d site ] || [ ! -e site/package.json ]
  then
    return 0
  fi

  cd site
  npm install
  make
  cd ..

  # Copying results into output directory "public".
  cp -a site/static public/$1
  rm -f public/$1/schemas
  cp -a d3m/metadata/schemas public/$1/schemas

  # Cleaning.
  cd site
  make clean
  rm -fr node_modules
  cd ..

  # Reverting changes, "package-lock.json" might be changed.
  git checkout -- .
}

rm -rf public
mkdir public

git checkout devel
cp -a d3m/metadata/schemas public/schemas
deploy devel

while read -r -a line
do
  IFS='/' read -r -a parts <<< ${line[1]}

  if [[ ${parts[-1]} == v* ]]
  then
    git checkout ${line[0]}
    deploy ${parts[-1]}
  fi
done <<< $(git show-ref --tags)
