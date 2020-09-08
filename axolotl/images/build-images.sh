#!/bin/sh -e

docker login -u gitlab-ci-token -p "$CI_JOB_TOKEN" "$CI_REGISTRY"

for IMAGE_NAME in "$@"; do
    if [ "$IMAGE_NAME" = "base" ]; then
        echo "Bulding "$CI_REGISTRY_IMAGE/$IMAGE_NAME":latest"
        docker build -t "$CI_REGISTRY_IMAGE/$IMAGE_NAME:latest" -f images/base.dockerfile .
        echo "Pushing "$CI_REGISTRY_IMAGE/$IMAGE_NAME":latest"
        docker push "$CI_REGISTRY_IMAGE/$IMAGE_NAME:latest"
        echo "Done"
    fi

    if [ "$IMAGE_NAME" = "axolotl" ]; then
        echo "Bulding "$CI_REGISTRY_IMAGE/$IMAGE_NAME":latest"
        docker build -t "$CI_REGISTRY_IMAGE/$IMAGE_NAME:latest" -f images/axolotl.dockerfile .
        echo "Pushing "$CI_REGISTRY_IMAGE/$IMAGE_NAME":latest"
        docker push "$CI_REGISTRY_IMAGE/$IMAGE_NAME:latest"
        echo "Done"
    fi
done
