#!/bin/bash -e

# Assumption is that this repository is cloned into "d3m-test-data" directory
# which is a sibling of "d3m-primitives" directory.

for PRIMITIVE in d3m.primitives.regression.monomial.Test \
    d3m.primitives.operator.increment.Test \
    d3m.primitives.operator.sum.Test \
    d3m.primitives.data_generation.random.Test \
    d3m.primitives.operator.primitive_sum.Test \
    d3m.primitives.operator.null.TransformerTest \
    d3m.primitives.operator.null.UnsupervisedLearnerTest \
    d3m.primitives.classification.random_classifier.Test \
    d3m.primitives.evaluation.compute_scores.Test ; do
  echo $PRIMITIVE
  python -m d3m primitive describe -i 4 $PRIMITIVE > primitive.json
  pushd ../d3m-primitives
  ./add.py ../d3m-tests-data/primitive.json
  popd
done
