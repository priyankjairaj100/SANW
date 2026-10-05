"""Independent finite-gallery and joint-triple objective checks for v9."""
import itertools
import unittest

import numpy as np

from gcr.practical_constrained_v8 import ConstrainedBilinearScorer, FullGalleryConstraints
from gcr.practical_joint_v9 import FullGallerySourceLoss, JointCompositionExamples, JointFitConfig


def lse(x):
    x = np.asarray(x)
    maximum = x.max()
    return float(maximum + np.log(np.exp(x-maximum).sum()))


class IndependentJointReview(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(193)
        self.images = rng.normal(size=(4, 6))
        self.images /= np.linalg.norm(self.images, axis=1, keepdims=True)
        self.texts = rng.normal(size=(20, 6))
        self.texts /= np.linalg.norm(self.texts, axis=1, keepdims=True)
        self.owner = np.array([0, 0, 0, 1, 2, 2, 3])
        self.scorer = ConstrainedBilinearScorer.from_training(self.images, self.texts, 3)
        self.constraints = FullGalleryConstraints(self.images, self.texts[:7], self.owner, self.scorer)
        self.retrieval = FullGallerySourceLoss(self.constraints, 4.2)
        self.coefficient = rng.normal(size=(3, 3)) * .03
        self.sources = [[0, 1, 2], [3], [4, 5], [6]]
        self.supported = [[7], [8, 9], [10], [11, 12, 13]]
        self.negatives = [[14, 15], [16], [17], [18, 19]]
        self.composition = JointCompositionExamples(self.images, self.texts, self.sources, self.supported, self.negatives, self.scorer)
        self.config = JointFitConfig(rank=3, temperature=.13)

    def retrieval_reference(self, a, query_images=None):
        c = self.constraints
        queries = np.arange(len(self.images)) if query_images is None else query_images
        scores = c.frozen + c.x @ a @ c.y.T
        image_losses, text_losses = [], []
        for i in queries:
            anchor = c.best_positive[i]
            candidates = np.concatenate(([anchor], np.flatnonzero(c.owner != i)))
            logits = self.retrieval.scale * scores[i]
            image_losses.append(lse(logits[candidates])-logits[anchor])
        for j in np.flatnonzero(np.isin(c.owner, queries)):
            logits = self.retrieval.scale * scores[:, j]
            text_losses.append(lse(logits)-logits[c.owner[j]])
        caption_denominator = len(queries) * len(c.owner) / len(self.images)
        return (np.mean(image_losses) + sum(text_losses)/caption_denominator) / 2

    def composition_reference(self, a):
        x = self.scorer.image_coordinates(self.images)
        y = self.scorer.text_coordinates(self.texts)
        scores = np.asarray([[np.sum(v*t) + np.sum((x[i]@a)*y[j]) for j,t in enumerate(self.texts)]
                             for i,v in enumerate(self.images)])
        image_means = []
        for i in range(len(self.images)):
            losses = []
            for source, supported, negative in itertools.product(self.sources[i], self.supported[i], self.negatives[i]):
                losses.append(lse([0, (scores[i,negative]-scores[i,source]+self.config.composition_margin)/self.config.temperature,
                                  (scores[i,negative]-scores[i,supported]+self.config.composition_margin)/self.config.temperature]))
            image_means.append(np.mean(losses))
        return float(np.mean(image_means))

    def test_full_gallery_ce_value_and_every_gradient_coordinate(self):
        actual, gradient = self.retrieval.loss_gradient(self.coefficient)
        self.assertAlmostEqual(actual, self.retrieval_reference(self.coefficient), places=14)
        for index in np.ndindex(self.coefficient.shape):
            plus, minus = self.coefficient.copy(), self.coefficient.copy()
            plus[index] += 1e-6; minus[index] -= 1e-6
            numeric = (self.retrieval_reference(plus)-self.retrieval_reference(minus))/2e-6
            self.assertAlmostEqual(numeric, gradient[index], places=8)

    def test_unequal_caption_owner_counts_give_unbiased_query_gradient(self):
        value, gradient = self.retrieval.loss_gradient(self.coefficient)
        parts = [self.retrieval.loss_gradient(self.coefficient, [i]) for i in range(len(self.images))]
        self.assertAlmostEqual(value, np.mean([p[0] for p in parts]), places=14)
        np.testing.assert_allclose(gradient, np.mean([p[1] for p in parts], axis=0), atol=3e-16, rtol=1e-14)
        for i, (part, _) in enumerate(parts):
            self.assertAlmostEqual(part, self.retrieval_reference(self.coefficient, [i]), places=14)

    def test_joint_lse_enumeration_and_every_gradient_coordinate(self):
        actual, gradient = self.composition.loss_gradient(self.coefficient, self.composition.eligible, self.config)
        self.assertAlmostEqual(actual, self.composition_reference(self.coefficient), places=13)
        for index in np.ndindex(self.coefficient.shape):
            plus, minus = self.coefficient.copy(), self.coefficient.copy()
            plus[index] += 1e-6; minus[index] -= 1e-6
            numeric = (self.composition_reference(plus)-self.composition_reference(minus))/2e-6
            self.assertAlmostEqual(numeric, gradient[index], places=8)

    def test_fixed_positive_losses_are_convex_in_the_affine_score(self):
        a, b, fraction = self.coefficient, -1.8*self.coefficient, .237
        for function in (self.retrieval_reference, self.composition_reference):
            self.assertLessEqual(function(fraction*a+(1-fraction)*b), fraction*function(a)+(1-fraction)*function(b)+1e-14)

    def test_uniform_owner_batches_match_global_objective_with_missing_labels(self):
        negative = [self.negatives[0], [], [], []]
        examples = JointCompositionExamples(self.images, self.texts, self.sources, self.supported, negative, self.scorer)
        loss, gradient = examples.loss_gradient(self.coefficient, examples.eligible, self.config)
        # Half these owner batches have no composition supervision.
        batches = [examples.query_batch_loss_gradient(self.coefficient, list(indices), self.config)
                   for indices in itertools.combinations(range(4), 2)]
        self.assertAlmostEqual(np.mean([x[0] for x in batches]), loss, places=14)
        np.testing.assert_allclose(np.mean([x[1] for x in batches], axis=0), gradient, atol=5e-16, rtol=1e-14)


if __name__ == "__main__":
    unittest.main()
