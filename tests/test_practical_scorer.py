"""Scoring, numerical envelope, gradient, and exact retrieval checks."""

import unittest

import torch
from torch.nn import functional as F

from gcr.practical_scorer import BoundedPairScorer, exact_topk


class PracticalScorerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1841)
        torch.set_num_threads(1)
        self.images = F.normalize(torch.randn(11, 8), dim=-1)
        self.texts = F.normalize(torch.randn(17, 8), dim=-1)

    def trained_scorer(self, epsilon=0.1):
        model = BoundedPairScorer(8, epsilon=epsilon, hidden=13, rank=5)
        with torch.no_grad():
            model.output.weight.normal_(std=2)
            model.output.bias.fill_(0.4)
        return model

    def test_zero_initialization_is_exact_frozen_paired_score(self):
        model = BoundedPairScorer(8, hidden=13, rank=5)
        actual = model.score_pairs(self.images, self.texts[:11])
        expected = (self.images.double() * self.texts[:11].double()).sum(-1)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(actual.dtype, torch.float64)
        self.assertEqual(torch.count_nonzero(model.residual_pairs(self.images, self.texts[:11])), 0)

    def test_final_bound_and_initial_trainability(self):
        model = BoundedPairScorer(8, epsilon=0.03, hidden=13, rank=5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
        loss = -model.score_pairs(self.images, self.texts[:11]).mean()
        loss.backward()
        self.assertGreater(float(model.output.weight.grad.abs().sum()), 0)
        optimizer.step()
        self.assertGreater(float(model.residual_pairs(self.images, self.texts[:11]).abs().sum()), 0)
        with torch.no_grad():
            model.output.weight.mul_(1e6)
        residual = model.residual_pairs(self.images, self.texts[:11])
        self.assertTrue(torch.all(residual.abs() <= model.epsilon))
        self.assertEqual(residual.dtype, torch.float64)

    def test_matrix_equals_paired_scores_across_chunk_sizes(self):
        model = self.trained_scorer()
        expected = model.score_pairs(
            self.images.repeat_interleave(17, dim=0), self.texts.repeat((11, 1))
        ).reshape(11, 17)
        for chunk in (1, 7, 8192):
            actual = model.score_matrix(self.images, self.texts, pair_chunk=chunk)
            torch.testing.assert_close(actual, expected, atol=3e-8, rtol=0)

    def test_exact_shortlist_matches_exhaustive_both_directions(self):
        for epsilon in (0, 0.01, 0.3):
            model = self.trained_scorer(epsilon)
            full = model.score_matrix(self.images, self.texts, pair_chunk=7)
            for direction in ("i2t", "t2i"):
                matrix = full if direction == "i2t" else full.T
                for k in (1, 4, matrix.shape[1]):
                    expected_indices = torch.argsort(matrix, descending=True, stable=True)[:, :k]
                    expected_scores = torch.gather(matrix, 1, expected_indices)
                    scores, indices = exact_topk(
                        model, self.images, self.texts, k, direction, query_chunk=3, pair_chunk=5
                    )
                    torch.testing.assert_close(indices, expected_indices, atol=0, rtol=0)
                    # The shortlist is exact; different float32 GEMM batch
                    # shapes retain normal network roundoff, scaled by epsilon.
                    torch.testing.assert_close(scores, expected_scores, atol=1e-6 * epsilon + 1e-13, rtol=0)

    def test_boundary_candidate_can_win_tie_outside_frozen_top_one(self):
        class SignResidual(BoundedPairScorer):
            def residual_pairs(self, images, texts):
                return self.epsilon * texts[:, 1].sign().double()

        model = SignResidual(2, epsilon=0.125, hidden=2, rank=1)
        images = torch.tensor([[1.0, 0.0]])
        texts = torch.tensor([[0.25, 1.0], [0.5, -1.0], [0.0, 1.0]])
        scores, indices = model.exact_topk(images, texts)
        # Candidate zero is exactly 2*epsilon below the frozen winner. Both
        # corrected scores equal 0.375, so its lower manifest index must win.
        self.assertEqual(indices.item(), 0)
        self.assertEqual(scores.item(), 0.375)

    def test_all_ties_use_lower_gallery_index_in_both_directions(self):
        model = BoundedPairScorer(8, hidden=13, rank=5)
        images = torch.zeros(7, 8)
        texts = torch.zeros(9, 8)
        for direction, rows in (("i2t", 7), ("t2i", 9)):
            scores, indices = model.exact_topk(images, texts, k=4, direction=direction, pair_chunk=3)
            torch.testing.assert_close(indices, torch.arange(4).expand(rows, -1), atol=0, rtol=0)
            self.assertEqual(torch.count_nonzero(scores), 0)

    def test_training_means_are_fixed_cloned_checkpoint_buffers(self):
        image_mean = self.images.mean(0)
        text_mean = self.texts.mean(0)
        model = BoundedPairScorer(8, hidden=13, rank=5, image_mean=image_mean, text_mean=text_mean)
        with torch.no_grad():
            model.output.weight.normal_()
        saved_mean = model.image_mean.clone()
        image_mean.add_(100)
        torch.testing.assert_close(model.image_mean, saved_mean, atol=0, rtol=0)
        self.assertNotIn("image_mean", dict(model.named_parameters()))
        restored = BoundedPairScorer(**model.config())
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(
            model.score_matrix(self.images, self.texts),
            restored.score_matrix(self.images, self.texts), atol=0, rtol=0,
        )

    def test_empty_shapes_and_invalid_arguments(self):
        model = BoundedPairScorer(8, hidden=13, rank=5)
        self.assertEqual(tuple(model.score_matrix(self.images[:0], self.texts).shape), (0, 17))
        self.assertEqual(tuple(model.exact_topk(self.images[:0], self.texts, k=3)[0].shape), (0, 3))
        for kwargs in ({"k": 18}, {"k": 0}, {"direction": "reverse"}, {"pair_chunk": 0}):
            with self.assertRaises(ValueError):
                model.exact_topk(self.images, self.texts, **kwargs)
        with self.assertRaises(ValueError):
            model.score_pairs(self.images, self.texts)
        with self.assertRaises(ValueError):
            BoundedPairScorer(8, epsilon=float("nan"))


if __name__ == "__main__":
    unittest.main()
