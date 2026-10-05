"""Independent synthetic checks for the two frozen-feature LABCLIP arms."""
import unittest

import numpy as np
import torch

from gcr.practical_labclip_v9 import (
    LABCLIPConfig, NormalizedFullRankAlignment, canonical_transform,
    epoch_batches, hard_negative_batch_loss, validate_owner_split,
)


class IndependentLABCLIPReview(unittest.TestCase):
    def test_hnb_value_and_gradients_against_explicit_formula(self):
        rng = np.random.default_rng(7)
        images = torch.tensor(rng.normal(size=(4, 5)), dtype=torch.float64)
        positive = torch.tensor(rng.normal(size=(4, 5)), dtype=torch.float64, requires_grad=True)
        negative = torch.tensor(rng.normal(size=(4, 5)), dtype=torch.float64, requires_grad=True)
        log_scale = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
        loss, _ = hard_negative_batch_loss(images, positive, negative, log_scale.exp())
        logits = log_scale.exp() * images @ torch.cat((positive, negative), dim=0).T
        explicit_i = torch.stack([torch.logsumexp(logits[i], 0) - logits[i, i] for i in range(4)]).mean()
        explicit_t = torch.stack([torch.logsumexp(logits[:, j], 0) - logits[j, j] for j in range(4)]).mean()
        expected = (explicit_i + explicit_t) / 2
        torch.testing.assert_close(loss, expected, rtol=1e-14, atol=1e-14)
        grad = torch.autograd.grad(loss, (positive, negative, log_scale), retain_graph=True)
        independent_grad = torch.autograd.grad(expected, (positive, negative, log_scale))
        for a, b in zip(grad, independent_grad):
            torch.testing.assert_close(a, b, rtol=1e-13, atol=1e-13)

    def test_alignment_identity_normalization_and_learned_scale_modes(self):
        texts = torch.tensor([[2., 0., 0.], [1., 2., 3.]])
        for learned in (True, False):
            model = NormalizedFullRankAlignment(3, native_logit_scale=70., learned_scale=learned)
            np.testing.assert_array_equal(model.linear.weight.detach().numpy(), np.eye(3))
            torch.testing.assert_close(model(texts), texts / texts.norm(dim=1, keepdim=True), rtol=0, atol=0)
            self.assertEqual(model.log_scale.requires_grad, learned)
            self.assertAlmostEqual(float(model.logit_scale), 1. if learned else 70., places=12)
            self.assertIsNone(model.linear.bias)

    def test_canonical_transform_matches_scalar_and_row_batching(self):
        rng = np.random.default_rng(300)
        texts, weight = rng.normal(size=(6, 7)), rng.normal(size=(7, 7))
        actual = canonical_transform(texts, weight)
        separate = np.concatenate([canonical_transform(texts[i:i+1], weight) for i in range(len(texts))])
        np.testing.assert_array_equal(actual, separate)
        expected = []
        for text in texts:
            z = np.einsum("d,kd->k", text, weight, optimize=False)
            expected.append(z / np.sqrt(np.sum(z*z)))
        np.testing.assert_array_equal(actual, expected)
        with self.assertRaises(FloatingPointError):
            canonical_transform(texts, np.zeros((7, 7)))

    def test_sampler_training_isolation_and_caption_coverage(self):
        sources = [list(range(i*5, i*5+5)) for i in range(6)]
        negatives = [[100+i, 200+i] for i in range(6)]
        train = np.array([0, 2, 3, 5])
        wanted = {(int(i), int(t)) for i in train for t in sources[i]}
        adapted = list(epoch_batches(train, sources, negatives, LABCLIPConfig(), np.random.default_rng(17)))
        published = list(epoch_batches(train, sources, negatives, LABCLIPConfig.published_control(), np.random.default_rng(17)))
        for batches in (adapted, published):
            actual = []
            for ii, pp, nn in batches:
                self.assertTrue(set(ii) <= set(train))
                actual.extend(zip(ii.tolist(), pp.tolist()))
                self.assertTrue(all(int(n) in negatives[int(i)] for i, n in zip(ii, nn)))
            self.assertEqual(len(actual), len(wanted))
            self.assertEqual(set(actual), wanted)
        self.assertTrue(all(len(ii) == len(set(ii)) for ii, _, _ in adapted))
        self.assertTrue(any(len(ii) > len(set(ii)) for ii, _, _ in published))
        self.assertEqual(len(adapted), 5)
        self.assertEqual(len(published), 1)

    def test_budget_uses_five_caption_passes_per_adapted_epoch(self):
        sources = [list(range(i*5, i*5+5)) for i in range(960)]
        negatives = [[6000+i] for i in range(960)]
        for config, expected_steps in ((LABCLIPConfig(), 40), (LABCLIPConfig.published_control(), 5)):
            batches = list(epoch_batches(np.arange(960), sources, negatives, config, np.random.default_rng(17)))
            self.assertEqual(len(batches), expected_steps)
            self.assertEqual(sum(len(ii) for ii, _, _ in batches), 4800)

    def test_split_rejects_overlap_and_maps_manifest_rows(self):
        split = {"schema": "sanw_inner_training_owner_split_v9", "train_image_manifest_indices": [4, 10],
                 "validation_image_manifest_indices": [7]}
        training, validation = validate_owner_split(split, [4, 7, 10])
        np.testing.assert_array_equal(training, [0, 2])
        np.testing.assert_array_equal(validation, [1])
        split["validation_image_manifest_indices"] = [7, 10]
        with self.assertRaises(ValueError):
            validate_owner_split(split, [4, 7, 10])


if __name__ == "__main__":
    unittest.main()
