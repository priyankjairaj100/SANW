import unittest

import torch

from gcr.practical_training import (
    joint_triplet_loss, pessimistic_top1, retained_margin_loss, select_development_epoch,
    source_retrieval_loss, worst_positive_loss,
)


class PracticalTrainingTests(unittest.TestCase):
    def test_joint_triplet_loss_matches_explicit_cartesian_mean(self):
        sources = torch.tensor([0.5, 0.6], requires_grad=True)
        supported = torch.tensor([0.2, 0.4], requires_grad=True)
        negatives = torch.tensor([0.3, 0.1], requires_grad=True)
        actual = joint_triplet_loss(sources, supported, negatives, margin=0.01, temperature=0.1)
        expected = torch.stack([torch.logsumexp(torch.stack((torch.tensor(0.0), (n-s+0.01)/0.1, (n-p+0.01)/0.1)), 0)
                                for s in sources for p in supported for n in negatives]).mean()
        self.assertTrue(torch.allclose(actual, expected))
        actual.backward()
        self.assertTrue(bool((sources.grad < 0).all()))
        self.assertTrue(bool((supported.grad < 0).all()))
        self.assertTrue(bool((negatives.grad > 0).all()))

    def test_worst_positive_updates_the_failing_valid_caption(self):
        positive = torch.tensor([0.6, 0.2], requires_grad=True)
        negative = torch.tensor([0.3], requires_grad=True)
        loss = worst_positive_loss(positive, negative, margin=0.0, temperature=0.1)
        loss.backward()
        self.assertEqual(float(positive.grad[0]), 0.0)
        self.assertLess(float(positive.grad[1]), 0.0)
        self.assertGreater(float(negative.grad[0]), 0.0)

    def test_source_ce_requires_caption_ownership(self):
        scores = torch.tensor([[0.8, 0.7, 0.1], [0.1, 0.0, 0.8]])
        owners = torch.tensor([[True, True, False], [False, False, True]])
        self.assertLess(float(source_retrieval_loss(scores, owners, 10)), 1.0)
        owners[1, 0] = True
        with self.assertRaises(ValueError):
            source_retrieval_loss(scores, owners, 10)

    def test_margin_preservation_ignores_teacher_errors_and_caps_target(self):
        loss = retained_margin_loss(torch.tensor([0.008, 0.001, -0.5]), torch.zeros(3),
                                    torch.tensor([0.5, 0.002, -0.1]), cap=0.01)
        self.assertAlmostEqual(float(loss), 0.0015, places=7)

    def test_pessimistic_i2t_tie_at_last_candidate_fails(self):
        # Two relevant captions: the third maximal candidate proves a bad tie.
        owner = torch.tensor([0, 0, 1, 1])
        scores = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.5, 0.4]])
        indices = torch.tensor([[0, 1, 2], [2, 3, 0]])
        self.assertEqual(pessimistic_top1(scores, indices, direction="i2t", owner=owner).tolist(), [False, True])

    def test_pessimistic_t2i_tie_fails(self):
        owner = torch.tensor([0, 1])
        scores = torch.tensor([[1.0, 1.0], [1.0, 0.5]])
        indices = torch.tensor([[0, 1], [1, 0]])
        self.assertEqual(pessimistic_top1(scores, indices, direction="t2i", owner=owner).tolist(), [False, True])

    def test_selection_rejects_retrieval_tradeoff_and_frozen_epoch(self):
        def row(epoch, ic, tc, jc, margin):
            return {"epoch": epoch, "optimizer_steps": epoch * 10,
                    "retrieval": {"image_correct_count": ic, "text_correct_count": tc, "i2t_r1": ic / 10, "t2i_r1": tc / 50},
                    "composition": {"paired_joint_accuracy": jc / 10, "mean_paired_joint_margin": margin},
                    "checkpoint": {"path": str(epoch)}}
        rows = [row(0, 8, 35, 4, 0.0), row(1, 7, 40, 10, 1.0), row(2, 8, 35, 5, 0.1), row(3, 8, 35, 5, 0.1)]
        self.assertEqual(select_development_epoch(rows)["selected_epoch"], 2)
        self.assertIsNone(select_development_epoch(rows[:2])["selected_epoch"])


if __name__ == "__main__":
    unittest.main()
