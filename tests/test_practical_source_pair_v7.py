"""Small semantic checks for the unfitted source-pair fallback."""
import dataclasses
import math
from types import SimpleNamespace
import unittest

import torch

from gcr.practical_scorer import BoundedPairScorer
from gcr.practical_training import PracticalTrainingConfig, TrainingExamples
from gcr.practical_source_pair_v7 import (
    SourcePairTrainingExamples, caption_key, mine_source_pair_plan, source_pair_triplet_loss,
)


class SourcePairFallbackChecks(unittest.TestCase):
    def test_conflicts_distinctness_and_closed_margin_band(self):
        # Two source IDs carry identical text. Neither can pair with the other.
        # A contradiction matching a supported caption must also be excluded.
        text = {0: "A dog runs.", 1: "A puppy runs.", 2: " A DOG runs! ",
                3: "An animal moves.", 4: "A dog sleeps.", 5: "A puppy rests.",
                6: "AN ANIMAL MOVES!", 7: "a dog runs", 8: "A bird flies."}
        scores = {0: 0., 1: 0., 2: 0., 4: .02, 5: -.02, 6: 0., 7: 0., 8: .0201}
        plan = mine_source_pair_plan([0, 1, 2], [4, 5, 6, 7, 8], [0, 1, 2, 3], text, scores, .01)
        self.assertEqual(plan.excluded_negative_indices, (6, 7))
        self.assertEqual(set(plan.triplets), {(0, 1, 4), (0, 1, 5), (1, 2, 4), (1, 2, 5)})
        self.assertEqual(plan.candidate_triplet_count, 6)
        # Numbers carry the very distinction a compositional loss must retain.
        self.assertNotEqual(caption_key("clothing from the 1800's"),
                            caption_key("clothing from the 1900's"))

    def test_two_positives_are_symmetric_and_both_receive_pressure(self):
        scores = torch.tensor([0., 0., 0.], requires_grad=True)
        loss = source_pair_triplet_loss(scores, torch.tensor([[0, 1, 2]]), margin=0., temperature=1.)
        self.assertAlmostEqual(float(loss.detach()), math.log(3), places=6)
        loss.backward()
        self.assertTrue(torch.allclose(scores.grad, torch.tensor([-1/3, -1/3, 2/3])))
        unequal = torch.tensor([.1, -.3, .2])
        first = source_pair_triplet_loss(unequal, torch.tensor([[0, 1, 2]]), margin=.005, temperature=.02)
        swapped = source_pair_triplet_loss(unequal, torch.tensor([[1, 0, 2]]), margin=.005, temperature=.02)
        self.assertEqual(float(first), float(swapped))

    def test_no_eligible_images_preserves_original_retention_and_gradient(self):
        # A complete small ownership problem, with every negative far outside
        # the compositional budget. The old and new retention terms must agree.
        images, texts, pairs, raw = torch.eye(3), [], [], []
        for image in range(3):
            relation = {}
            for caption in range(3):
                relation[len(texts)] = 1
                texts.append(images[image]); raw.append({"text": f"source {image} caption {caption}"})
            relation[len(texts)] = 2
            texts.append(images[image]); raw.append({"text": f"supported {image}"})
            relation[len(texts)] = 3
            texts.append(-images[image]); raw.append({"text": f"negative {image}"})
            pairs.append(relation)
        data = SimpleNamespace(images=images, texts=torch.stack(texts), pairs=pairs,
                               split_indices={"train": [0, 1, 2]}, logit_scale=10.,
                               manifest={"texts": raw})
        original = TrainingExamples(data, hard_count=1)
        fallback = SourcePairTrainingExamples(data, hard_count=1, epsilon=.01)
        self.assertEqual(fallback.mining_summary["eligible_image_count"], 0)
        model = BoundedPairScorer(dimension=3, epsilon=.01, hidden=4, rank=2)
        config = PracticalTrainingConfig(epsilon=.01, hard_negative_count=1)
        old_loss, old_values = original.batch_loss(model, [0, 1, 2], dataclasses.replace(config, composition_weight=0.))
        new_loss, new_values = fallback.batch_loss(model, [0, 1, 2], config)
        self.assertEqual(float(old_loss.detach()), float(new_loss.detach()))
        self.assertEqual(new_values["composition_loss"], 0.)
        for key in ("source_retrieval_ce", "retention_hinge_scaled"):
            self.assertEqual(old_values[key], new_values[key])
        old_grad = torch.autograd.grad(old_loss, tuple(model.parameters()), retain_graph=True)
        new_grad = torch.autograd.grad(new_loss, tuple(model.parameters()))
        for old, new in zip(old_grad, new_grad):
            self.assertTrue(torch.equal(old, new))


if __name__ == "__main__":
    unittest.main()
