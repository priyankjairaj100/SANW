#!/usr/bin/env python3
"""Independent loss and selector checks. This script performs no model fitting."""
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from gcr.allocation_distillation import allocation_distillation_loss
from gcr.strengthen_retention import choose_family


def reference(images, texts, relations, teacher_images, teacher_texts, mix, beta):
    logits = 7.0 * images @ texts.T
    source = relations == 1
    expanded = source | (relations == 2)

    def directional(values, mask):
        eligible = mask.any(1)
        values, mask = values[eligible], mask[eligible]
        return (values.logsumexp(1) - (values * mask).sum(1) / mask.sum(1)).mean()

    source_loss = .5 * (directional(logits, source) + directional(logits.T, source.T))
    supported_loss = .5 * (directional(logits, expanded) + directional(logits.T, expanded.T))
    columns = source.any(0)
    ti = teacher_images.detach() / teacher_images.detach().norm(dim=1, keepdim=True)
    tt = teacher_texts.detach() / teacher_texts.detach().norm(dim=1, keepdim=True)
    teacher = (7.0 * ti @ tt.T)[:, columns] / 2.0
    student = logits[:, columns] / 2.0

    def divergence(t, s):
        lt = t - t.logsumexp(1, keepdim=True)
        ls = s - s.logsumexp(1, keepdim=True)
        return (lt.exp() * (lt - ls)).sum(1).mean()

    kl = .5 * (divergence(teacher, student) + divergence(teacher.T, student.T))
    return mix * source_loss + (1.0 - mix) * supported_loss + beta * 4.0 * kl


def main():
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(92851)
    relations = torch.tensor([[1, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 2, 2, 3, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2]])
    images = torch.randn(3, 7, generator=generator, dtype=torch.float64).requires_grad_()
    texts = torch.randn(11, 7, generator=generator, dtype=torch.float64).requires_grad_()
    teacher_images = torch.randn(3, 7, generator=generator, dtype=torch.float64).requires_grad_()
    teacher_texts = torch.randn(11, 7, generator=generator, dtype=torch.float64).requires_grad_()
    largest_value_error, largest_gradient_error, cases = 0.0, 0.0, 0
    for mix in (0.0, .5, .8, 1.0):
        for beta in (0.0, 1.0, 4.0, 16.0):
            actual = allocation_distillation_loss(images, texts, relations, 7.0,
                source_mix=mix, beta=beta, frozen_images=teacher_images,
                frozen_texts=teacher_texts)
            expected = reference(images, texts, relations, teacher_images, teacher_texts, mix, beta)
            torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-13)
            ga = torch.autograd.grad(actual, (images, texts), retain_graph=True)
            ge = torch.autograd.grad(expected, (images, texts), retain_graph=True)
            for a, e in zip(ga, ge):
                torch.testing.assert_close(a, e, atol=2e-11, rtol=2e-13)
                largest_gradient_error = max(largest_gradient_error, (a-e).abs().max().item())
            largest_value_error = max(largest_value_error, abs((actual-expected).item()))
            teacher_gradients = torch.autograd.grad(actual, (teacher_images, teacher_texts),
                                                   allow_unused=True, retain_graph=True)
            assert teacher_gradients == (None, None)
            cases += 1
    changed_teacher = teacher_texts.detach().clone()
    changed_teacher[~(relations == 1).any(0)] += 100.0
    before = allocation_distillation_loss(images, texts, relations, 7.0, source_mix=.8,
        beta=4.0, frozen_images=teacher_images, frozen_texts=teacher_texts)
    after = allocation_distillation_loss(images, texts, relations, 7.0, source_mix=.8,
        beta=4.0, frozen_images=teacher_images, frozen_texts=changed_teacher)
    assert torch.equal(before, after)

    def row(epoch, accuracy, image=.8, text=.7):
        return {"epoch": epoch, "native": {"relation_accuracy": accuracy},
                "source_retrieval": {"i2t_r1": image, "t2i_r1": text}}
    candidates = []
    for rate in (3e-4, 1e-4):
        for parameter in ((.8, 1.0), (.5, 16.0), (.5, 4.0)):
            histories = {seed: [row(0, .5), row(1, .95, text=.68),
                               row(2, .9, image=.79, text=.69), row(3, .9)]
                         for seed in (17, 29, 43)}
            candidates.append({"policy": str(parameter), "learning_rate": rate,
                               "parameter": parameter, "histories": histories})
    primary = choose_family(candidates, (17, 29, 43), 1.0)
    sensitivity = choose_family(candidates, (17, 29, 43), 0.0)
    assert primary["learning_rate"] == 1e-4 and primary["parameter"] == (.5, 4.0)
    assert {r["epoch"] for r in primary["best"].values()} == {2}
    assert {r["epoch"] for r in sensitivity["best"].values()} == {3}
    for candidate in candidates:
        candidate["histories"][17][2]["native"]["relation_accuracy"] = .4
        candidate["histories"][17][3]["native"]["relation_accuracy"] = .4
    mixed = choose_family(candidates, (17, 29, 43), 1.0)
    assert [mixed["best"][seed]["epoch"] for seed in (17, 29, 43)] == [0, 2, 2]
    paths = ["src/gcr/allocation_distillation.py", "src/gcr/strengthen_retention.py",
             "src/gcr/retention_losses.py", "src/gcr/losses.py"]
    report = {"passed": True, "scope": "independent objective, gradients, teacher domain, selector",
              "scientific_training_performed": False, "test_data_read": False,
              "objective_cases": cases, "largest_value_error": largest_value_error,
              "largest_gradient_error": largest_gradient_error,
              "teacher_detached": True, "hypotheses_excluded_from_teacher_domain": True,
              "selector_checks": ["both directions", "1pp and 0pp tolerance", "earliest epoch",
                  "shared learning rate", "lexicographic mixture/beta ties", "mixed frozen/trained states"],
              "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}
    output = Path(__file__).with_name("allocation_distillation_independent_design_checks.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
