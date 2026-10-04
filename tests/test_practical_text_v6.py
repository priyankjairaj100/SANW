"""The backup must preserve exact initialization and a task-independent bound."""
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from gcr.practical_text_v6 import TextBlock
from gcr.practical_text_training_v6 import LastTextBlock, PreparedTextScores


def test_last_block_initialization_is_exact_and_trainable():
    torch.manual_seed(17)
    tower = SimpleNamespace(
        transformer=SimpleNamespace(resblocks=[TextBlock()]),
        ln_final=nn.LayerNorm(512),
        text_projection=nn.Parameter(torch.randn(512, 16)),
        attention_mask=torch.full((77, 77), float("-inf")).triu_(1),
    )
    model = LastTextBlock(tower).train()
    prefix = torch.randn(3, 5, 512)
    ends = torch.tensor([2, 3, 4])
    delta = model(prefix, ends)
    assert torch.equal(delta, torch.zeros_like(delta))
    (delta * torch.randn_like(delta)).sum().backward()
    assert any(parameter.grad is not None and parameter.grad.count_nonzero() > 0
               for parameter in model.block.parameters())
    assert all(parameter.grad is None for parameter in model.reference_block.parameters())
    with torch.inference_mode():
        model.eval()
        assert model(prefix, ends).count_nonzero() == 0


def test_bounded_scores_paired_dense_and_manifest_ties():
    torch.manual_seed(29)
    texts = torch.randn(4, 8)
    # First two captions tie exactly, including their residuals.
    texts[1] = texts[0]
    images = torch.randn(3, 8)
    delta = torch.randn(4, 8) * 100
    delta[1] = delta[0]
    lookup = {row.numpy().tobytes(): index for index, row in reversed(list(enumerate(texts)))}
    cache = SimpleNamespace(indices_for_vectors=lambda value: [lookup[row.numpy().tobytes()] for row in value])
    scorer = PreparedTextScores(cache, list(range(4)), delta, .01)
    dense = scorer.score_matrix(images, texts)
    frozen = images.double() @ texts.double().T
    assert (dense - frozen).abs().max() <= .01 + 1e-14
    paired = scorer.score_pairs(images.repeat_interleave(4, 0), texts.repeat(3, 1)).reshape(3, 4)
    assert torch.allclose(paired, dense, rtol=0, atol=1e-14)
    for direction, scores in [("i2t", dense), ("t2i", dense.T)]:
        values, indices = scorer.exact_topk(images, texts, k=1, direction=direction)
        expected = scores.argmax(dim=1, keepdim=True)
        assert torch.equal(indices, expected)
        assert torch.equal(values, scores.gather(1, expected))
