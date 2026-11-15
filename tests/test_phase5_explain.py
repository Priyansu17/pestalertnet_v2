"""CPU smoke tests for Phase 5 explainability utilities."""

import numpy as np
import pytest
import torch
from torch import nn

from src.phase5.explain import (
    aggregate_attribution,
    feature_names,
    integrated_gradients_attr,
)


class LinearReveal(nn.Module):
    """Deterministic linear model returning one logit per sample."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(21 * 25, 1)
        with torch.no_grad():
            self.linear.weight.fill_(1.0)
            self.linear.bias.zero_()

    def forward(self, x):
        return self.linear(x.reshape(x.shape[0], -1)).squeeze(-1)


@pytest.fixture
def model():
    return LinearReveal().eval()


@pytest.fixture
def batch():
    rng = np.random.default_rng(7)
    return torch.from_numpy(
        rng.standard_normal((3, 21, 25)).astype(np.float32)
    )


def test_feature_names():
    from src.phase5.features import EXPECTED_FEATURE_NAMES
    assert feature_names() == EXPECTED_FEATURE_NAMES
    assert len(feature_names()) == 25


def test_single_sample_shape_and_finite(model):
    x = torch.randn(21, 25)
    attr = integrated_gradients_attr(model, x, n_steps=8)
    assert attr.shape == (1, 21, 25)
    assert np.isfinite(attr).all()


def test_batch_shape_and_aggregation(model, batch):
    attr = integrated_gradients_attr(model, batch, n_steps=8)
    assert attr.shape == (3, 21, 25)
    assert np.isfinite(attr).all()

    summary = aggregate_attribution(attr)
    assert summary.shape == (3, 25)
    np.testing.assert_allclose(summary, np.abs(attr).sum(axis=1))


def test_zero_input_zero_baseline(model):
    x = torch.zeros(1, 21, 25)
    attr = integrated_gradients_attr(model, x, n_steps=8)
    assert np.allclose(attr, 0.0, atol=1e-6)


def test_custom_baseline_expansion(model, batch):
    baseline = torch.zeros(21, 25)
    attr = integrated_gradients_attr(
        model, batch, baseline=baseline, n_steps=8
    )
    assert attr.shape == (3, 21, 25)
    assert np.isfinite(attr).all()


def test_invalid_input_shape(model):
    with pytest.raises(ValueError):
        integrated_gradients_attr(model, torch.randn(21, 24))


def test_nonfinite_input(model):
    x = torch.randn(1, 21, 25)
    x[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        integrated_gradients_attr(model, x)


def test_invalid_baseline_shape(model, batch):
    with pytest.raises(ValueError):
        integrated_gradients_attr(
            model, batch, baseline=torch.randn(3, 21, 1)
        )


def test_nonfinite_baseline(model, batch):
    baseline = torch.zeros(3, 21, 25)
    baseline[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        integrated_gradients_attr(model, batch, baseline=baseline)


def test_invalid_n_steps(model):
    with pytest.raises(ValueError):
        integrated_gradients_attr(
            model, torch.randn(21, 25), n_steps=0
        )
