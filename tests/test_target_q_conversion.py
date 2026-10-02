import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "tdmpc2"))
from common.world_model import WorldModel


def make_model(dropout=0.):
    cfg = SimpleNamespace(multitask=False, obs_shape={"state": (3,)}, task_dim=0,
                          num_enc_layers=2, enc_dim=8, latent_dim=8, simnorm_dim=4,
                          action_dim=2, mlp_dim=8, num_bins=3, episodic=False,
                          num_q=2, dropout=dropout, log_std_min=-5., log_std_max=2., tau=0.1)
    return WorldModel(cfg)


def tensors(params):
    return {key: value for key, value in params.items(True, True)
            if torch.is_floating_point(value)}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_conversion_preserves_distinct_online_and_target_values(dtype):
    model = make_model()
    with torch.no_grad():
        for value in tensors(model._Qs.params).values():
            value.fill_(2.)
        for value in tensors(model._target_Qs_params).values():
            value.fill_(0.5)
    assert model.to(dtype=dtype) is model
    for value in tensors(model._Qs.params).values():
        assert value.dtype == dtype
        torch.testing.assert_close(value, torch.full_like(value, 2.))
    for value in tensors(model._target_Qs_params).values():
        assert value.dtype == dtype
        torch.testing.assert_close(value, torch.full_like(value, 0.5))
    for key, value in tensors(model._detach_Qs_params).items():
        assert value.data_ptr() == model._Qs.params[key].data_ptr()
    assert model._target_Qs.params is model._target_Qs_params


def test_polyak_history_survives_conversion_and_remains_independent():
    model = make_model()
    with torch.no_grad():
        for value in tensors(model._Qs.params).values():
            value.fill_(2.)
        for value in tensors(model._target_Qs_params).values():
            value.zero_()
        model.soft_update_target_Q()
        model.to(dtype=torch.float64)
        model.soft_update_target_Q()
    # Two Polyak steps: (1-tau)^2 * 0 + (1-(1-tau)^2) * 2.
    for value in tensors(model._target_Qs_params).values():
        torch.testing.assert_close(value, torch.full_like(value, 0.38))
    for key, value in tensors(model._target_Qs_params).items():
        assert value.data_ptr() != model._Qs.params[key].data_ptr()


def test_converted_target_forward_matches_the_preconversion_network():
    model = make_model().eval()
    with torch.no_grad():
        for value in tensors(model._target_Qs_params).values():
            value.add_(0.1)
        z = torch.randn(4, 8)
        action = torch.randn(4, 2)
        before = model.Q(z, action, None, return_type="all", target=True)
        model.to(dtype=torch.float64)
        after = model.Q(z.double(), action.double(), None, return_type="all", target=True)
    torch.testing.assert_close(after, before.double(), atol=1e-6, rtol=1e-5)


def test_conversion_keeps_target_dropout_disabled_during_online_training():
    model = make_model(dropout=0.5).train()
    with torch.no_grad():
        for value in tensors(model._target_Qs_params).values():
            value.add_(0.1)
    assert model._Qs.training
    assert not model._target_Qs.training
    model.to(dtype=torch.float64)
    assert model.training and model._Qs.training
    assert not model._target_Qs.training
    z = torch.randn(4, 8, dtype=torch.float64)
    action = torch.randn(4, 2, dtype=torch.float64)
    first = model.Q(z, action, None, return_type="all", target=True)
    second = model.Q(z, action, None, return_type="all", target=True)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
