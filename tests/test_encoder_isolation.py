import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

spec = importlib.util.spec_from_file_location(
    "tdmpc_layers", Path(__file__).parents[1] / "tdmpc2/common/layers.py")
layers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layers)


def cfg(obs):
    return SimpleNamespace(obs_shape=obs, task_dim=0, num_enc_layers=2,
                           enc_dim=8, latent_dim=8, simnorm_dim=4, num_channels=2)


def test_observation_types_do_not_leak_between_models():
    rgb = layers.enc(cfg({"rgb": (3, 64, 64)}))
    state = layers.enc(cfg({"state": (5,)}))
    assert list(rgb) == ["rgb"]
    assert list(state) == ["state"]
    assert not (set(map(id, rgb.parameters())) & set(map(id, state.parameters())))
    assert state["state"](torch.zeros(2, 5)).shape == (2, 8)


def test_later_model_does_not_inherit_earlier_state_encoder():
    first = layers.enc(cfg({"state": (5,)}))
    before = {key: value.clone() for key, value in first.state_dict().items()}
    second = layers.enc(cfg({"rgb": (3, 64, 64)}))
    assert list(second) == ["rgb"]
    assert set(first.state_dict()) == set(before)
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_explicit_output_mapping_still_collects_encoders():
    shared = {}
    state = layers.enc(cfg({"state": (5,)}), shared)
    combined = layers.enc(cfg({"rgb": (3, 64, 64)}), shared)
    assert set(combined) == {"state", "rgb"}
    assert combined["state"] is state["state"]
    assert set(shared) == {"state", "rgb"}
