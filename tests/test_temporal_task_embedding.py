import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "tdmpc2"))
from common.world_model import WorldModel


def make_model():
    cfg = SimpleNamespace(multitask=True, tasks=["a", "b"], task_dim=4,
                          action_dims=[2, 1], obs="state", obs_shape={"state": (3,)},
                          num_enc_layers=2, enc_dim=8, latent_dim=8, simnorm_dim=4,
                          action_dim=2, mlp_dim=8, num_bins=3, episodic=False,
                          num_q=2, dropout=0., log_std_min=-5., log_std_max=2., tau=0.1)
    return WorldModel(cfg).eval()


@pytest.mark.parametrize("task", [1, torch.tensor([1])])
def test_single_task_broadcasts_across_time_and_batch(task):
    model = make_model()
    x = torch.randn(3, 5, 8, requires_grad=True)
    actual = model.task_emb(x, task)
    expected_embedding = model._task_emb(torch.tensor([1])).detach()[0]
    torch.testing.assert_close(actual[..., :8], x)
    for time in range(3):
        for batch in range(5):
            torch.testing.assert_close(actual[time, batch, 8:], expected_embedding)
    actual[..., :8].sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_per_batch_tasks_keep_their_identity_across_time():
    model = make_model()
    tasks = torch.tensor([0, 1, 0, 1, 1])
    x = torch.randn(3, 5, 8)
    actual = model.task_emb(x, tasks)
    embeddings = model._task_emb(tasks)
    for time in range(3):
        torch.testing.assert_close(actual[time, :, 8:], embeddings)


def test_native_temporal_dynamics_accepts_a_shared_task_id():
    model = make_model()
    z = torch.randn(3, 5, 8)
    action = torch.randn(3, 5, 2)
    actual = model.next(z, action, 1)
    reference = model.next(z, action, torch.ones(5, dtype=torch.long))
    torch.testing.assert_close(actual, reference)
    assert actual.shape == z.shape


def test_two_dimensional_shared_task_behavior_is_preserved():
    model = make_model()
    x = torch.randn(5, 8)
    torch.testing.assert_close(model.task_emb(x, 1),
                               model.task_emb(x, torch.ones(5, dtype=torch.long)))
