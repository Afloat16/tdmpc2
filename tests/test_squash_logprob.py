import importlib.util
from decimal import Decimal, localcontext
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "tdmpc_math", Path(__file__).parents[1] / "tdmpc2/common/math.py")
math = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math)


def decimal_log_cosh(value):
    with localcontext() as ctx:
        ctx.prec = 100
        x = Decimal(str(value))
        return float(((x.exp() + (-x).exp()) / 2).ln())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tanh_change_of_variables_matches_high_precision_density(dtype):
    raw = torch.tensor([[-100., -20., -10., -1., 0., 1., 10., 20., 100.]], dtype=dtype)
    logp = torch.tensor([[0.3]], dtype=dtype)
    mu, action, result = math.squash(raw / 2, raw, logp)
    expected = logp.item() + 2 * sum(decimal_log_cosh(x) for x in raw[0].tolist())
    torch.testing.assert_close(result, torch.tensor([[expected]], dtype=dtype))
    torch.testing.assert_close(action, raw.tanh())
    torch.testing.assert_close(mu, (raw / 2).tanh())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_saturated_log_density_keeps_the_analytic_gradient(dtype):
    raw = torch.tensor([[-100., -20., -10., 0., 10., 20., 100.]], dtype=dtype, requires_grad=True)
    _, _, result = math.squash(torch.zeros_like(raw), raw, torch.zeros((1, 1), dtype=dtype))
    gradient, = torch.autograd.grad(result.sum(), raw)
    torch.testing.assert_close(gradient, 2 * raw.detach().tanh())


def test_density_and_gradient_agree_with_direct_formula_away_from_saturation():
    raw = torch.tensor([[-2., -0.5, 0., 0.5, 2.]], dtype=torch.float64, requires_grad=True)
    logp = torch.tensor([[0.4]], dtype=torch.float64, requires_grad=True)
    _, _, result = math.squash(raw, raw, logp)
    reference = logp - torch.log1p(-raw.tanh().square()).sum(-1, keepdim=True)
    torch.testing.assert_close(result, reference, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(torch.autograd.grad(result.sum(), raw, retain_graph=True)[0],
                               torch.autograd.grad(reference.sum(), raw)[0])
    torch.testing.assert_close(torch.autograd.grad(result.sum(), logp)[0], torch.ones_like(logp))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_saturation_is_finite(dtype):
    raw = torch.tensor([[20., -20.]], dtype=dtype, requires_grad=True)
    _, _, result = math.squash(raw, raw, torch.zeros((1, 1), dtype=dtype))
    expected = 4 * decimal_log_cosh(20)
    assert result.item() == pytest.approx(expected, abs=0.4, rel=0.01)
    gradient, = torch.autograd.grad(result.sum(), raw)
    assert gradient.tolist() == [[2., -2.]]
