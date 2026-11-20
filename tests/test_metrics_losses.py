import numpy as np
import pytest
import torch

from crackvit.evaluation.metrics import Confusion, bootstrap_confusions, paired_bootstrap
from crackvit.losses import BCEDiceLoss


def test_known_confusion_and_two_class_miou():
    c = Confusion()
    c.update(np.array([1, 1, 0, 0, 0]), np.array([1, 0, 1, 0, 0]))
    m = c.metrics()
    assert c.counts() == {"tp": 1, "fp": 1, "fn": 1, "tn": 2}
    assert m["precision"] == 0.5 and m["f1"] == 0.5
    assert m["crack_iou"] == pytest.approx(1 / 3)
    assert m["miou"] == pytest.approx((1 / 3 + 1 / 2) / 2)


def test_absent_class_policy():
    m = Confusion(tn=10).metrics()
    assert m["f1"] is None and m["precision"] is None
    assert m["miou"] == 1 and m["accuracy"] == 1
    with pytest.raises(ValueError):
        Confusion().metrics()


def test_shape_mismatch():
    with pytest.raises(ValueError):
        Confusion().update([1, 0], [1])


def test_micro_identity():
    for c in [Confusion(7, 3, 2, 80), Confusion(1, 8, 7, 1)]:
        m = c.metrics()
        assert m["crack_iou"] == pytest.approx(m["f1"] / (2 - m["f1"]))


def test_loss_perfect_wrong_and_gradients():
    y = torch.tensor([[[[0., 1.], [1., 0.]]]])
    good, wrong = (2 * y - 1) * 30, (1 - 2 * y) * 30
    loss = BCEDiceLoss()
    assert loss(good, y) < 1e-5
    assert loss(wrong, y) > 10
    logits = torch.zeros_like(y, requires_grad=True)
    loss(logits, y).backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0, 0, 0] > 0 and logits.grad[0, 0, 0, 1] < 0


@pytest.mark.parametrize("target", [0., 1.])
def test_extreme_loss_is_finite(target):
    x = torch.full((1, 1, 8, 8), 1000., requires_grad=True)
    y = torch.full_like(x, target)
    loss = BCEDiceLoss()(x, y)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()


def test_loss_shape_validation():
    with pytest.raises(ValueError):
        BCEDiceLoss()(torch.zeros(2, 3), torch.zeros(2, 3))


def test_bootstrap_repeatable_and_paired():
    rows = [{"sample_id": str(i), "group": str(i), "tp": i + 1, "fp": 2, "fn": 1, "tn": 20,
             "image_sha256": str(i), "mask_sha256": str(i)} for i in range(4)]
    assert bootstrap_confusions(rows, 20) == bootstrap_confusions(rows, 20)
    assert paired_bootstrap(rows, rows, 20)["ci95"] == [0, 0]
    with pytest.raises(ValueError):
        paired_bootstrap(rows, rows[:-1])
