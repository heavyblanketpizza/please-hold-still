"""The training loop. End-to-end runs need VJEPA2_REPO (see tests/test_vjepa.py)."""

import csv
import os

import pytest
import torch
from conftest import make_phantom

from please_hold_still.data import preprocess as pp
from please_hold_still.training import TrainConfig, lr_at, train

VJEPA2_REPO = os.environ.get("VJEPA2_REPO")
needs_repo = pytest.mark.skipif(not VJEPA2_REPO, reason="set VJEPA2_REPO to a vjepa2 clone")


def test_lr_schedule_warms_up_then_decays():
    cfg = TrainConfig(steps=1000, warmup_steps=100, lr=1e-4, final_lr_fraction=0.1)
    lrs = [lr_at(s, cfg) for s in range(1000)]
    assert lrs[0] == pytest.approx(1e-6)
    assert lrs[99] == pytest.approx(1e-4)  # peak at the end of warm-up
    assert all(a <= b for a, b in zip(lrs[:100], lrs[1:100], strict=False))  # rising
    assert all(a >= b for a, b in zip(lrs[100:], lrs[101:], strict=False))  # then falling
    assert lrs[-1] == pytest.approx(1e-5, rel=0.01)  # ends at final_lr_fraction * lr


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("root")
    raw = root / "raw"
    rows = []
    for i in range(3):  # ds000000..2 all hash to the train split
        f = make_phantom(raw, f"v{i}", seed=i)
        rows.append(
            {
                "id": f"ds00000{i}__v{i}",
                "dataset_id": f"ds00000{i}",
                "modality": "T1w",
                "image": f["image"].name,
                "anat_mask": f["anat"].name,
                "anon_mask": "",
            }
        )
    assert all(
        s == "ok" for _, s, _ in pp.process_many(rows, raw, root / "processed", progress=False)
    )
    return root


def tiny_config(**kw):
    base = dict(
        run_name="t",
        steps=4,
        batch_size=2,
        num_frames=4,
        size=64,
        num_workers=0,
        warmup_steps=2,
        predictor_only_steps=2,
        save_every=2,
        log_every=1,
    )
    return TrainConfig(**(base | kw))


def run(root, cfg):
    return train(
        cfg,
        root,
        hub_repo=VJEPA2_REPO,
        random_weights=True,
        device="cpu",
        progress_print=lambda *_: None,
    )


@needs_repo
def test_training_saves_logs_and_resumes(data_root):
    cfg = tiny_config(run_name="resume")
    assert run(data_root, cfg) == {
        "status": "finished",
        "step": 4,
        "run_dir": str(data_root / "runs" / "resume"),
    }
    run_dir = data_root / "runs" / "resume"
    for name in (
        "config.json",
        "log.csv",
        "checkpoint_last.pt",
        "encoder_last.pt",
        "encoder_step2.pt",
        "encoder_step4.pt",
    ):
        assert (run_dir / name).is_file(), name
    with open(run_dir / "log.csv") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["step"]) for r in rows] == [1, 2, 3, 4]
    assert [r["phase"] for r in rows] == ["predictor-only"] * 2 + ["full"] * 2

    # Asking for more steps continues where it stopped.
    result = run(data_root, tiny_config(run_name="resume", steps=6))
    assert result["step"] == 6
    with open(run_dir / "log.csv") as f:
        assert [int(r["step"]) for r in csv.DictReader(f)] == [1, 2, 3, 4, 5, 6]


@needs_repo
def test_encoder_snapshot_loads_with_the_standard_loader(data_root):
    from please_hold_still.models.vjepa import load_vjepa2_1_encoder

    run(data_root, tiny_config(run_name="snap", steps=2))
    path = data_root / "runs" / "snap" / "encoder_last.pt"
    encoder = load_vjepa2_1_encoder(checkpoint=path, device="cpu", hub_repo=VJEPA2_REPO)
    saved = torch.load(path, weights_only=True)["ema_encoder"]
    for name, tensor in encoder.state_dict().items():
        assert torch.equal(tensor, saved[name]), name


@needs_repo
def test_student_is_frozen_during_predictor_warmup(data_root):
    """During warm-up only the predictor learns; the teacher stays equal to the student."""
    run(data_root, tiny_config(run_name="warm", steps=2, predictor_only_steps=2))
    state = torch.load(data_root / "runs" / "warm" / "checkpoint_last.pt", weights_only=True)
    model = state["model"]
    student = {k[len("student.") :]: v for k, v in model.items() if k.startswith("student.")}
    teacher = {k[len("teacher.") :]: v for k, v in model.items() if k.startswith("teacher.")}
    assert all(torch.equal(student[k], teacher[k]) for k in student)
    assert state["optimizer"]["state"], "the predictor should have optimiser state"


def test_disk_estimate_counts_checkpoints_and_snapshots():
    from types import SimpleNamespace

    from torch import nn

    from please_hold_still.training import disk_needed

    student, teacher, predictor = nn.Linear(10, 10), nn.Linear(10, 10), nn.Linear(10, 5)
    model = nn.Module()
    model.student, model.teacher, model.predictor = student, teacher, predictor
    cfg = SimpleNamespace(steps=100, save_every=50)
    # params: student 110, teacher 110, predictor 55 -> all 275, trainable 165
    checkpoint = 4 * (275 + 2 * 165)
    assert disk_needed(model, cfg) == 2 * checkpoint + 4 * 110 * (100 // 50 + 2)
