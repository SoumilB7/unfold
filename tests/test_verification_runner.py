"""Anti-vacuous laws for the parallel committed-tree receipt coordinator."""
from __future__ import annotations

import pathlib

import pytest

from scripts import verify_commit as verify


def test_runner_requires_an_explicit_affected_focus():
    with pytest.raises(SystemExit, match="at least one --focus"):
        verify.main(["--commit", "HEAD"])


def test_kernel_and_u2_lanes_are_nonempty_and_disjoint():
    assert verify.KERNEL_TESTS
    assert verify.U2_AUTHORITY_TESTS
    assert not (set(verify.KERNEL_TESTS) & set(verify.U2_AUTHORITY_TESTS))
    for path in (*verify.KERNEL_TESTS, *verify.U2_AUTHORITY_TESTS):
        assert pathlib.Path(path).is_file(), path


def test_preservation_partition_is_an_exact_existing_file():
    assert verify.PRESERVATION_TEST == "tests/test_preservation.py"
    assert pathlib.Path(verify.PRESERVATION_TEST).is_file()


def test_lane_cannot_pass_when_its_tree_changed(tmp_path):
    log = tmp_path / "lane.log"
    log.write_text("ok\n")
    result = verify.LaneResult(
        name="poison", returncode=0, duration=0.0,
        before="before", after="after",
        artifacts_before="same", artifacts_after="same", log_path=log)
    assert not result.passed


def test_lane_cannot_pass_on_a_nonzero_result(tmp_path):
    log = tmp_path / "lane.log"
    log.write_text("failed\n")
    result = verify.LaneResult(
        name="poison", returncode=1, duration=0.0,
        before="same", after="same",
        artifacts_before="same", artifacts_after="same", log_path=log)
    assert not result.passed


def test_lane_pass_requires_success_and_identical_fingerprint(tmp_path):
    log = tmp_path / "lane.log"
    log.write_text("passed\n")
    result = verify.LaneResult(
        name="control", returncode=0, duration=0.0,
        before="same", after="same",
        artifacts_before="same", artifacts_after="same", log_path=log)
    assert result.passed


def test_lane_cannot_pass_when_external_artifacts_changed(tmp_path):
    log = tmp_path / "lane.log"
    log.write_text("ok\n")
    result = verify.LaneResult(
        name="poison", returncode=0, duration=0.0,
        before="same", after="same",
        artifacts_before="before", artifacts_after="after", log_path=log)
    assert not result.passed


def test_serial_full_flag_is_explicit_and_defaults_off():
    parser = verify._parser()
    assert parser.parse_args(["--focus", "tests/test_denoiser.py"]).serial_full is False
    assert parser.parse_args([
        "--focus", "tests/test_denoiser.py", "--serial-full"
    ]).serial_full is True


def test_worker_plan_fills_ten_core_host_without_oversubscription():
    full, preservation, authority = verify._worker_plan(10)
    assert (full, preservation, authority) == (7, 1, 1)
    assert full + preservation + authority * 2 == 10


def test_worker_plan_scales_down_and_honors_override():
    assert verify._worker_plan(8) == (5, 1, 1)
    assert verify._worker_plan(4) == (1, 1, 1)
    assert verify._worker_plan(10, 5) == (5, 1, 1)


def test_single_worker_omits_xdist_overhead():
    assert verify._xdist_args(1, "loadfile") == ()
    assert verify._xdist_args(2, "loadfile") == (
        "-n", "2", "--dist", "loadfile")
