"""Anti-vacuous laws for the parallel committed-tree receipt coordinator."""
from __future__ import annotations

import json
import pathlib

import pytest

from scripts import verify_commit as verify
from scripts import pytest_file_bracket as file_bracket


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


def test_parallel_full_is_the_exact_remainder_not_duplicate_authority_work():
    base = ("python", "-m", "pytest", "-q")
    command = verify._partitioned_full_command(base, 3)
    assert command[0] == verify.sys.executable
    assert command[1].endswith("scripts/pytest_file_bracket.py")
    assert command[2:4] == ("--workers", "3")
    ignored = {
        command[index + 1]
        for index, item in enumerate(command) if item == "--ignore"
    }
    assert ignored == {verify.PRESERVATION_TEST, *verify.U2_AUTHORITY_TESTS}
    assert command[-1] == "tests"


def test_coordinator_fingerprint_covers_both_runner_files(monkeypatch, tmp_path):
    coordinator = tmp_path / "verify_commit.py"
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    bracket = scripts / "pytest_file_bracket.py"
    coordinator.write_text("coordinator-v1\n")
    bracket.write_text("bracket-v1\n")
    monkeypatch.setattr(verify, "__file__", str(coordinator))
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    before = verify._coordinator_fingerprint()
    bracket.write_text("bracket-v2\n")
    assert verify._coordinator_fingerprint() != before


def test_batch_bracket_partitions_every_node_once_in_stable_file_order():
    items = (
        file_bracket.CollectedItem("tests/test_b.py::test_2", "tests/test_b.py"),
        file_bracket.CollectedItem("tests/test_a.py::test_1", "tests/test_a.py"),
        file_bracket.CollectedItem("tests/test_b.py::test_1", "tests/test_b.py"),
    )
    assert file_bracket._partition(items) == {
        "tests/test_a.py": ("tests/test_a.py::test_1",),
        "tests/test_b.py": (
            "tests/test_b.py::test_2", "tests/test_b.py::test_1"),
    }


def test_batch_bracket_rejects_duplicate_nodes_and_empty_collection():
    duplicate = file_bracket.CollectedItem(
        "tests/test_a.py::test_1", "tests/test_a.py")
    with pytest.raises(ValueError, match="duplicate collected node"):
        file_bracket._partition((duplicate, duplicate))
    with pytest.raises(ValueError, match="empty"):
        file_bracket._partition(())


def test_batch_bracket_balances_collected_weight_and_bounds_process_size():
    files = {
        "tests/test_a.py": tuple(
            f"tests/test_a.py::test_{index}" for index in range(100)),
        "tests/test_b.py": ("tests/test_b.py::test_value",),
        "tests/test_c.py": ("tests/test_c.py::test_value",),
        "tests/test_d.py": tuple(
            f"tests/test_d.py::test_{index}" for index in range(90)),
        "tests/test_e.py": ("tests/test_e.py::test_value",),
        "tests/test_f.py": ("tests/test_f.py::test_value",),
    }
    schedule = file_bracket._batch_schedule(files, 2)
    assert len(schedule) == 3
    assert all(1 <= len(row["sources"]) <= 2
               for row in schedule.values())
    weights = tuple(len(row["items"]) for row in schedule.values())
    # The old alphabetic round-robin paired the two heavy files (190 items).
    # Exact collection weights keep the heaviest bounded batch at 101.
    assert max(weights) == 101
    assert max(weights) < 190
    assert {item["nodeid"] for row in schedule.values()
            for item in row["items"]} == {
                nodeid for nodeids in files.values() for nodeid in nodeids}


def test_batch_bracket_weighted_schedule_is_deterministic():
    files = {
        "tests/test_b.py": tuple(
            f"tests/test_b.py::test_{index}" for index in range(3)),
        "tests/test_a.py": tuple(
            f"tests/test_a.py::test_{index}" for index in range(3)),
        "tests/test_c.py": ("tests/test_c.py::test_value",),
    }
    first = file_bracket._batch_schedule(files, 2)
    second = file_bracket._batch_schedule(dict(reversed(tuple(files.items()))), 2)
    assert first == second
    assert first["batch-000"]["sources"][0] == "tests/test_a.py"
    assert first["batch-001"]["sources"][0] == "tests/test_b.py"


def test_batch_bracket_requires_positive_worker_and_batch_bounds():
    with pytest.raises(SystemExit, match="positive"):
        file_bracket.main(["--workers", "0", "tests"])
    with pytest.raises(SystemExit, match="positive"):
        file_bracket.main(["--files-per-process", "0", "tests"])


def _write_schedule(path, nodeid, *, source="tests/test_a.py"):
    path.write_text(json.dumps({
        "batch-000": {
            "sources": ["tests/test_a.py"],
            "items": [{"nodeid": nodeid, "source": source}],
        },
    }))


def test_batch_bracket_blocks_when_batch_local_collection_differs(
        monkeypatch, tmp_path):
    schedule = tmp_path / "schedule.json"
    _write_schedule(schedule, "tests/test_a.py::test_expected")

    def collect_different(_args, plugins):
        plugins[0].items = (
            file_bracket.CollectedItem(
                "tests/test_a.py::test_other", "tests/test_a.py"),
        )
        return file_bracket.pytest.ExitCode.OK

    monkeypatch.setattr(file_bracket.pytest, "main", collect_different)
    assert file_bracket._run_batch_mode("batch-000", schedule) == 5


def test_batch_bracket_rejects_right_nodeid_from_wrong_source(
        monkeypatch, tmp_path):
    nodeid = "tests/test_a.py::test_expected"
    schedule = tmp_path / "schedule.json"
    _write_schedule(schedule, nodeid, source="tests/test_expected.py")

    def collect_right_node_wrong_source(_args, plugins):
        plugins[0].items = (
            file_bracket.CollectedItem(nodeid, "tests/test_actual.py"),
        )
        return file_bracket.pytest.ExitCode.OK

    monkeypatch.setattr(
        file_bracket.pytest, "main", collect_right_node_wrong_source)
    assert file_bracket._run_batch_mode("batch-000", schedule) == 5


def test_batch_bracket_propagates_test_failure_after_exact_collection(
        monkeypatch, tmp_path):
    nodeid = "tests/test_a.py::test_expected"
    schedule = tmp_path / "schedule.json"
    _write_schedule(schedule, nodeid)

    def collect_exact_but_fail(_args, plugins):
        plugins[0].items = (
            file_bracket.CollectedItem(nodeid, "tests/test_a.py"),
        )
        return file_bracket.pytest.ExitCode.TESTS_FAILED

    monkeypatch.setattr(file_bracket.pytest, "main", collect_exact_but_fail)
    assert file_bracket._run_batch_mode("batch-000", schedule) == 1


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
    full, preservation, authority, focused = verify._worker_plan(10)
    assert (full, preservation, authority, focused) == (6, 4, 3, 1)
    assert full + preservation == 10
    assert authority + focused <= 10


def test_worker_plan_scales_down_and_honors_override():
    assert verify._worker_plan(12) == (8, 4, 4, 1)
    assert verify._worker_plan(8) == (5, 3, 2, 1)
    assert verify._worker_plan(4) == (3, 1, 2, 1)
    assert verify._worker_plan(10, 5) == (5, 4, 3, 1)


def test_single_worker_omits_xdist_overhead():
    assert verify._xdist_args(1, "loadfile") == ()
    assert verify._xdist_args(2, "loadfile") == (
        "-n", "2", "--dist", "loadfile")
    # Authority tests include one much slower file; test-level scheduling keeps
    # one worker from becoming the long pole after the other files finish.
    assert verify._xdist_args(3, "load") == (
        "-n", "3", "--dist", "load")
