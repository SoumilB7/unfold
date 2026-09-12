# S7 Linux-workflow closure poison

`tests/test_s7_artifacts.py::test_quality_workflow_installs_renderer_before_example_check`
requires exactly one `librsvg2-bin` install and exactly one release-example check,
with installation ordered first. Removing, duplicating, renaming, or moving the
dependency after the check turns the gate red.
