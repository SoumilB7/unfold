# Component input label correction

The actual refiner entry overflowed long declared formal names. The component-only renderer now splits names at existing underscores (or a bounded character width), retains every character and all 13 identities, uses the existing multiline node height, and leaves evidence/IR unchanged. The full-pipeline branch is untouched.

The standalone replay verifies exact name reconstruction and existing node text/height metrics from the saved actual refiner IR. Native after.png was inspected: labels fit, all input lanes remain distinct. This is saved-input renderer evidence; separately pinned actual full-page replays are required for the trio, with a supplied-pipeline unchanged control. No model rebuild, pytest, or blessing here.
