# Supported scope and honesty boundary

## Implemented domain adapters

The working tree contains first-class adapters for:

- transformer-style model configurations;
- diffusion pipelines, diffusion transformers, UNets, and VAEs;
- transformer-hosted vision, audio, conditioning, and fusion paths;
- secondary stacks, heterogeneous layer schedules, MoE, and expanded attention
  and FFN drills where evidence readers support them.

The implementation locations are the transformer and diffusor packages under
`unfold-pkg/model_unfolder/adapters/`, plus the modality builders under
`adapters/transformer/special_parts/modalities/`.

## What “support” means

Support is not a family-name row. A mechanism is supported when the system can:

1. acquire its relevant config/source evidence;
2. bind it to the correct owner;
3. interpret it into registered facts or geometry;
4. project it consistently;
5. distinguish positive, negative, ambiguous, and missing-source cases;
6. pass mechanism counterexamples and preservation controls.

Support has three states:

- **supported:** the complete promised mechanism detail is evidenced and
  projected;
- **honest partial:** established structure is shown and unresolved detail is
  explicit;
- **unsupported:** the source or mechanism lies outside the declared product
  boundary and receives no substitute architecture.

## Unseen mechanisms

An unseen model may render partially if its mechanisms are already understood.
A genuinely unseen mechanism must stay visible as unknown or raise a structured
finding. It must not inherit the closest known architecture.

## Coverage claims

Coverage is measured against explicitly supported upstream versions and a
published architecture-class corpus. It is not measured against every Hub
repository, because repositories can contain duplicated checkpoints, arbitrary
remote code, inaccessible source, or genuinely new mechanisms.

Current model coverage belongs in tested corpus manifests and release receipts.
Architectural coverage belongs in mechanism tests and the fact registry. A new
checkpoint using an already-supported class is not a new support project.
