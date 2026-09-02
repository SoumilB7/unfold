# Documentation boundary

## Current authority

The active chapters under `z-docs/` describe the product being built, the
architecture laws that govern it, the implementation state verified against the
current codebase, and the finite route to release.

The authority order is:

1. product doctrine and supported scope;
2. architecture and evidence contracts;
3. current implementation state;
4. development and verification protocols;
5. detailed implementation worklists.

A detailed worklist cannot override an architecture law or promote unfinished
work into current state.

## Historical archive

`z-docs/stale/` is a quarantined historical archive. It preserves discarded
plans, session logs, intermediate naming schemes, old test receipts, and model
investigations so useful evidence is not destroyed.

Archive content is never an implementation instruction and never proves the
current state. Before reusing anything from it, re-establish the claim from the
current code, tests, and active chapters.

## What belongs in active documentation

- a durable product or architecture law;
- a code-verified description of current behavior;
- a precisely named unfinished seam;
- a verification contract or release boundary;
- a finite work item with an observable achieved output.

## What does not belong in active documentation

- dated execution labels, step-number histories, or session narration;
- claims whose only evidence is an old commit or test count;
- model anecdotes presented as general architecture;
- plans written as though they were already implemented;
- overlapping status trackers;
- temporary agent instructions;
- line-number references that are expected to drift.

## Update rule

When implementation changes:

1. update the relevant architecture chapter if the contract changed;
2. update `07-current-state/implementation-state.md` with the verified boundary;
3. update the delivery path only if the remaining product work changed;
4. move replaced narrative to the archive or delete it when it has no historical
   value;
5. validate active links and terminology.

Active documentation should always answer three questions without reconstructing
an old implementation diary: what the product is, what the code proves today,
and what finite work remains.
