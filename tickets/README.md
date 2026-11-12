# Tickets

Project tracker for the clearance-horizon pipeline. Goal: measure the clearance horizon H and validate current equations. We start with minimal
and simplest equations, then build up complexity to increase accuracy and reduce uncertainty to be able to push horizon H further. Current experimentation is on PhysicalAI-AV clips with Alpamayo trajectories, to decide the inference-time budget.

## Structure

```
tickets/
  README.md                 ← this file: conventions + the tracker table
  _templates/               ← copy one of these to start a new epic or ticket
  E01-<epic-slug>/
    README.md               ← the epic (from _templates/epic.md)
    T001-<ticket-slug>.md   ← tickets in this epic
    T002-<ticket-slug>.md
  E02-<epic-slug>/
    ...
```

Two levels only: **epic folder → ticket file**. Anything smaller than a ticket is a
checklist item inside the ticket, not a nested folder.

## Ticket types

| Type | Use it for | Template | Done when |
|---|---|---|---|
| Epic | A milestone grouping related tickets (e.g. "Data layer") | `epic.md` | All its tickets are done and its exit criteria hold |
| Task | Building or changing code | `task.md` | Tests pass and the Rerun check looks right |
| Spike | Answering a question by investigation (docs, data, literature) | `spike.md` | The question has an answer with evidence |
| Decision | Choosing between options and recording why | `decision.md` | A choice is recorded with its consequences |
| Experiment | Running the pipeline to produce numbers | `experiment.md` | Results are saved, reproducible, and interpreted |
| Bug | Something that is wrong | `bug.md` | Fixed, with a test that would have caught it |

## Conventions

- **IDs** are global and never reused: epics `E01, E02, …`, tickets `T001, T002, …`
  (ticket numbers do not restart per epic). File name: `T007-short-slug.md`.
- **Status**: `todo` → `in-progress` → `review` → `done`. Also `blocked` (say on what) and
  `dropped` (say why). Update the ticket file *and* the tracker table below together.
- **Dependencies**: list ticket IDs in `Depends on`. A ticket is not started before its
  dependencies are `done`.
- **Verification is two-sided** wherever it applies: an automated test in `tests/`, and a
  Rerun check describing what we should see in the viewer.
- **Rerun conventions** (set up in the Rerun scaffolding ticket): one timeline `clip_time`
  in seconds; entity paths under `world/` (`world/ego`, `world/obstacles/<track_id>`,
  `world/reference/<source>`, …).
- **Evidence over assumption**: claims about the dataset cite the NVIDIA docs or a data
  check, recorded in the ticket's notes.
- **Closing a ticket**: tick every acceptance box, fill in "Outcome", set status `done`,
  update the tracker row, and reference the ticket ID in the commit message.

## Tracker

| ID | Type | Title | Epic | Status | Depends on |
|---|---|---|---|---|---|
| E01 | Epic | Raw data access layer | — | in-progress | none |
| T001 | Decision | Data scope for phase 1 | E01 | done | none |
| T002 | Task | General on-request feature fetcher | E01 | done | T001 |
| T003 | Task | Clip catalog: browse and select clips without downloading | E01 | done | T002 |
| T004 | Spike | `egomotion` vs `egomotion.offline` | E01 | done | T002 |
| E02 | Epic | Scene building | — | in-progress | E01 |
| T005 | Spike | Lidar format: columns, frame, timing, size | E02 | done | T002 |
| T006 | Spike | Is the rig origin the rear-axle midpoint? | E02 | done | T002 |
| T007 | Task | Scene builder: grid, ego, objects, frames, derived motion | E02 | done | T002, T006 |
| T008 | Task | Lidar observability and `in_scene` | E02 | done | T005, T007 |
| T009 | Spike | Derived ego motion vs online egomotion | E02 | done | T007 |
| T010 | Task | Stream a scene to Rerun | E02 | done | T007 |
| E03 | Epic | Horizon replay | — | in-progress | E02 |
| T011 | Task | Reference format + recorded-path reference | E03 | review | T007 |
| T012 | Task | Alpamayo trajectory import | E03 | todo | T011 |
| T013 | Task | Horizon config + baseline H solver | E03 | todo | T011 |
| T014 | Task | Horizon runner: one H per plan | E03 | todo | T008, T012, T013 |
| T015 | Task | Safety check: was H actually safe? | E03 | todo | T014 |
| T016 | Experiment | H: how long, and was it safe? | E03 | todo | T014, T015 |
| T017 | Task | Rerun view of the horizon | E03 | todo | T010, T014, T015 |
| T018 | Decision | What the truth check validates | E03 | done | none |
| T019 | Experiment | Do the object bounds hold on our scenes? | E03 | todo | T008, T009, T013 |
| T020 | Bug | Duplicate check is too strict for distant objects | E02 | done | T007 |
| T021 | Task | Causal lidar observability | E02 | done | T008, T009 |
| T022 | Task | Ego speed and acceleration in the viewer | E02 | done | T010 |
| T023 | Decision | What H means: once per plan on receipt, checked against the true future | E03 | done | T011 |
