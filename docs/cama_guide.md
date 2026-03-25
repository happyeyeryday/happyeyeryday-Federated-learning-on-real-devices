# Real-Device CAMA Implementation Guide

## Purpose
This document explains how to implement a real-device version of `CAMA` in this repository.

The goal is not to reproduce every paper detail literally. The goal is:

1. keep the experiment comparable with the current real-device `HELCFL` baseline,
2. reuse the existing real-device communication and hetero-model interfaces,
3. preserve the same model family and data path,
4. make the implementation simple enough for Codex to finish locally and for lab debugging to remain manageable.

This guide is based on:

- the current real-device `HELCFL` implementation in this repo,
- the simulation-side `CAMA` implementation already added under `recovered_from_zip/code`,
- the original `CAMA` paper idea:
  - energy-aware client filtering,
  - fairness-aware participation,
  - dynamic model size allocation,
  - HeteroFL / ordered-dropout style heterogeneous aggregation.

## Core positioning

For this project, `CAMA` should be implemented as:

- a **real-device baseline for heterogeneous + energy-aware FL**,
- **not** as a DVFS-first baseline,
- **not** as a sleep / wake-up system,
- **not** as a separate model family.

So the real-device `CAMA` implementation should:

- still use `ResNet18`,
- still use `CIFAR10`,
- still use the same 10-device setup,
- still use the same socket-based server/client protocol,
- still use the same heterogeneous `model_rate` pipeline,
- but replace the server-side decision logic with a `CAMA`-style policy.

## Fixed experimental assumptions

Use the same physical setup as the real-device `HELCFL` baseline:

- dataset: `CIFAR10`
- model family: heterogeneous `ResNet18`
- number of devices: `10`
- device mix:
  - `CID 0-6`: `nano`
  - `CID 7-8`: `agx_xavier`
  - `CID 9`: `orinnanosuper`
- width levels:
  - `model_id 0 -> rate 0.125`
  - `model_id 1 -> rate 0.25`
  - `model_id 2 -> rate 0.5`
  - `model_id 3 -> rate 1.0`
- DVFS modes:
  - keep the same two native Jetson modes already abstracted in `utils/helcfl_real_profiles.py`
  - current placeholder mapping is low=`0`, high=`1`

This means the real-device `CAMA` baseline must remain directly comparable with:

- the current real-device `HELCFL`,
- the later real-device main method,
- the simulation-side baselines.

## Existing files to reuse

Use these files as the direct starting point.

### Main starting points

- [server_helcfl_real.py](/mnt/sda/zzr/code_unzipped/real_devices_repo/server_helcfl_real.py)
- [client_helcfl_real.py](/mnt/sda/zzr/code_unzipped/real_devices_repo/client_helcfl_real.py)
- [utils/helcfl_real_profiles.py](/mnt/sda/zzr/code_unzipped/real_devices_repo/utils/helcfl_real_profiles.py)

### Existing communication / FL utilities

- [utils/ConnectHandler_server.py](/mnt/sda/zzr/code_unzipped/real_devices_repo/utils/ConnectHandler_server.py)
- [utils/ConnectHandler_client.py](/mnt/sda/zzr/code_unzipped/real_devices_repo/utils/ConnectHandler_client.py)
- [utils/get_dataset.py](/mnt/sda/zzr/code_unzipped/real_devices_repo/utils/get_dataset.py)
- [models/hetero_model.py](/mnt/sda/zzr/code_unzipped/real_devices_repo/models/hetero_model.py)

### Simulation-side CAMA reference

- [CAMAScheduler.py](/mnt/sda/zzr/code_unzipped/recovered_from_zip/code/utils/CAMAScheduler.py)
- [cama_utils.py](/mnt/sda/zzr/code_unzipped/recovered_from_zip/code/utils/cama_utils.py)
- [run_cama.py](/mnt/sda/zzr/code_unzipped/recovered_from_zip/code/scripts/run_cama.py)

## What should be reused directly

### Reuse unchanged

These parts should stay as close as possible to the current `HELCFL` real-device code:

- server/client socket communication
- client registration by `CID`
- dataset loading
- heterogeneous `ResNet18(model_rate=...)`
- hetero distribute / hetero aggregate flow
- log format style with round timestamps
- `sudo nvpmodel -m ...` + `sudo jetson_clocks`

### Do not reuse

Do not reuse or reintroduce:

- wake / sleep
- low-battery shutdown logic
- BatteryManager-style device sleep control
- unnecessary old experiment entry scripts

## New files to implement

The clean implementation should add these files:

- `server_cama_real.py`
- `client_cama_real.py`
- `utils/cama_real_profiles.py`

If the implementer wants to minimize duplication, it is also acceptable to:

- copy `server_helcfl_real.py` -> `server_cama_real.py`
- copy `client_helcfl_real.py` -> `client_cama_real.py`
- copy `utils/helcfl_real_profiles.py` -> `utils/cama_real_profiles.py`

and then edit only the server-side scheduling logic.

## High-level design

## 1. Main difference from real HELCFL

The key difference is:

- `HELCFL`: mainly chooses clients, width, and a simple high/low DVFS policy.
- `CAMA`: mainly chooses clients and width based on energy/resource/fairness, and uses hetero-model training as the core mechanism.

For real devices, the easiest correct approximation is:

- keep the current real `HELCFL` client code structure,
- keep the same hetero-model training and aggregation,
- replace the server-side policy with a `CAMA` policy.

## 2. Main difference from simulation-side CAMA

The simulation `CAMA` implementation had to approximate model heterogeneity by:

- selecting parameter subsets,
- masking updates,
- simulating submodels through layer plans.

On real devices, **do not do that**.

Real devices already have a better and more faithful path:

- `models.hetero_model.resnet18(model_rate=...)`
- hetero distribute / aggregate already built for width-scaled models

So for real devices:

- use **real width-scaled submodels**,
- do **not** use the simulation-side parameter-mask approximation.

This is important for fairness and comparability.

## Server-side implementation guide

## 1. File layout

Start from `server_helcfl_real.py`.

Create:

- `server_cama_real.py`

Keep the following blocks almost unchanged:

- `init_weights`
- `hetero_distribute`
- `hetero_aggregate`
- `calibrate_bn`
- `evaluate`
- dataset loading
- connection setup
- message send / receive loop

The part that must change is:

- `select_clients(...)`
- policy profile lookup
- round plan construction

## 2. Add a dedicated profile file

Create:

- `utils/cama_real_profiles.py`

This file should isolate:

- `MODEL_ID_TO_RATE`
- `CID_TO_DEVICE_TYPE`
- `DEVICE_MAX_MODEL_ID`
- `DEVICE_DVFS_MODES`
- device capability ranking
- width assignment helper
- fairness penalty helper
- CAMA utility helper

Recommended content:

```python
MODEL_ID_TO_RATE = {
    0: 0.125,
    1: 0.25,
    2: 0.5,
    3: 1.0,
}

CID_TO_DEVICE_TYPE = {
    0: "nano",
    1: "nano",
    2: "nano",
    3: "nano",
    4: "nano",
    5: "nano",
    6: "nano",
    7: "agx_xavier",
    8: "agx_xavier",
    9: "orinnanosuper",
}

DEVICE_DVFS_MODES = {
    "nano": {"low": 0, "high": 1},
    "agx_xavier": {"low": 0, "high": 1},
    "orinnanosuper": {"low": 0, "high": 1},
}
```

Also define:

- `device_strength(device_type)`
- `max_model_id_for_device(device_type)`
- `model_rate_from_id(model_id)`
- `dvfs_mode_for(device_type, label)`

## 3. Recommended CAMA state to maintain

The server should maintain per-client state across rounds:

- `appearance_counter[cid]`
- `weighted_participation[cid]`
- `last_selected_round[cid]`
- `failure_counter[cid]`
- `last_model_id[cid]`
- `last_dvfs_label[cid]`
- optional:
  - `last_round_duration[cid]`
  - `last_upload_duration[cid]`

This is the minimum useful state for a paper-faithful approximation.

## 4. Recommended real-device CAMA selection logic

Implement CAMA in two steps:

### Step A: candidate filtering

Candidates must satisfy:

- connected and responsive
- not failed in the previous round beyond tolerance
- not on cooldown

Recommended cooldown rule:

- if a client was selected in the last `cooldown_rounds`, suppress it

### Step B: utility scoring

For each candidate, compute a simple utility score.

Recommended real-device approximation:

```text
utility(cid, model_id)
    = capability_score(device_type, model_id)
      / fairness_penalty(cid)
```

Where:

- `capability_score` is larger for:
  - stronger devices
  - larger supported model sizes
- `fairness_penalty` increases with:
  - repeated recent participation
  - weighted participation history

Suggested form:

```text
fairness_penalty = 1 + weighted_participation[cid]
utility = (model_rate * device_strength) / fairness_penalty
```

This is intentionally simple and robust.

For real devices, **do not** wait for accurate online energy estimation before implementing.
The power meter xlsx will be the final energy source of truth.

## 5. Width assignment policy

This is the most important part of real-device `CAMA`.

Recommended policy:

- `orinnanosuper`: prefer `model_id=3`
- `agx_xavier`: prefer `model_id=2`, sometimes `3`
- `nano`: prefer `model_id=1`, sometimes `0`

Then add participation-aware adjustment:

- if a client has participated too often recently, reduce model_id by 1
- if a device failed or lagged badly in the previous round, reduce model_id by 1

Suggested helper:

```python
def choose_model_id(device_type, appearance_count, failure_count):
    base = {
        "nano": 1,
        "agx_xavier": 2,
        "orinnanosuper": 3,
    }[device_type]
    if appearance_count >= 8:
        base -= 1
    if failure_count >= 1:
        base -= 1
    return max(0, base)
```

This is close to the spirit of the simulation-side `CAMA`, but uses actual hetero submodels instead of layer masks.

## 6. DVFS policy for real-device CAMA

Since this project is not DVFS-first, keep DVFS simple.

Recommended rule:

- `model_id >= 2` -> `high`
- `model_id <= 1` -> `low`
- optionally force the first-ranked selected device to `high`

This keeps DVFS consistent with:

- current real HELCFL code,
- simulation-side simplification,
- later comparability with the main method.

Do not over-engineer online slack-time estimation in the first version.

## 7. Round plan construction

Each selected client should receive a plan item containing:

- `cid`
- `device_type`
- `model_id`
- `model_rate`
- `dvfs_label`
- `dvfs_mode`

The server should log all of these each round.

Recommended log line:

```text
YYYY-MM-DD HH:MM:SS ROUND=12
selected=[0,7,9]
types=['nano','agx_xavier','orinnanosuper']
model_ids=[1,2,3]
model_rates=[0.25,0.5,1.0]
dvfs_modes=[0,1,1]
```

## Client-side implementation guide

## 1. File layout

Start from `client_helcfl_real.py`.

Create:

- `client_cama_real.py`

The client side should remain almost identical to `HELCFL`.

That is intentional.

For real-device `CAMA`, the algorithmic difference is overwhelmingly server-side.

## 2. What should remain the same

Keep unchanged:

- connection logic
- `CID` registration
- message receive loop
- `set_jetson_mode(...)`
- local training loop
- upload payload structure
- local log structure

## 3. What may change slightly

Add or keep fields in uploaded metadata:

- `cid`
- `round`
- `model_id`
- `model_rate`
- `dvfs_mode`
- `train_start_time`
- `train_end_time`
- `upload_start_time`
- `upload_end_time`
- `num_samples`
- optional:
  - `device_type`
  - `status`
  - `failure_reason`

This lets the server and the external power xlsx be aligned later.

## Message protocol

Use the same protocol style as current `HELCFL`.

## Server -> client message

Recommended payload:

```python
{
    "type": "train_round",
    "round": round_idx,
    "net": hetero_distribute(...),
    "idxs_list": dict_users[cid],
    "model_id": model_id,
    "model_rate": model_rate,
    "dvfs_label": dvfs_label,
    "dvfs_mode": dvfs_mode,
    "local_ep": args.local_ep,
    "lr": lr,
}
```

## Client -> server message

Recommended payload:

```python
{
    "type": "client_update",
    "cid": args.CID,
    "round": round_idx,
    "model_id": model_id,
    "model_rate": model_rate,
    "dvfs_mode": dvfs_mode,
    "net": local_model.state_dict(),
    "num_samples": len(dict_users[args.CID]),
    "mode_switch_start_time": ...,
    "mode_switch_end_time": ...,
    "train_start_time": ...,
    "train_end_time": ...,
    "upload_start_time": ...,
    "upload_end_time": ...,
}
```

## Logging requirements

This project will judge energy mainly by:

- program logs
- external power meter xlsx

So logs must be explicit and aligned.

## Server log requirements

Each round should log:

- timestamp
- round index
- selected client ids
- device types
- model ids
- model rates
- dvfs modes
- test accuracy
- best accuracy
- upload order
- round wall-clock duration

## Client log requirements

Each round should log:

- timestamp
- cid
- device type
- model_id
- model_rate
- dvfs_mode
- mode switch start/end
- train start/end
- upload start/end
- status

## Time format

Use the same format already agreed for lab alignment:

```text
2026-03-22 14:03:25
```

This matches the planned xlsx time parsing.

## Recommended implementation order

Implement in this order.

### Phase 1: mechanical duplication

1. copy `server_helcfl_real.py` -> `server_cama_real.py`
2. copy `client_helcfl_real.py` -> `client_cama_real.py`
3. copy `utils/helcfl_real_profiles.py` -> `utils/cama_real_profiles.py`

At this point:

- no new algorithm yet,
- only new file names and imports.

### Phase 2: profile isolation

Edit `utils/cama_real_profiles.py` to contain:

- width mapping
- device-type mapping
- dvfs mapping
- capability helpers
- fairness helpers
- `choose_model_id(...)`
- `choose_dvfs_label(...)`

### Phase 3: server-side CAMA policy

In `server_cama_real.py`:

- replace `select_clients(...)`
- keep the rest of the training loop unchanged

Get a minimal version running first.

### Phase 4: richer round metadata

Only after the first version runs:

- add `appearance_counter`
- add `failure_counter`
- add richer logs
- add summary JSON

### Phase 5: lab smoke

Run:

- server on host
- 1 Nano + 1 Xavier + 1 Orin if possible

Verify:

- mode switching works
- model width switching works
- uploads and aggregation work
- logs are complete

### Phase 6: full 10-device experiment

Only after 3-device smoke is stable:

- scale to 10 devices
- keep the same ResNet18 + CIFAR10

## What not to overdo in the first version

To keep local Codex implementation tractable, do **not** overbuild the following:

- no sleep / wake
- no online battery estimation
- no complex slack-time solver
- no exact paper-level energy utility reconstruction
- no simulation-style parameter-mask submodel system
- no dynamic reboot management if `nvpmodel` requests reboot

If a mode change fails:

- log it
- report `client_error`
- skip that client for the round

## Suggested acceptance criteria

Treat the implementation as acceptable if it satisfies all of these:

1. `server_cama_real.py` and `client_cama_real.py` can run end-to-end.
2. Devices can switch Jetson mode before local training.
3. Different clients can receive different `model_rate`.
4. Heterogeneous updates aggregate successfully.
5. Server logs each round with full plan metadata.
6. Client logs contain timestamps that can be aligned with xlsx.
7. The implementation remains comparable with real-device `HELCFL`.

## Comparison policy with HELCFL and main method

To keep the comparison fair:

- keep `ResNet18`
- keep `CIFAR10`
- keep the same 10-device mapping
- keep the same 4 width levels
- keep the same two DVFS mode abstraction
- keep the same logging style
- do not add wake/sleep only to one method

The comparison should isolate:

- `HELCFL`: scheduling-leaning hetero baseline
- `CAMA`: hetero-model-size + energy-aware baseline
- main method: your method

## Important difference from simulation-side CAMA

Simulation-side `CAMA` used:

- `build_submodel_plan(...)`
- parameter masks
- update masks

That was only necessary because the simulation repo did not have a native real hetero-model path for CAMA.

For real devices:

- do **not** copy that approximation
- use the actual hetero-model pipeline instead

This is cleaner, more faithful, and easier to debug.

## Concrete recommendation to local Codex

If you hand this to local Codex, tell it to do exactly this:

1. duplicate the current real HELCFL files into CAMA equivalents
2. keep client-side code nearly unchanged
3. move all new logic into:
   - `server_cama_real.py`
   - `utils/cama_real_profiles.py`
4. reuse:
   - `resnet18(model_rate=...)`
   - `hetero_distribute(...)`
   - `hetero_aggregate(...)`
5. implement only a simple, robust CAMA-style policy first
6. do a 1-3 device smoke before any 10-device run

## Final recommendation

Do not try to make real-device `CAMA` more complicated than real-device `HELCFL`.

The right target is:

- comparable,
- stable,
- easy to debug in the lab,
- rich enough to produce logs and power traces,
- close enough to the paper idea to serve as a valid baseline.

That is the correct engineering target for this repository.
