# Real-load perception and command renewal implementation plan

> **For agentic workers:** Execute task-by-task using superpowers:executing-plans. Record measured results and do not treat simulated commands as physical motion.

**Goal:** Separate sensor processing from control renewal, demonstrate measured motion, then add repeated model decisions and observed turning.

**Architecture:** A bounded producer processes timestamped sensor observations once into immutable verdicts; a small controller consumes only completed evidence and live stop/fault state, while a separate relay and driver retain command deadlines. Record source, processing, decision, intent, ROS output, odometry and stop events in a bounded, replayable timeline. Preserve old supervised trials as diagnostic tools.

**Tech stack:** ROS 1 Melodic / Python 2.7, existing observation and relay modules, offline tests and isolated PTY with the real relay gate.

**Spec:** `~/rosclaw/jev/limo_nav_diagnosis_and_codex_handoff_2026-11-20.md` and the user's stage ordering.

## Global constraints

- Never extend acquisition timestamps or resurrect an expired command; missing and unobserved regions stay unknown.
- Bound sensor backlog and fail to zero on expired evidence, obstacle, broken producer, odometry/clock fault or relay/driver failure.
- No reverse or turn until the swept path, body mode and full command chain have been checked.
- Never report physical motion or stop from fake serial, ROS messages, or odometry alone.
- No model credentials, image payloads, or authorization headers in logs.

## Review focus

- A newer sensor frame arriving while a previous valid result is processed must not cause unbounded retries, but newly observed hazards and stale/backlogged processing must prevent renewal.
- The producer failing, hanging, reordering timestamps or advancing epochs must stop renewal and never refresh a stale deadline.
- Loss of controller, stalled relay, command expiry and communication loss must still stop independently.
- Odom discontinuities, swept-footprint unknown space and uncalibrated depth must not be represented as verified clearance.
- Every elapsed duration and trajectory must distinguish acquisition, processing, command, observed odometry and independently recorded physical evidence.

## Tasks

### 1. Reproduce and instrument real sensing without actuation

- [x] Reproduce the supplied 96 ms + 98 ms timing case using the real relay gate.
- [x] Add acquisition/processing/coverage and accelerator diagnostics with a bounded timeline; test timestamp fidelity, redaction and nonblocking behavior.
- [x] Measure real ROS observations over a sustained interval without publishing velocity; record accepted windows, latencies and specific failure causes.

### 2. Separate perception production and command renewal

- [x] Test an immutable processed-verdict cache under 96 ms sensing, faster arrivals, obstacle events, missing frames, producer failure and clock/epoch changes. This exposed a remaining renewal gap and does not prove real-time operation.
- [ ] Implement one producer and lightweight bounded consumer with an explicit backlog and source-age policy; preserve independent hazard/stop handling and source timestamps.
- [ ] Test the entire renewal path against real RelayGate and the isolated ROS/PTY relay under sensor delays; check wire expiry and stopping during stalled producer.
- [x] Re-run read-only on-device sensing with diagnostic producers and save a timeline. They are not connected to an actuator.

### 3. Measure real movement under real sensing

- [ ] Verify current driver/serial/ROS graph and observe current scene; select only geometrically supported forward motion with explicitly bounded parameters.
- [ ] Record full command, obstacle, decision, sensor and odometry timelines plus trajectory; distinguish odometry from independently measured motion.
- [ ] Analyze actual displacement and stop reason against stage A and B goals; do not advance if the execution chain cannot renew under the real load.

### 4. Repeat decisions and actual turning

- [ ] Implement a session with explicit WAIT and latched FAULT states, decision/scene/candidate IDs and fresh observation per action.
- [ ] Test bounded angular commands through all layers and independently check swept footprint using measured sensor visibility before enabling a candidate.
- [ ] Demonstrate multiple Decisions choices and actual change in measured heading/trajectory; retain replayable evidence and report precise limitations.
