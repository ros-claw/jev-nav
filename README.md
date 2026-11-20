# jev-nav

Experimental mapless LIMO navigation driven by an OpenAI Decisions model. This repository contains the navigation controller and experiments, separate from [limo-ros-mcp](https://github.com/ros-claw/limo-ros-mcp).

**Status:** autonomous exploration is not complete. The real robot has not demonstrated smooth continuous motion. The current supervised real-robot candidate can execute only a single bounded forward trial; turning, reverse and persistent exploration are not implemented. Laser coverage is front-facing, camera calibration and physical stopping are not independently verified. Do not interpret sensor coverage or simulated serial results as proof of physical clearance or braking.

## Contents

- `standalone/`: direct ROS 1 observation, decision integration, experimental bounded forward trial, relay, monitors and tests. See [standalone/README.md](standalone/README.md).
- `worker/`: model, scan fusion, and experimental mapless controller/relay workers.
- `patches/`: candidate patches for the upstream base and lidar drivers; these are not installed automatically.
- `experiments/`: earlier observation, calibration, simulation scripts and tests. Some historical experiments refer to local ROSClaw or MCP checkouts and may need their original environment.
- `docs/`: designs, diagnostic notes, and the original navigation idea.

The navigation code was extracted from `ros-claw/limo-ros-mcp` development commit `dce648e`; this is a new repository with its own history. The MCP repository remains responsible for MCP functionality, not this controller.

## Offline tests

ROS Melodic Python 2.7 and the dependencies available in that environment are required for the standalone suite:

```sh
/usr/bin/python2 -m pytest -q standalone
```

The optional scan accelerator is built locally using Cython; see the standalone README. Credentials and local build products must never be committed. Model code reads the credential file configured by `ROSCLAW_LIMO_DECISIONS_KEY_FILE`; the existing machine default is `~/rosclaw/jev/.env`.
