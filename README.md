# jev-nav

Experimental mapless LIMO navigation driven by an OpenAI Decisions model. This repository contains the navigation controller and experiments, separate from [limo-ros-mcp](https://github.com/ros-claw/limo-ros-mcp).

`standalone/run_navigation.py` runs persistent navigation with asynchronous Decisions and live ROS sensing. Its default output is a private velocity preview. The supervised modes connect to the LIMO driver watchdog. The real robot has driven straight and turned with the fixed sensors in a supervised static room. A continuous full lap on hardware remains pending.

## Contents

- `standalone/`: persistent local trajectory tracking, ROS perception, Decisions integration, bounded physical diagnostics, relay, monitors and tests. See [standalone/README.md](standalone/README.md).
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
