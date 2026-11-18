# Mapless exploration candidate

`limo.explore` and the fixed worker in `worker/` are under development. The
corresponding ROSClaw worktree registers only a SHADOW executor. Do not install
the candidate as a REAL capability or treat a synthetic worker result as a
hardware navigation receipt.

The candidate sends Decisions model directions to a local controller, streams
bounded commands through a relay, and stops requesting motion on stale inputs,
relay heartbeat loss or cancellation. Relay publication acknowledgement only
means a ROS subscriber was connected; it does not identify the chassis driver
or prove the robot moved or stopped. The ROSClaw candidate therefore reports
its physical outcome as unverified.

Remaining work includes trustworthy three-dimensional and swept-footprint
coverage, verified depth and TF calibration, arbitration of command publishers,
measured control timing, independent chassis command-loss stopping, worker and
helper process lifetime management, and cancellation/startup race resolution
across the daemon. Synthetic tests do not establish these properties on LIMO.

One live setup used an X2L launch configuration publishing only the front 180°;
zero range values were frequent. The diagnostic source marks missing readings
unknown and never treats a missing rear sector as clear.
