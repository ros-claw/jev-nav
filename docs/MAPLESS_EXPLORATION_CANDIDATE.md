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

The operator reports that the installed lidar sees only part of the forward
field and cannot observe behind the robot. One live scan had a ±90° message
range with frequent zero values. The model receives measured front-direction
rays and missing-angle statistics; missing or rear observations remain unknown.
Changing the ROS scan angle does not create physical sensor coverage.
