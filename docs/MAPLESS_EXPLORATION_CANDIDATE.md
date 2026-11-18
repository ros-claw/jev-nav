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

The candidate now reduces its commanded target when observed model latency
would consume the decision's pose window, and preserves the original image
pose as the decision anchor. It requires a fresh local sample before accepting
a motion preference, accounts for a bounded camera/odometry time offset, and
handles a model `stop` without requiring new sensor data. It requests zero
immediately. Because individual ROS `Twist` commands cannot be correlated to
serial writes, the relay latches and exits on any zero after published motion;
automatic resumption within the same task is not yet supported. A full
worker-to-relay pipe fails fast instead of blocking the control lock. A synthetic test
with repeated 2.23-second responses keeps moving until an obstacle is presented;
this does not prove real-time performance or continuous physical motion.

The time-offset and sensor-age corrections assume bounded robot speed; the
controller's acceleration/deceleration figures are not measured braking
guarantees. Turn targets scale linear and angular speeds together, but a smooth
transition from a previous command can initially follow a wider arc. Its swept
path, dynamic obstacles, and depth/camera calibration remain unverified. The
installed front-only scan can still yield unknown values for every sector in
the current controller, so these latency changes do not enable live motion.
