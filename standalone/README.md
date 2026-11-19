# Standalone LIMO forward trial

This is a diagnostic implementation that communicates with ROS 1 directly;
it does not require the `rosclawd` service. `supervised_trial.py` defaults to
observation only. Its explicit `--execute` mode obtains one Decisions API
choice from a camera image and sensor state, and only streams forward commands
if that choice is `forward` and the local checks pass. It does not implement
turning, reverse motion, a persistent exploration loop, or autonomous mapping.

The ordinary trial has a 0.3-second relay motion deadline. The explicit
`--longer-forward` option selects a separate, bounded two-second forward trial.
Both require fresh camera, depth, lidar, odometry and base status, current ROS
and serial-ownership observations, an operator site report, and an explicit
floor-height and camera-rectification assumption. A dedicated relay publishes
velocity heartbeats and stops when its intent expires, its parent disappears,
the driver watchdog fails, or the selected motion deadline elapses. The relay
also receives the earliest absolute sensor, model and monitor expiry with each
motion intent. The relay and the candidate driver do not prove that the
physical chassis stopped.

The depth projection uses the supplied floor-height assumption and a rigid
camera transform; neither is independently calibrated by this code. The lidar
sees only part of the surroundings, and gaps, occlusions, the camera near
field and areas outside the sensors' view remain unknown. The camera TF
fallback only accepts a nearby latest transform matching a previously
observed timestamped transform. The first model decision is not renewed
during a single trial. A fresh full image is supplied for the model; later
safety snapshots continue checking projected depth and lidar without repeated
JPEG encoding or unused image-grid statistics. A newer depth frame must be
projected and checked; a newer lidar frame must have its own measured sampled
cross-section coverage before an older fused sample can still be used.

`continuous_policy.py` is an offline-only forward-intent prototype. It is not
wired to the real relay. `sensor_ipc.py` and `depth_ray_coverage.py` are also
offline prototypes, not a running sensor worker or permission to merge blind
regions. `relay_registry.py` provides optional, narrowly
validated cleanup for a dead relay's stale ROS registrations; its default mode
does not mutate the ROS graph. The candidate source changes for the base
watchdog and the YDLIDAR diagnostic driver are in `patches/`; applying a
patch does not install or start either driver. In the current development
session the candidate base binary was launched from `/tmp`, so its continued
use after reboot or process restart has not been established.

For offline verification, run the standalone tests with Python 2.7 in a ROS
Melodic environment, and run the repository's non-live suite with Python 3.
`threaded_relay_pty_check.py` exercises a real relay and candidate driver
against a fake serial chassis only inside its documented isolated namespace;
its results are not physical stopping or clearance tests.
