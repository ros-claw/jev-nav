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
cross-section coverage before an older fused sample can still be used. The
supervised runner defers the adapter's intermediate current-frame check to
avoid duplicate work; it still rechecks current sensors immediately before
each motion command. Intent validity also ends when historical rays required
for the accepted coverage expire. If snapshots cannot be produced and
checked fast enough, the relay stops instead of continuing an expired intent;
smooth real-robot operation has not yet been established.

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

`native_sensing_probe.py` is a separate, bounded observation experiment. It
creates a private TF topic and rolling grid using ROS native lidar and depth
processing, without publishing `/tf` or `/cmd_vel`. It does not need a saved
map. The optional `--test-synthetic-obstacle` injects a point only into the
private probe topic to check obstacle marking, then discards the grid. The
private TF topic derives `odom → base_link` from odometry, since the current
base configuration does not publish that transform. The standard obstacle
layer can retain old free cells. The optional experimental native layer resets
laser free space on each update and exposes the final grid and the scan's
acquisition timestamp in a single `FreshGrid` message. It requires a recent
scan and rejects unsafe laser and unknown-space configurations. A delayed scan
older than the last accepted scan is rejected, including while it is within
the age limit. After a clock or sensor timestamp reset, scans remain unavailable
until their timestamps catch up or the layer is restarted. The separate
depth layer retains marked obstacles, but its frame freshness and validity are
not bound to the grid. The depth height thresholds and camera rectification
remain unverified; unobserved space remains unknown. None of these grids is a
motion permit, and no real controller consumes them yet.

To build the experimental layer, create a catkin workspace with
`standalone/limo_fresh_layer` linked or copied into its `src` directory and
run `catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python2`. Pass that workspace to
`native_sensing_probe.py --fresh-layer-workspace PATH` to test live sensor
inputs without motion. `fresh_laser_integration_probe.py --workspace PATH`
checks behavior using private synthetic ROS topics; `--unsafe-inf-config` and
`--unsafe-unknown-config` test that invalid configurations are rejected. The
build is local to the chosen catkin workspace and is not installed by these
commands.
