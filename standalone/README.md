# Standalone LIMO forward trial

This is a diagnostic implementation that communicates with ROS 1 directly;
it does not require the `rosclawd` service. `supervised_trial.py` defaults to
observation only. Its explicit `--execute` mode obtains one Decisions API
choice from a camera image and sensor state, and only streams forward commands
if that choice is `forward` and the local checks pass. It does not implement
turning, reverse motion, a persistent exploration loop, or autonomous mapping.

The ordinary trial has a 0.3-second relay motion deadline. The explicit
`--longer-forward` option selects a separate, bounded two-second forward trial.
Adding `--single-threaded` to that option selects an experimental synchronous
sampling loop with the same motion bounds and per-command freshness checks.
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

The longer supervised trial optionally refreshes observation history while
the model request is pending; that worker cannot command motion and must exit
before the trial resumes observation. A worker that fails to exit prevents
motion. Expensive current-sensor checks use a captured view rather than holding
the ROS callback lock. They recheck the latest odometry, state, transforms and
sensor generations before returning, and cap any newer-scan verdict by the
expiry of the historical rays on which it depended. Unchecked later scans or
depth frames still reject the result.

The optional Cython scan accelerator can be built for ROS Melodic Python 2
with `/usr/bin/python2 standalone/build_scan_accel.py build_ext --build-lib
standalone/build/py2 --build-temp standalone/build/temp-py2`. Start a new
controller process after building. The binary is local and excluded from git;
without it, the existing checked Python implementation is used. Geometric
threshold ties and inputs outside the accelerated format also use that
implementation. This optimization does not establish physical clearance.
The accelerator also projects new raw scans and computes the expiry of an
already checked forward cross-section. Unusual input types and boundary ties
fall back to the scalar checks. The expiry accelerator bounds its work to
2,048 rays and eight distinct acquisition timestamps; larger histories use
the scalar implementation. These changes do not extend sensor freshness.
Fusion coordinate transforms use batches of up to 1,536 rays per native call.
A single record beyond that bound or a declined batch triggers the checked
scalar path for the whole snapshot, including its historical obstacles.
For the longer supervised trial, observation production and final validation
are serialized with a bounded wait. A failed observation or assessment blocks
new proofs; hard failures latch for the trial. The final proof and relay pipe
write share the same processing guard. Sensor callbacks remain independent,
and the relay still enforces each absolute intent expiry.
The guarded longer trial starts its sampler while the relay is still at zero,
waits for an assessed sample and a sufficiently long first intent, and retries
temporarily incomplete forward coverage only while stationary. Once motion
starts, the same incomplete coverage latches the trial to stop.

The November 20 perception and renewal work adds **read-only diagnostics**:
`sensing_timeline.py` and `live_independent_probe.py` record acquisition,
queueing and processing times, while `multiprocess_probe.py` runs independent
scan and depth workers and an in-memory `RelayGate`. The latter does not
publish velocity or check base status, odometry continuity or clearance.
Its `timeline_complete`/`mux_writer_complete` field refers to the mux writer
only; child shutdown events can be present only in the child timelines.
The immutable cache and local cell evidence modules are experimental and are
not connected to a motion controller. Sampled centers do not prove continuous
3-D coverage or a swept-footprint clearance. Full scan/depth coverage and
independent fault propagation remain open problems.

The supervised runner's opt-in `--longer-forward --prepared-forward` mode
performs projection and full sensor validation in the existing background
worker. Before submitting, the controller checks the original proof expiry,
current sensor generations, odometry, base state and command monitors. It
withholds commands if a new frame supersedes the prepared proof. The first live
tests of this mode completed at zero because proofs were repeatedly superseded.
See the [November 20 trial records](../docs/2026-11-20-perception-renewal-progress.md).

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
