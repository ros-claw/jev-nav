#!/usr/bin/env python3
"""Exercise the candidate LIMO driver using only an isolated ROS master and PTY.

Run inside a network/device-isolated container with no /dev/ttyTHS1.  This
checks software serial frames, not chassis firmware or physical stopping.
"""

import json
import os
import pty
import select
import signal
import socket
import struct
import subprocess
import tempfile
import threading
import time
import tty
import xmlrpc.client
from pathlib import Path

PUBLISHER = r'''
import json, sys, time
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String

rospy.init_node("limo_pty_publisher", anonymous=False, disable_signals=True)
seen = []
def status(msg):
    fields = msg.data.split()
    if len(fields) == 9 and fields[0] == "v2":
        seen.append((time.time(), fields))
sub = rospy.Subscriber("/rosclaw_limo/command_watchdog", String, status, queue_size=20)
pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
deadline = time.time() + 8.0
while time.time() < deadline and pub.get_num_connections() == 0:
    time.sleep(0.05)
if not pub.get_num_connections():
    raise RuntimeError("fake driver did not subscribe")
for _ in range(16):
    pub.publish(Twist())
    time.sleep(0.05)
if not any(f[3] == "0" and f[4] == "1" and f[5] == "0" and
           f[6] == "/limo_pty_publisher" and int(f[2]) > 0 for _, f in seen):
    raise RuntimeError("fake zero prime or 2 Hz mode status failed")
message = Twist()
message.linear.x = 0.05
for _ in range(32):
    pub.publish(message)
    time.sleep(0.05)
stopped_at = time.time()
time.sleep(0.8)
latched = any(t >= stopped_at and f[5] == "1" for t, f in seen)
resumed_at = time.time()
for _ in range(6):
    pub.publish(message)
    time.sleep(0.05)
print(json.dumps({"stopped_at": stopped_at, "resumed_at": resumed_at,
                  "latched_after_loss": latched,
                  "latched_during_motion": any(f[5] == "1" and t < stopped_at for t, f in seen)}))
sys.stdout.flush()
pub.unregister()
rospy.signal_shutdown("fixture complete")
'''


def _free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _wait_master(uri, process):
    socket.setdefaulttimeout(0.25)
    end = time.monotonic() + 12.0
    while time.monotonic() < end:
        if process.poll() is not None:
            raise RuntimeError("isolated ROS master exited")
        try:
            if xmlrpc.client.ServerProxy(uri).getSystemState("/limo_pty_test")[0] == 1:
                return
        except (OSError, xmlrpc.client.Error):
            time.sleep(0.1)
    raise RuntimeError("isolated ROS master did not start")


def _terminate(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass
        process.wait(timeout=3)


def main():
    if Path("/dev/ttyTHS1").exists():
        raise RuntimeError("refusing fake test while real chassis device is visible")
    candidate = Path("/tmp/rosclaw-limo-base-check/devel/lib/limo_base/limo_base_node")
    if not candidate.is_file():
        raise RuntimeError("isolated candidate driver binary is missing")
    probe = socket.socket()
    probe.settimeout(0.2)
    try:
        if probe.connect_ex(("127.0.0.1", 11311)) == 0:
            raise RuntimeError("refusing fake test with an existing default ROS master")
    finally:
        probe.close()

    master_fd, slave_fd = pty.openpty()
    tty.setraw(slave_fd)
    pty_name = os.ttyname(slave_fd)
    if not pty_name.startswith("/dev/pts/"):
        raise RuntimeError("fake serial target is not a pseudoterminal")
    run_dir = Path(tempfile.mkdtemp(prefix="limo-pty-"))
    link = run_dir / "ttyFake"
    link.symlink_to(pty_name)
    if os.stat(str(link)).st_rdev != os.fstat(slave_fd).st_rdev:
        raise RuntimeError("fake serial target mismatch")
    port = _free_port()
    uri = "http://127.0.0.1:{}".format(port)
    env = os.environ.copy()
    env.update({"ROS_MASTER_URI": uri, "ROS_HOSTNAME": "127.0.0.1", "ROS_IP": "127.0.0.1",
                "ROS_HOME": str(run_dir / "ros"), "ROS_LOG_DIR": str(run_dir / "logs")})
    env["PYTHONPATH"] = ("/tmp/rosclaw-limo-base-check/devel/lib/python2.7/dist-packages:" +
                         env.get("PYTHONPATH", ""))
    frames = []
    configured = threading.Event()
    stopping = threading.Event()

    def fake_chassis():
        buffer = b""
        last_status = 0.0
        data = bytes([0, 1, 0, 122, 0, 0, 0, 0])
        status_frame = bytes([0x55, 0x0e, 0x02, 0x11]) + data + bytes([0, sum(data) & 0xff])
        while not stopping.is_set():
            ready, _, _ = select.select([master_fd], [], [], 0.02)
            if ready:
                try:
                    buffer += os.read(master_fd, 4096)
                except OSError:
                    break
                while len(buffer) >= 14:
                    if buffer[0] != 0x55 or buffer[1] != 0x0e:
                        buffer = buffer[1:]
                        continue
                    frame, buffer = buffer[:14], buffer[14:]
                    identifier = frame[2] * 256 + frame[3]
                    if identifier == 0x421:
                        configured.set()
                    elif identifier == 0x111:
                        frames.append((time.time(), struct.unpack(">h", frame[4:6])[0]))
            now = time.monotonic()
            if configured.is_set() and now - last_status >= 0.501:
                try:
                    os.write(master_fd, status_frame)
                except OSError:
                    break
                last_status = now

    master = None
    driver = None
    emulator = threading.Thread(target=fake_chassis, daemon=True)
    try:
        with open(os.devnull, "wb") as quiet:
            master = subprocess.Popen(["/opt/ros/melodic/bin/roscore", "-p", str(port)],
                                      env=env, stdout=quiet, stderr=quiet, start_new_session=True)
            _wait_master(uri, master)
            emulator.start()
            fake_port = os.path.relpath(str(link), "/dev")
            driver = subprocess.Popen([
                str(candidate), "__name:=limo_pty_base", "_port_name:=" + fake_port,
                "_command_watchdog_enabled:=true", "_command_watchdog_timeout_sec:=0.3",
                "_command_watchdog_mode_timeout_sec:=1.5", "_pub_odom_tf:=false",
            ], env=env, stdout=quiet, stderr=quiet, start_new_session=True)
            if not configured.wait(6.0):
                raise RuntimeError("candidate did not open fake serial port")
            result = subprocess.run(["/usr/bin/python2", "-u", "-c", PUBLISHER],
                                    env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True, timeout=15, check=True)
            evidence = json.loads(result.stdout.strip().splitlines()[-1])
        before = [speed for stamp, speed in frames if stamp < evidence["stopped_at"]]
        after = [speed for stamp, speed in frames if stamp >= evidence["stopped_at"] + 0.45]
        resumed = [speed for stamp, speed in frames if stamp >= evidence["resumed_at"]]
        if (not any(speed == 50 for speed in before) or not after or any(after)
                or not resumed or any(resumed) or not evidence["latched_after_loss"]
                or evidence["latched_during_motion"]):
            raise RuntimeError("fake command loss did not latch repeated zero frames")
        print(json.dumps({"result": "passed", "simulated_serial_only": True,
                          "driver_mode_report_period_sec": 0.501,
                          "motion_frames": sum(speed == 50 for speed in before),
                          "zero_frames_after_loss": len(after),
                          "nonzero_after_resume": sum(speed != 0 for speed in resumed),
                          "latched_after_loss": evidence["latched_after_loss"]}))
    finally:
        stopping.set()
        _terminate(driver)
        _terminate(master)
        if emulator.is_alive():
            emulator.join(timeout=1)
        os.close(master_fd)
        os.close(slave_fd)


if __name__ == "__main__":
    main()
