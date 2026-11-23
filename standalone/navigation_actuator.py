"""Own the navigation relay and enforce local output authority."""
from __future__ import division

import json
import threading

from navigation import Command, observation_deadline
from relay import safe_velocity


class ActuationError(RuntimeError):
    pass


class NavigationHeartbeat(object):
    """Keep the relay fed with zero while local geometry updates are late."""
    EXPIRED = Command(0., 0., 'stop', 'navigation_sample_expired')

    def __init__(self, actuator, clock):
        self.actuator, self.clock = actuator, clock
        self.latest = self.EXPIRED, 0.
        self._wake = threading.Event()
        self._closed = threading.Event()
        self._worker = threading.Thread(target=self._run, name='navigation-heartbeat')
        self._worker.daemon = True
        self.fault = None

    def start(self):
        self._worker.start()

    def submit(self, command, view, now):
        if self.fault is not None:
            raise self.fault
        if command.linear or command.angular:
            until = observation_deadline(view, now)
            command = self.EXPIRED if until is None else command
        else:
            until = None
        self.latest = command, until
        self._wake.set()
        return command

    def _run(self):
        try:
            while not self._closed.is_set():
                self._wake.clear()
                command, until = self.latest
                if until is not None and self.clock.now() >= until:
                    command = self.EXPIRED
                self.actuator.publish(command)
                delay = .04
                if until is not None and (command.linear or command.angular):
                    delay = min(delay, max(0., until - self.clock.now()))
                self._wake.wait(delay)
        except (ActuationError, IOError, OSError) as error:
            self.fault = error
        except Exception:
            self.fault = ActuationError('navigation_heartbeat_failed')

    def close(self):
        self._closed.set()
        self._wake.set()
        self._worker.join(.4)


class NavigationActuator(object):
    def __init__(self, transport, ros_guard, serial_guard, clock, arc_navigation=False):
        self.transport = transport
        self.ros_guard, self.serial_guard, self.clock = ros_guard, serial_guard, clock
        self.sequence = 0
        self.caller = None
        self.armed = False
        self.stop_line = None
        self.unavailable_guards = ()
        self.arc_navigation = arc_navigation

    def open(self):
        if not self.ros_guard.acquire() or not self.serial_guard.acquire():
            raise ActuationError('command_authority_unavailable')
        if self.arc_navigation:
            self.transport.start_arc_navigation()
            ready_line = b'READY arc_navigation\n'
        else:
            self.transport.start_forward_navigation()
            ready_line = b'READY forward_navigation\n'
        self.stop_line = self.transport.read_line(3.5)
        if self.stop_line != ready_line:
            raise ActuationError('navigation_relay_not_ready')
        self.stop_line = None
        self._send(0.)
        binding = []

        def bind():
            binding.append(self.ros_guard.bind_relay(self))

        worker = threading.Thread(target=bind, name='navigation-relay-binding')
        worker.daemon = True
        worker.start()
        stop_waiting = self.clock.now() + 1.
        while worker.is_alive() and self.clock.now() < stop_waiting:
            self.clock.sleep(.04)
            self._send(0.)
        if binding != [True] or not self.serial_guard.verify():
            raise ActuationError('navigation_relay_binding_failed')
        self.caller = self.ros_guard.evidence().get('relay_node')
        if not self.caller:
            raise ActuationError('navigation_relay_caller_missing')
        self.armed = True

    def _send(self, linear, angular=0.):
        if self.stop_line is None:
            self.stop_line = self.transport.check_status()
        if self.stop_line is not None:
            raise ActuationError('navigation_relay_fault')
        frame = {'sequence': self.sequence, 'stamp': self.clock.now(), 'linear': linear, 'angular': angular}
        payload = (json.dumps(frame, allow_nan=False, separators=(',', ':')) + '\n').encode('ascii')
        if self.transport.write_line(payload, .01) is not True:
            raise ActuationError('navigation_relay_write_failed')
        self.sequence += 1

    def publish(self, command):
        if not self.armed:
            raise ActuationError('navigation_relay_not_armed')
        if not safe_velocity(command.linear, command.angular, forward_navigation=True,
                             arc_navigation=self.arc_navigation):
            self._send(0.)
            raise ActuationError('unexpected_arc_command' if self.arc_navigation else 'unexpected_forward_command')
        self.unavailable_guards = ()
        if command.linear:
            self.unavailable_guards = tuple(
                name for name, guard in (('ros_authority', self.ros_guard), ('serial_authority', self.serial_guard))
                if guard.cached_command_deadline() is None)
        if self.unavailable_guards:
            command = Command(0., 0., 'stop', 'command_authority_unavailable')
        self._send(command.linear, command.angular)
        return command

    def close(self):
        try:
            if self.armed and self.stop_line is None:
                self._send(0.)
        except (ActuationError, IOError, OSError):
            pass
        finally:
            try:
                self.transport.close()
            finally:
                self.armed = False
                self.serial_guard.release()
                self.ros_guard.release()
