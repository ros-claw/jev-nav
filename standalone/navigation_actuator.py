"""Own the forward navigation relay and enforce local output authority."""
from __future__ import division

import json
import threading

from navigation import Command


class ActuationError(RuntimeError):
    pass


class NavigationActuator(object):
    def __init__(self, transport, ros_guard, serial_guard, clock):
        self.transport = transport
        self.ros_guard, self.serial_guard, self.clock = ros_guard, serial_guard, clock
        self.sequence = 0
        self.caller = None
        self.armed = False
        self.stop_line = None

    def open(self):
        if not self.ros_guard.acquire() or not self.serial_guard.acquire():
            raise ActuationError('command_authority_unavailable')
        self.transport.start_forward_navigation()
        self.stop_line = self.transport.read_line(3.5)
        if self.stop_line != b'READY forward_navigation\n':
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

    def _send(self, linear):
        if self.stop_line is None:
            self.stop_line = self.transport.check_status()
        if self.stop_line is not None:
            raise ActuationError('navigation_relay_fault')
        frame = {'sequence': self.sequence, 'stamp': self.clock.now(), 'linear': linear, 'angular': 0.}
        payload = (json.dumps(frame, allow_nan=False, separators=(',', ':')) + '\n').encode('ascii')
        if self.transport.write_line(payload, .01) is not True:
            raise ActuationError('navigation_relay_write_failed')
        self.sequence += 1

    def publish(self, command):
        if not self.armed:
            raise ActuationError('navigation_relay_not_armed')
        if command.angular != 0. or not 0. <= command.linear <= .02:
            self._send(0.)
            raise ActuationError('unexpected_forward_command')
        if command.linear and (self.ros_guard.cached_command_deadline() is None
                               or self.serial_guard.cached_command_deadline() is None):
            command = Command(0., 0., 'stop', 'command_authority_unavailable')
        self._send(command.linear)
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
