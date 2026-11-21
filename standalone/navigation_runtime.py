"""Run local control with an asynchronous Decisions worker."""
from __future__ import division

import threading

from navigation import Command, MAX_RESPONSE_AGE, Navigation, ready


class NavigationRuntime(object):
    def __init__(self, decide, publish):
        self.navigation = Navigation()
        self.decide, self.publish = decide, publish
        self._wake = threading.Event()
        self._closed = threading.Event()
        self._job = self._result = None
        self._next_request = 0.
        self._last_grid = None
        self._worker = threading.Thread(target=self._work, name='navigation-decisions')
        self._worker.daemon = True
        self._worker.start()

    def _work(self):
        while not self._closed.is_set():
            self._wake.wait()
            self._wake.clear()
            if self._closed.is_set():
                return
            request = self._job
            try:
                choice = self.decide(request, self._closed.is_set)
            except Exception:
                choice = None
            self._result = (request, choice)

    def cycle(self, view, now):
        """Called by one periodic controller. Returns decision and motion telemetry."""
        result = self._result
        decision = None
        if result is not None:
            request, choice = result
            waiting_for_sensors = (choice not in (None, 'stop') and request is self.navigation.pending
                                   and 0 <= now - request.at < MAX_RESPONSE_AGE and not ready(view, now))
            if not waiting_for_sensors:
                self._result = self._job = None
                accepted = self.navigation.accept(request, choice, view, now)
                decision = (request.sequence, choice, accepted)
                if not accepted or choice == 'stop':
                    self._next_request = now + 1.

        command = self.navigation.step(view, now)
        self.publish(command)
        offered = None
        if (not self._closed.is_set() and not self.navigation.fault and self._job is None and view is not None
                and now >= self._next_request and view.grid_at != self._last_grid):
            self._last_grid = view.grid_at
            offered = self.navigation.request(view, now)
            if offered is not None:
                self._job = offered
                self._wake.set()
        return command, offered, decision

    def close(self):
        self._closed.set()
        self._wake.set()
        try:
            self.publish(Command(0., 0., 'stop', 'shutdown'))
        finally:
            self._worker.join(1.)
        return not self._worker.is_alive()
