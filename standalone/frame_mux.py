"""Combine private processed-frame channels for read-only experiments.

This layer has no geometry, vehicle state, actuation or physical-stop proof.
Its ready value is not a motion permit.
"""
from __future__ import division

from frame_pipe import _valid
from processed_evidence import ProcessedEvidence


class FrameMux(object):  # noqa: UP004 - ROS Melodic Python 2
    def __init__(self, max_processing_lag=.2):
        self.store = ProcessedEvidence({'scan': .3, 'depth': .15}, max_processing_lag,
                                       max_pending=8)
        self.epoch = None
        self.expected = {'scan': 0, 'depth': 0}
        self.pending = {'scan': {}, 'depth': {}}

    def channel_failed(self, stream):
        self.store.producer_failed('producer_failed')

    def accept(self, item):
        stream = item.get('stream') if isinstance(item, dict) else None
        if stream not in ('scan', 'depth') or not _valid(item, stream):
            self.store.producer_failed('invalid_frame')
            return False
        epoch, sequence = item['epoch'], item['sequence']
        if self.epoch is None:
            self.epoch = epoch
        if epoch != self.epoch:
            self.store.producer_failed('epoch_changed')
            return False
        if item['kind'] == 'ingress':
            if sequence != self.expected[stream]:
                self.store.producer_failed('ipc_sequence')
                return False
            got = self.store.ingest(stream, item['acquired_at'], item['received_at'], epoch)
            if got is None or got != sequence:
                self.store.producer_failed('ipc_sequence')
                return False
            self.expected[stream] += 1
            self.pending[stream][sequence] = (item['acquired_at'], item['received_at'])
            return True
        captured = self.pending[stream].get(sequence)
        if captured != (item['acquired_at'], item['received_at']):
            self.store.producer_failed('ipc_mismatch')
            return False
        status = item['status']
        clear = True if status == 'candidate' else False if status == 'obstacle' else None
        accepted = self.store.complete(stream, sequence, clear, item['valid_until'], item['completed_at'],
                                       reason=item['reason'])
        if accepted:
            del self.pending[stream][sequence]
        return accepted

    def read(self, now):
        return self.store.read(now, 0 if self.epoch is None else self.epoch)

    def read_current(self, clock):
        return self.store.read_current(clock, 0 if self.epoch is None else self.epoch)
