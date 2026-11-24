"""Bounded PowerContext client outside the local motion loop."""
from __future__ import division

import collections
import json
import re
import threading

try:
    from Queue import Empty, Full, Queue
    from urllib2 import HTTPRedirectHandler, Request, URLError, build_opener
    from urlparse import urlsplit
    STRING = (str, unicode)  # noqa: F821 - Python 2
except ImportError:
    from queue import Empty, Full, Queue
    from urllib.error import URLError
    from urllib.parse import urlsplit
    from urllib.request import HTTPRedirectHandler, Request, build_opener
    STRING = (str,)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class MemoryBridge(object):
    def __init__(self, url, scope, token=None, transport=None):
        address = urlsplit(url)
        if (address.scheme not in ('http', 'https') or not address.hostname or address.username is not None
                or address.query or address.fragment or address.path not in ('', '/')
                or not isinstance(scope, STRING) or not 1 <= len(scope) <= 256):
            raise ValueError('invalid PowerContext origin or scope')
        self.url, self.scope, self.token = url.rstrip('/'), scope, token
        self.transport = transport if transport is not None else self._post
        self.opener = build_opener(_NoRedirect())
        self.counters = collections.Counter()
        self.pending_candidates = collections.deque(maxlen=64)
        self._queue = Queue(maxsize=64)
        self._closing = threading.Event()
        self._writer = threading.Thread(target=self._write, name='powercontext-events')
        self._writer.daemon = True
        self._writer.start()

    def _post(self, endpoint, payload, timeout):
        headers = {'Content-Type': 'application/json'}
        if self.token:
            headers['Authorization'] = 'Bearer ' + self.token
        data = json.dumps(payload, allow_nan=False).encode('utf-8')
        response = self.opener.open(Request(self.url + endpoint, data, headers), timeout=timeout)
        try:
            body = response.read(16385)
        finally:
            response.close()
        if len(body) > 16384:
            raise ValueError('context response is oversized')
        packet = json.loads(body)
        if not isinstance(packet, dict):
            raise ValueError('unexpected context response')
        return packet

    def record(self, event_id, content, metadata, remember=None, experience=None):
        if self._closing.is_set():
            return False
        item = {'scope_id': self.scope, 'source_id': event_id, 'content': content, 'metadata': metadata}
        try:
            self._queue.put_nowait((item, remember, experience))
            return True
        except Full:
            self.counters['write_dropped'] += 1
            return False

    def _write(self):
        while not self._closing.is_set() or not self._queue.empty():
            try:
                item, remember, experience = self._queue.get(timeout=.10)
            except Empty:
                continue
            try:
                saved = self.transport('/v1/sources/content', item, .9)
                source = saved.get('source')
                if (not isinstance(source, dict) or source.get('name') != 'content'
                        or not isinstance(source.get('source_id'), STRING)):
                    raise ValueError('invalid source acknowledgement')
                self.counters['sources_written'] += 1
                if remember:
                    self.transport('/v1/memory/remember',
                                   {'scope_id': self.scope, 'kind': 'event',
                                    'text': remember + ' Evidence event ' + item['source_id'] + '.'}, .9)
                    self.counters['memories_written'] += 1
                if experience:
                    candidate = self.transport('/v1/experience/propose',
                                               {'scope_id': self.scope, 'proposal': experience,
                                                'source_refs': [source], 'artifact_refs': []}, .9)
                    key, version = candidate.get('candidate_id'), candidate.get('version')
                    if not isinstance(key, STRING) or type(version) is not int or version < 1:
                        raise ValueError('invalid experience acknowledgement')
                    self.pending_candidates.append((item['source_id'], key, version))
                    self.counters['experiences_proposed'] += 1
            except (URLError, IOError, OSError, ValueError, AttributeError, TypeError):
                self.counters['write_failed'] += 1

    def prepare(self, query):
        try:
            result = self.transport('/v1/context/prepare',
                                    {'scope_id': self.scope, 'query': query[:512], 'max_bytes': 6000,
                                     'include_receipt': True, 'assembly': {
                                         'sections': [{'family': 'memory', 'limit': 5},
                                                      {'family': 'experience', 'limit': 2}]}}, .6)
            if result.get('status') == 'empty':
                return {'status': 'empty'}
            text = result.get('content')
            if (result.get('status') == 'ready' and isinstance(text, STRING)
                    and 0 < len(text.encode('utf-8')) <= 6000):
                receipt = result.get('receipt') or {}
                references = []
                for item in receipt.get('selected', [])[:7]:
                    address = item.get('artifact_address') or {}
                    artifact = address.get('artifact') or {}
                    key, revision = artifact.get('artifact_id'), artifact.get('revision')
                    if (address.get('scope_id') == self.scope and isinstance(key, STRING)
                            and re.match(r'\A[A-Za-z0-9_.:-]{1,128}\Z', key) and type(revision) is int):
                        references.append((key, revision))
                return {'status': 'ready', 'content': text, 'references': references}
        except (URLError, IOError, OSError, ValueError, AttributeError, TypeError):
            pass
        return {'status': 'unavailable'}

    def close(self):
        self._closing.set()
        self._writer.join(3.)
        return not self._writer.is_alive()
