"""In-process event bus: what the arm is doing, for a UI to show while it happens.

Publishers are the executor's threads, including the monitor inside the follow loop, so publishing never blocks and
never raises: each subscriber gets a bounded queue (events are dropped for a subscriber that falls behind) or a
callback whose exceptions are logged and swallowed.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

log = logging.getLogger("mjguard")
Event = dict[str, Any]

EVENT_TYPES = (
    "motion_started",
    "progress",
    "motion_finished",
    "refused",
    "stopped",
    "paused",
    "resumed",
    "breach",
    "pose_saved",
    "pose_deleted",
)


class Subscription:
    """A bounded queue of events. `get` blocks up to `timeout` seconds and returns None if nothing came."""

    def __init__(self, bus: EventBus, maxsize: int) -> None:
        self._bus = bus
        self._queue: queue.Queue[Event] = queue.Queue(maxsize)
        self.dropped = 0

    def put(self, event: Event) -> None:
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self.dropped += 1

    def get(self, timeout: float | None = None) -> Event | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        self._bus.unsubscribe(self)


class EventBus:
    def __init__(self, keep: int = 200) -> None:
        self._lock = threading.Lock()
        self._recent: deque[Event] = deque(maxlen=keep)
        self._callbacks: list[Callable[[Event], None]] = []
        self._queues: list[Subscription] = []
        self._seq = 0

    def publish(self, type: str, **data: Any) -> Event:
        with self._lock:
            self._seq += 1
            event: Event = {"seq": self._seq, "t": round(time.time(), 3), "type": type, **data}
            self._recent.append(event)
            callbacks = list(self._callbacks)
            queues = list(self._queues)
        for q in queues:
            q.put(event)
        for cb in callbacks:
            try:
                cb(event)
            except Exception:
                log.exception("event subscriber failed on %s", type)
        return event

    def subscribe(self, callback: Callable[[Event], None]) -> Callable[[], None]:
        """Call `callback(event)` for every event, on the publishing thread. Keep it short. Returns unsubscribe."""
        with self._lock:
            self._callbacks.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._callbacks:
                    self._callbacks.remove(callback)

        return unsubscribe

    def subscribe_queue(self, maxsize: int = 1000) -> Subscription:
        """A queue that receives every event from now on; close() it when done."""
        sub = Subscription(self, maxsize)
        with self._lock:
            self._queues.append(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            if sub in self._queues:
                self._queues.remove(sub)

    def latest(self, n: int = 50, after: int = 0) -> list[Event]:
        """The last `n` events, optionally only those with seq > `after`."""
        with self._lock:
            return [e for e in self._recent if e["seq"] > after][-n:] if n > 0 else []
