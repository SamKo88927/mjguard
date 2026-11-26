"""In-process event bus: what the arm is doing, for a UI to show while it happens.

Publishers are the executor's threads, including the monitor inside the follow loop, so publishing never blocks and
never raises: every subscriber gets a bounded queue (events are dropped for a subscriber that falls behind). A
callback subscriber is fed from its queue by its own daemon thread, so a slow or failing callback can never delay
stop or breach handling; its exceptions are logged and swallowed.
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

    def done(self) -> None:
        self._queue.task_done()

    def pending(self) -> int:
        return self._queue.unfinished_tasks

    def close(self) -> None:
        self._bus.unsubscribe(self)


class EventBus:
    def __init__(self, keep: int = 200) -> None:
        self._lock = threading.Lock()
        self._recent: deque[Event] = deque(maxlen=keep)
        self._callback_subs: list[Subscription] = []
        self._queues: list[Subscription] = []
        self._seq = 0

    def publish(self, type: str, **data: Any) -> Event:
        with self._lock:
            self._seq += 1
            event: Event = {"seq": self._seq, "t": round(time.time(), 3), "type": type, **data}
            self._recent.append(event)
            queues = list(self._queues)
        for q in queues:
            q.put(event)
        return event

    def subscribe(self, callback: Callable[[Event], None], maxsize: int = 1000) -> Callable[[], None]:
        """Call `callback(event)` for every event, on a dedicated thread (never the publisher's). Returns unsubscribe.

        Events reach the callback in order. `flush()` waits until every callback has seen what was published.
        """
        sub = self.subscribe_queue(maxsize)
        stopped = threading.Event()

        def run() -> None:
            while not stopped.is_set():
                event = sub.get(timeout=0.1)
                if event is None:
                    continue
                try:
                    callback(event)
                except Exception:
                    log.exception("event subscriber failed on %s", event.get("type"))
                finally:
                    sub.done()

        threading.Thread(target=run, name="mjguard-events", daemon=True).start()
        with self._lock:
            self._callback_subs.append(sub)

        def unsubscribe() -> None:
            stopped.set()
            sub.close()
            with self._lock:
                if sub in self._callback_subs:
                    self._callback_subs.remove(sub)

        return unsubscribe

    def flush(self, timeout: float = 2.0) -> bool:
        """Wait until every callback subscriber has handled every event published so far. False on timeout."""
        deadline = time.monotonic() + timeout
        with self._lock:
            subs = list(self._callback_subs)
        for sub in subs:
            while sub.pending() and time.monotonic() < deadline:
                time.sleep(0.002)
            if sub.pending():
                return False
        return True

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
