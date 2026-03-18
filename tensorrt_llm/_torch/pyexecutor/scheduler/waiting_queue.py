from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from typing import Callable, List, Optional

from tensorrt_llm.llmapi.llm_args import WaitingQueuePolicy

from ..executor_request_queue import RequestQueueItem


class WaitingQueue(ABC):
    """Abstract base class for waiting queues."""

    @abstractmethod
    def add_request(self, request: RequestQueueItem) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def add_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Add multiple requests to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> RequestQueueItem:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> RequestQueueItem:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: RequestQueueItem) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Prepend all requests from another iterable to the front of this queue."""
        pass

    @abstractmethod
    def remove_by_ids(self, request_ids: set[int]) -> None:
        """Remove requests with the given IDs."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[RequestQueueItem]:
        """Iterate over the queue according to the policy."""
        pass


class FCFSWaitingQueue(deque, WaitingQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: RequestQueueItem) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def add_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Add multiple requests to the queue according to FCFS policy."""
        self.extend(requests)

    def pop_request(self) -> RequestQueueItem:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> RequestQueueItem:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: RequestQueueItem) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Prepend all requests from another iterable to the front of this queue.

        Note: The requests will be prepended in reverse order of their
        appearance in the `requests` iterable.
        """
        self.extendleft(requests)

    def remove_by_ids(self, request_ids: set[int]) -> None:
        """Remove requests with the given IDs."""
        filtered_requests = [req for req in self if req.id not in request_ids]
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[RequestQueueItem]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()


class PriorityWaitingQueue(WaitingQueue):
    """A priority queue that serves requests in descending priority order.

    Requests are inserted into a sorted list so that the highest-priority
    request is always at the front.  Ties in priority are broken by arrival
    order (FCFS), matching the behaviour of the C++ batch manager's
    ``insertRequestInOrder`` helper (``requestUtils.cpp``).

    Priority values are floats in ``[0, 1]`` where **1.0 is the highest
    priority** and **0.0 is the lowest**.  The default is ``0.5``.

    Preemption semantics
    --------------------
    The C++ batch manager also implements a preemption check in
    ``getLeaderNewReqWithIds``: queued requests whose priority exceeds the
    lowest-priority *active* request are promoted into the active batch first.
    This is naturally replicated here because ``pop_request`` always returns
    the highest-priority waiting request, so when a scheduling slot opens the
    scheduler will always pick the most urgent queued request regardless of
    what is already active.

    Thread safety
    -------------
    This class is **not** thread-safe.  All accesses must be serialised by the
    caller (matching the contract of ``FCFSWaitingQueue``).
    """

    def __init__(self) -> None:
        # Maintained in **descending** priority order: _queue[0] is the
        # highest-priority item.
        self._queue: List[RequestQueueItem] = []

    def _find_insert_pos(self, priority: float) -> int:
        """Return the index at which *priority* should be inserted.

        The list is kept in descending order.  Equal priorities are appended
        after existing items of the same priority (FCFS tiebreaking).

        Uses binary search – O(log n) comparisons, O(n) list shift.
        """
        lo, hi = 0, len(self._queue)
        while lo < hi:
            mid = (lo + hi) // 2
            # Move past all items whose priority is >= the new one so that
            # equal-priority items maintain their arrival order.
            if self._queue[mid].priority >= priority:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def add_request(self, request: RequestQueueItem) -> None:
        """Insert *request* at the position determined by its priority."""
        pos = self._find_insert_pos(request.priority)
        self._queue.insert(pos, request)

    def add_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Insert each request in *requests* at its correct priority position."""
        for req in requests:
            self.add_request(req)

    def pop_request(self) -> RequestQueueItem:
        """Remove and return the highest-priority request."""
        if not self._queue:
            raise IndexError("pop from an empty queue")
        return self._queue.pop(0)

    def peek_request(self) -> RequestQueueItem:
        """Return the highest-priority request without removing it."""
        if not self._queue:
            raise IndexError("peek from an empty queue")
        return self._queue[0]

    def prepend_request(self, request: RequestQueueItem) -> None:
        """Re-insert *request* at its correct priority position.

        Unlike ``FCFSWaitingQueue.prepend_request`` (which unconditionally
        pushes to the front), this method uses ``add_request`` so that the
        sorted invariant is maintained.  This is used to put back requests
        that were temporarily dequeued but could not be scheduled (e.g.
        attention-DP rank capacity exceeded).
        """
        self.add_request(request)

    def prepend_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Re-insert each request at its correct priority position."""
        self.add_requests(requests)

    def remove_by_ids(self, request_ids: set) -> None:
        """Remove all requests whose ``id`` is in *request_ids*."""
        self._queue = [req for req in self._queue if req.id not in request_ids]

    def __bool__(self) -> bool:
        return len(self._queue) > 0

    def __len__(self) -> int:
        return len(self._queue)

    def __iter__(self) -> Iterator[RequestQueueItem]:
        """Iterate from highest to lowest priority."""
        return iter(self._queue)


def create_waiting_queue(
    policy: WaitingQueuePolicy = WaitingQueuePolicy.FCFS,
    priority_fn: Optional[Callable[[RequestQueueItem], float]] = None,
) -> WaitingQueue:
    """Create a waiting queue based on the specified policy.

    Args:
        policy: The scheduling policy to use.

            * ``WaitingQueuePolicy.FCFS`` – First-Come-First-Served (default).
              Requests are served strictly in arrival order.
            * ``WaitingQueuePolicy.PRIORITY`` – Priority-ordered scheduling.
              Requests are served in descending order of
              ``RequestQueueItem.priority`` (float in ``[0, 1]``, higher =
              more urgent).  Equal priorities are broken by arrival order.
              This mirrors the C++ batch manager algorithm implemented in
              ``requestUtils.cpp:insertRequestInOrder``.

        priority_fn: Unused; reserved for future use.

    Returns:
        A WaitingQueue instance.
    """
    if policy == WaitingQueuePolicy.FCFS:
        return FCFSWaitingQueue()
    elif policy == WaitingQueuePolicy.PRIORITY:
        return PriorityWaitingQueue()
    else:
        raise ValueError(f"Unsupported waiting queue policy: {policy}")
