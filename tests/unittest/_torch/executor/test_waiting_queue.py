"""Tests for WaitingQueue implementations.

This module tests the waiting queue functionality including:
- FCFSWaitingQueue operations
- PriorityWaitingQueue operations
- WaitingQueue abstract interface
- create_waiting_queue factory function
"""

from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.scheduler import (
    FCFSWaitingQueue,
    PriorityWaitingQueue,
    WaitingQueue,
    create_waiting_queue,
)
from tensorrt_llm.llmapi.llm_args import WaitingQueuePolicy


def create_mock_request_item(request_id: int, priority: float = 0.5) -> RequestQueueItem:
    """Create a mock RequestQueueItem for testing."""
    mock_request = Mock()
    return RequestQueueItem(request_id, mock_request, priority=priority)


class TestFCFSWaitingQueue:
    """Tests for FCFSWaitingQueue."""

    def test_add_request(self):
        """Test adding a single request."""
        queue = FCFSWaitingQueue()
        item = create_mock_request_item(1)

        queue.add_request(item)

        assert len(queue) == 1
        assert queue.peek_request() == item

    def test_add_requests(self):
        """Test adding multiple requests."""
        queue = FCFSWaitingQueue()
        items = [create_mock_request_item(i) for i in range(3)]

        queue.add_requests(items)

        assert len(queue) == 3

    def test_pop_request_fcfs_order(self):
        """Test that pop_request returns requests in FCFS order."""
        queue = FCFSWaitingQueue()
        items = [create_mock_request_item(i) for i in range(3)]
        queue.add_requests(items)

        # Should pop in order: 0, 1, 2
        assert queue.pop_request().id == 0
        assert queue.pop_request().id == 1
        assert queue.pop_request().id == 2

    def test_pop_from_empty_queue(self):
        """Test that pop_request raises IndexError on empty queue."""
        queue = FCFSWaitingQueue()

        with pytest.raises(IndexError):
            queue.pop_request()

    def test_peek_request(self):
        """Test peeking at the front of the queue."""
        queue = FCFSWaitingQueue()
        items = [create_mock_request_item(i) for i in range(3)]
        queue.add_requests(items)

        # Peek should return first item without removing it
        assert queue.peek_request().id == 0
        assert len(queue) == 3  # Size unchanged

    def test_peek_from_empty_queue(self):
        """Test that peek_request raises IndexError on empty queue."""
        queue = FCFSWaitingQueue()

        with pytest.raises(IndexError):
            queue.peek_request()

    def test_prepend_request(self):
        """Test prepending a request to the front."""
        queue = FCFSWaitingQueue()
        queue.add_request(create_mock_request_item(1))
        queue.add_request(create_mock_request_item(2))

        # Prepend item 0 to front
        queue.prepend_request(create_mock_request_item(0))

        # Should pop in order: 0, 1, 2
        assert queue.pop_request().id == 0
        assert queue.pop_request().id == 1
        assert queue.pop_request().id == 2

    def test_prepend_requests(self):
        """Test prepending multiple requests."""
        queue = FCFSWaitingQueue()
        queue.add_request(create_mock_request_item(3))

        # Prepend items [1, 2] - note: extendleft reverses order
        queue.prepend_requests([create_mock_request_item(i) for i in [1, 2]])

        # After extendleft([1, 2]), order is: 2, 1, 3
        assert queue.pop_request().id == 2
        assert queue.pop_request().id == 1
        assert queue.pop_request().id == 3

    def test_remove_by_ids(self):
        """Test removing requests by their IDs."""
        queue = FCFSWaitingQueue()
        items = [create_mock_request_item(i) for i in range(5)]
        queue.add_requests(items)

        # Remove items 1 and 3
        queue.remove_by_ids({1, 3})

        assert len(queue) == 3
        remaining_ids = [item.id for item in queue]
        assert remaining_ids == [0, 2, 4]

    def test_remove_nonexistent_ids(self):
        """Test removing IDs that don't exist (should not raise)."""
        queue = FCFSWaitingQueue()
        items = [create_mock_request_item(i) for i in range(3)]
        queue.add_requests(items)

        # Remove IDs that don't exist
        queue.remove_by_ids({10, 20})

        assert len(queue) == 3

    def test_bool_empty_queue(self):
        """Test bool conversion for empty queue."""
        queue = FCFSWaitingQueue()
        assert not queue
        assert bool(queue) is False

    def test_bool_nonempty_queue(self):
        """Test bool conversion for non-empty queue."""
        queue = FCFSWaitingQueue()
        queue.add_request(create_mock_request_item(1))
        assert queue
        assert bool(queue) is True

    def test_len(self):
        """Test length of queue."""
        queue = FCFSWaitingQueue()
        assert len(queue) == 0

        queue.add_request(create_mock_request_item(1))
        assert len(queue) == 1

        queue.add_requests([create_mock_request_item(i) for i in range(2, 5)])
        assert len(queue) == 4

    def test_iter(self):
        """Test iteration over queue."""
        queue = FCFSWaitingQueue()
        items = [create_mock_request_item(i) for i in range(3)]
        queue.add_requests(items)

        iterated_ids = [item.id for item in queue]
        assert iterated_ids == [0, 1, 2]

        # Iteration should not consume items
        assert len(queue) == 3

    def test_is_waiting_queue_subclass(self):
        """Test that FCFSWaitingQueue is a WaitingQueue."""
        queue = FCFSWaitingQueue()
        assert isinstance(queue, WaitingQueue)


class TestPriorityWaitingQueue:
    """Tests for PriorityWaitingQueue.

    The queue must mirror the behaviour of the C++ batch manager's
    ``insertRequestInOrder`` helper: requests are served in descending
    priority order, with FCFS tiebreaking for equal priorities.
    """

    def test_add_single_request(self):
        """A single request can be added and peeked."""
        queue = PriorityWaitingQueue()
        item = create_mock_request_item(1, priority=0.7)
        queue.add_request(item)
        assert len(queue) == 1
        assert queue.peek_request() is item

    def test_pop_returns_highest_priority_first(self):
        """Requests are returned in descending priority order."""
        queue = PriorityWaitingQueue()
        queue.add_request(create_mock_request_item(1, priority=0.3))
        queue.add_request(create_mock_request_item(2, priority=0.9))
        queue.add_request(create_mock_request_item(3, priority=0.6))

        assert queue.pop_request().priority == 0.9
        assert queue.pop_request().priority == 0.6
        assert queue.pop_request().priority == 0.3

    def test_equal_priority_preserves_fcfs_order(self):
        """Requests with equal priority are served in arrival order (FCFS)."""
        queue = PriorityWaitingQueue()
        # All at the same priority – should come out in id order (1, 2, 3).
        for req_id in [1, 2, 3]:
            queue.add_request(create_mock_request_item(req_id, priority=0.5))

        assert queue.pop_request().id == 1
        assert queue.pop_request().id == 2
        assert queue.pop_request().id == 3

    def test_mixed_priority_and_fcfs(self):
        """Higher-priority requests preempt lower-priority ones; equal priorities are FCFS."""
        queue = PriorityWaitingQueue()
        # Arrival order: req 1 (low), req 2 (high), req 3 (high), req 4 (low)
        queue.add_request(create_mock_request_item(1, priority=0.3))
        queue.add_request(create_mock_request_item(2, priority=0.8))
        queue.add_request(create_mock_request_item(3, priority=0.8))
        queue.add_request(create_mock_request_item(4, priority=0.3))

        # Expected order: 2 (0.8, arrived first), 3 (0.8, arrived second),
        #                 1 (0.3, arrived first), 4 (0.3, arrived second)
        assert queue.pop_request().id == 2
        assert queue.pop_request().id == 3
        assert queue.pop_request().id == 1
        assert queue.pop_request().id == 4

    def test_add_requests_bulk(self):
        """add_requests inserts all items in priority order."""
        queue = PriorityWaitingQueue()
        items = [
            create_mock_request_item(1, priority=0.1),
            create_mock_request_item(2, priority=0.9),
            create_mock_request_item(3, priority=0.5),
        ]
        queue.add_requests(items)

        assert queue.pop_request().priority == 0.9
        assert queue.pop_request().priority == 0.5
        assert queue.pop_request().priority == 0.1

    def test_pop_from_empty_queue_raises(self):
        queue = PriorityWaitingQueue()
        with pytest.raises(IndexError):
            queue.pop_request()

    def test_peek_from_empty_queue_raises(self):
        queue = PriorityWaitingQueue()
        with pytest.raises(IndexError):
            queue.peek_request()

    def test_peek_does_not_remove_item(self):
        queue = PriorityWaitingQueue()
        queue.add_request(create_mock_request_item(1, priority=0.7))
        queue.peek_request()
        assert len(queue) == 1

    def test_prepend_request_inserts_at_correct_priority(self):
        """prepend_request re-inserts at the correct priority position.

        This is used by the attention-DP scheduler to put back requests that
        could not be assigned to a rank.  Unlike FCFSWaitingQueue.prepend_request
        (which always pushes to the front), PriorityWaitingQueue.prepend_request
        respects priority ordering.
        """
        queue = PriorityWaitingQueue()
        queue.add_request(create_mock_request_item(1, priority=0.9))
        queue.add_request(create_mock_request_item(2, priority=0.3))

        # Re-insert a medium-priority request
        queue.prepend_request(create_mock_request_item(3, priority=0.6))

        assert queue.pop_request().id == 1   # 0.9
        assert queue.pop_request().id == 3   # 0.6
        assert queue.pop_request().id == 2   # 0.3

    def test_prepend_requests_inserts_at_correct_priority(self):
        """prepend_requests re-inserts each item at its correct priority position."""
        queue = PriorityWaitingQueue()
        queue.add_request(create_mock_request_item(1, priority=0.9))

        queue.prepend_requests([
            create_mock_request_item(2, priority=0.4),
            create_mock_request_item(3, priority=0.7),
        ])

        assert queue.pop_request().id == 1   # 0.9
        assert queue.pop_request().id == 3   # 0.7
        assert queue.pop_request().id == 2   # 0.4

    def test_remove_by_ids(self):
        queue = PriorityWaitingQueue()
        for i, p in [(1, 0.9), (2, 0.7), (3, 0.5), (4, 0.3)]:
            queue.add_request(create_mock_request_item(i, priority=p))

        queue.remove_by_ids({2, 4})

        assert len(queue) == 2
        assert [item.id for item in queue] == [1, 3]

    def test_remove_nonexistent_ids(self):
        queue = PriorityWaitingQueue()
        queue.add_request(create_mock_request_item(1, priority=0.5))
        queue.remove_by_ids({99, 100})
        assert len(queue) == 1

    def test_bool_empty(self):
        queue = PriorityWaitingQueue()
        assert not queue

    def test_bool_nonempty(self):
        queue = PriorityWaitingQueue()
        queue.add_request(create_mock_request_item(1))
        assert queue

    def test_len(self):
        queue = PriorityWaitingQueue()
        assert len(queue) == 0
        queue.add_request(create_mock_request_item(1, priority=0.8))
        assert len(queue) == 1
        queue.add_requests([create_mock_request_item(i, priority=0.5) for i in range(2, 5)])
        assert len(queue) == 4

    def test_iter_descending_priority_order(self):
        """Iteration yields items from highest to lowest priority."""
        queue = PriorityWaitingQueue()
        queue.add_requests([
            create_mock_request_item(1, priority=0.2),
            create_mock_request_item(2, priority=0.8),
            create_mock_request_item(3, priority=0.5),
        ])
        priorities = [item.priority for item in queue]
        assert priorities == [0.8, 0.5, 0.2]
        # Iteration must not consume items
        assert len(queue) == 3

    def test_is_waiting_queue_subclass(self):
        assert isinstance(PriorityWaitingQueue(), WaitingQueue)

    def test_boundary_priorities(self):
        """Edge-case priorities 0.0 and 1.0 are handled correctly."""
        queue = PriorityWaitingQueue()
        queue.add_request(create_mock_request_item(1, priority=0.0))
        queue.add_request(create_mock_request_item(2, priority=1.0))
        queue.add_request(create_mock_request_item(3, priority=0.5))

        assert queue.pop_request().priority == 1.0
        assert queue.pop_request().priority == 0.5
        assert queue.pop_request().priority == 0.0

    def test_preemption_scenario(self):
        """High-priority late-arriving request is served before low-priority earlier requests.

        This mirrors the C++ batch manager's preemption logic in
        ``executorImpl.cpp:getLeaderNewReqWithIds``: a queued request whose
        priority exceeds the lowest active request's priority is promoted
        into the active batch first.  In PyExecutor this is achieved simply
        by having the highest-priority queued request at the front of the
        waiting queue.
        """
        queue = PriorityWaitingQueue()

        # Three low-priority requests arrive first
        queue.add_request(create_mock_request_item(1, priority=0.2))
        queue.add_request(create_mock_request_item(2, priority=0.2))
        queue.add_request(create_mock_request_item(3, priority=0.2))

        # A high-priority request arrives late
        queue.add_request(create_mock_request_item(4, priority=0.9))

        # The high-priority request should be scheduled first
        assert queue.pop_request().id == 4
        # Followed by the earlier low-priority requests in FCFS order
        assert queue.pop_request().id == 1
        assert queue.pop_request().id == 2
        assert queue.pop_request().id == 3


class TestCreateWaitingQueue:
    """Tests for create_waiting_queue factory function."""

    def test_create_fcfs_queue(self):
        """Test creating FCFS queue."""
        queue = create_waiting_queue(WaitingQueuePolicy.FCFS)
        assert isinstance(queue, FCFSWaitingQueue)

    def test_create_default_queue(self):
        """Test creating queue with default policy."""
        queue = create_waiting_queue()
        assert isinstance(queue, FCFSWaitingQueue)

    def test_create_priority_queue(self):
        """Test creating a priority queue."""
        queue = create_waiting_queue(WaitingQueuePolicy.PRIORITY)
        assert isinstance(queue, PriorityWaitingQueue)

    def test_unsupported_policy_raises(self):
        """Unsupported policy values raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported waiting queue policy"):
            create_waiting_queue("nonexistent_policy")
