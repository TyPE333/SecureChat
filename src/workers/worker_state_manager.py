# src/workers/worker_state_manager.py

from common.schemas.worker_state import WorkerState
from common.logging_utils import log_event


class WorkerStateManager:
    """
    Tracks and logs worker state transitions.
    States: INIT → MODEL_LOADING → READY → BUSY → STREAMING → TEARDOWN → FAILED
    """

    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.state = WorkerState.INIT

    def set_state(self, new_state: WorkerState, request_id: str = None):
        """
        Transition to a new worker state (Enum-based, not string-based)
        """
        self.state = new_state
        log_event(
            "worker_state_change",
            worker_id=self.worker_id,
            request_id=request_id,
            extra={"state": new_state.value}
        )

    def get_state(self) -> WorkerState:
        return self.state
