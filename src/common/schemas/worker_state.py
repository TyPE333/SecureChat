from enum import Enum

class WorkerState(Enum):
    INIT = "INIT"
    MODEL_LOADING = "MODEL_LOADING"
    READY = "READY"
    BUSY = "BUSY"
    STREAMING = "STREAMING"
    TEARDOWN = "TEARDOWN"
    FAILED = "FAILED"