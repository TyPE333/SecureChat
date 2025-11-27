# src/workers/model_loader.py

import time
from common.logging_utils import log_event
from common.config import config

class ModelLoader:
    """
    Mock model loader: simulates loading time.
    """

    def __init__(self, worker_id: str):
        self.worker_id = worker_id

    def load(self):
        # Simulate loading delay
        log_event("model_loading_started", worker_id=self.worker_id)
        time.sleep(0.5)  # mimic load
        log_event("model_loading_completed", worker_id=self.worker_id)
