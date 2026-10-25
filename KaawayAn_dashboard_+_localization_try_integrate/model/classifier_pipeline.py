from queue import Queue, Empty
import numpy as np
from threading import Thread, Event


from model.model import DroneClassifier
from database.database import Database
from config import Config


REQUIRED_AUDIO_SAMPLES = Config.SAMPLING_RATE
DRONE_THRESHOLD = Config.drone_detection_threshold

# === TEMPORARY DEBUG — REMOVE AFTER ACOUSTIC DIAGNOSIS ===
# Dumps each 16k-sample inference window to staging/dumps/ so you can listen
# and FFT exactly what the model hears. To remove: delete this block, the
# _dump_window() function, and its one call site below. Nothing else depends
# on it. Set False to silence without deleting.
DUMP_WINDOWS = True
DUMP_DIR = "staging/dumps"


def _dump_window(node_id: int, timestamp: int, samples: np.ndarray) -> None:
    """TEMPORARY: write one inference window as 16kHz float WAV. Delete with block above."""
    import soundfile as sf
    from pathlib import Path as _P
    _P(DUMP_DIR).mkdir(parents=True, exist_ok=True)
    sf.write(f"{DUMP_DIR}/window_node{node_id}_{timestamp}.wav",
             np.asarray(samples, dtype=np.float32) / 32768.0,
             Config.SAMPLING_RATE)

class ClassifierPipeline:
    def __init__ (self, model : DroneClassifier, database : Database, features_queue : Queue[tuple[int, int, np.ndarray]]):
        self.model = model
        self.database = database
        self.features_queue = features_queue
        self._is_enabled = Event()


    def enable_pipeline(self):
        if  self._is_enabled.is_set():
            print("Classifier pipeline already running")
            return

        self._is_enabled.set()

        while self._is_enabled.is_set():
            # expected array: [node_id, timestamp, [samples]]
            try:
                node_id, timestamp, samples = self.features_queue.get(timeout=0.5)
            except Empty:
                continue

            if len(samples) != REQUIRED_AUDIO_SAMPLES:
                continue
            if DUMP_WINDOWS:  # TEMPORARY: delete this call + _dump_window() when done
                _dump_window(node_id, timestamp, samples)
            prob = self.model.predict(samples)
            prediction = 1 if prob >= DRONE_THRESHOLD else 0
            self.database.log_classifier_prediction(node_id, prediction, prob, timestamp)


    def stop(self):
        self._is_enabled.clear()
