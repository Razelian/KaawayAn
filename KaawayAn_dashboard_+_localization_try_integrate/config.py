"""Classifier and dashboard thresholds only. Node poses and acoustic
parameters live in config/site.toml (kaawayan.load_config).
"""


class Config:
    drone_detection_threshold: float = 0.60
    drone_high_confidence_threshold: float = 0.85
    non_drone_high_confidence_threshold: float = 0.10
    loc_max_age_s: float = 3.0
    CLASSIFIER_FS: int = 16_000
    AUDIO_FS: int = 48_000
    CLASSIFIER_QUEUE_MAX: int = 8
    DRONE_RECENT_S: float = 2.0
    CLASSIFIER_NORMALIZE_RMS_DBFS: float | None = None
    RETENTION_HOURS: float = 24.0
