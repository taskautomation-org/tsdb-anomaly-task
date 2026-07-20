"""The detector family.

All detectors share :class:`~tsdb_anomaly_task.detectors.base.Detector`, so
they are interchangeable inside an
:class:`~tsdb_anomaly_task.task.AnomalyTask`.  They differ in what question
they ask of the data:

===================================  =====================================
Detector                             Question
===================================  =====================================
:class:`ThresholdDetector`           Is the value outside a fixed band?
:class:`MADDetector`                 Is the value far from its own recent history?
:class:`SeasonalDetector`            Is the value far from this time-of-day's normal?
:class:`DeadmanDetector`             Has the series stopped reporting?
:class:`RateOfChangeDetector`        Did the value move faster than physics allows?
===================================  =====================================
"""

from .base import Candidate, Detector, FluxContext, FluxSupport, FluxUnsupportedError, Judgement
from .deadman import DeadmanDetector
from .mad import NORMAL_CONSISTENCY, MADDetector, robust_scale, small_sample_correction
from .rate_of_change import RateOfChangeDetector
from .seasonal import BucketBaseline, SeasonalDetector, SeasonalPeriod
from .threshold import ThresholdDetector

__all__ = [
    "NORMAL_CONSISTENCY",
    "BucketBaseline",
    "Candidate",
    "DeadmanDetector",
    "Detector",
    "FluxContext",
    "FluxSupport",
    "FluxUnsupportedError",
    "Judgement",
    "MADDetector",
    "RateOfChangeDetector",
    "SeasonalDetector",
    "SeasonalPeriod",
    "ThresholdDetector",
    "robust_scale",
    "small_sample_correction",
]
