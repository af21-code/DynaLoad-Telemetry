"""
Unsupervised Thermo-Hydraulic & Structural Anomaly Detection Engine.

Uses an Isolation Forest model trained on nominal stroke-vs-load-vs-pressure-vs-velocity
signatures to detect:
1. Nitrogen (N2) gas pre-charge loss or dynamic seal blow-by (depressed polytropic curve)
2. Hydraulic damping degradation, oil cavitation, or orifice/shim-stack fatigue
3. Unbounded mechanical bottoming and high-energy transient excursions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from scipy.ndimage import uniform_filter1d

from .kinematics import compute_damper_velocity
from .models import (
    AnomalyResult,
    OperationalProfile,
    ProfileConfig,
    TelemetryBatch,
    get_profile_config,
)

try:
    from sklearn.ensemble import IsolationForest as _SklearnIsolationForest

    HAS_SKLEARN = True
except ImportError:  # pragma: no cover - fallback for minimal offline environments
    _SklearnIsolationForest = None
    HAS_SKLEARN = False


EULER_MASCHERONI = 0.5772156649015329


def _c_factor(n: np.ndarray | float) -> np.ndarray | float:
    """
    Average path length of unsuccessful search in a Binary Search Tree (Liu et al., 2008):
    c(n) = 2 * H(n - 1) - 2 * (n - 1) / n
    """
    arr = np.asarray(n, dtype=np.float64)
    result = np.zeros_like(arr, dtype=np.float64)

    eq2 = arr == 2.0
    gt2 = arr > 2.0
    result[eq2] = 1.0
    if np.any(gt2):
        n_gt2 = arr[gt2]
        harmonic = np.log(n_gt2 - 1.0) + EULER_MASCHERONI
        result[gt2] = 2.0 * harmonic - (2.0 * (n_gt2 - 1.0) / n_gt2)

    return float(result) if np.ndim(n) == 0 else result


@dataclass
class _IsoNode:
    feature_idx: int = -1
    split_val: float = 0.0
    size: int = 0
    left: Optional["_IsoNode"] = None
    right: Optional["_IsoNode"] = None


class _NumpyIsolationForest:
    """
    Pure NumPy implementation of Isolation Forest (Liu, Ting & Zhou, ICDM 2008)
    providing a drop-in fallback when scikit-learn binary wheels are not installed.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int | str = "auto",
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self._trees: List[_IsoNode] = []
        self._max_samples_int: int = 256
        self.offset_: float = -0.5

    def _build_tree(
        self,
        X: np.ndarray,
        depth: int,
        max_depth: int,
        rng: np.random.Generator,
    ) -> _IsoNode:
        n_samples, n_features = X.shape
        if depth >= max_depth or n_samples <= 1:
            return _IsoNode(size=n_samples)

        mins = np.min(X, axis=0)
        maxs = np.max(X, axis=0)
        valid_features = np.flatnonzero(maxs > mins + 1e-12)
        if len(valid_features) == 0:
            return _IsoNode(size=n_samples)

        q = int(rng.choice(valid_features))
        p = float(rng.uniform(mins[q], maxs[q]))
        left_mask = X[:, q] < p
        if np.all(left_mask) or not np.any(left_mask):
            return _IsoNode(size=n_samples)

        return _IsoNode(
            feature_idx=q,
            split_val=p,
            size=n_samples,
            left=self._build_tree(X[left_mask], depth + 1, max_depth, rng),
            right=self._build_tree(X[~left_mask], depth + 1, max_depth, rng),
        )

    def _path_length(self, X: np.ndarray, node: _IsoNode, current_depth: int) -> np.ndarray:
        if node.left is None or node.right is None or node.feature_idx < 0:
            return np.full(len(X), current_depth + float(_c_factor(node.size)), dtype=np.float64)

        lengths = np.empty(len(X), dtype=np.float64)
        left_mask = X[:, node.feature_idx] < node.split_val
        if np.any(left_mask):
            lengths[left_mask] = self._path_length(X[left_mask], node.left, current_depth + 1)
        if np.any(~left_mask):
            lengths[~left_mask] = self._path_length(X[~left_mask], node.right, current_depth + 1)
        return lengths

    def fit(self, X: np.ndarray) -> "_NumpyIsolationForest":
        X_arr = np.asarray(X, dtype=np.float64)
        n_samples = len(X_arr)
        if isinstance(self.max_samples, str) and self.max_samples == "auto":
            self._max_samples_int = min(256, n_samples)
        else:
            self._max_samples_int = min(int(self.max_samples), n_samples)

        max_depth = int(np.ceil(np.log2(max(2, self._max_samples_int))))
        rng = np.random.default_rng(self.random_state)
        self._trees = []

        for _ in range(self.n_estimators):
            idx = rng.choice(n_samples, size=self._max_samples_int, replace=False)
            tree = self._build_tree(X_arr[idx], depth=0, max_depth=max_depth, rng=rng)
            self._trees.append(tree)

        raw_scores = self.score_samples(X_arr)
        self.offset_ = float(np.percentile(raw_scores, 100.0 * self.contamination))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        avg_lengths = np.zeros(len(X_arr), dtype=np.float64)
        for tree in self._trees:
            avg_lengths += self._path_length(X_arr, tree, 0)
        avg_lengths /= max(1, len(self._trees))

        c_norm = max(1e-9, float(_c_factor(self._max_samples_int)))
        anomaly_score = np.power(2.0, -avg_lengths / c_norm)
        return -anomaly_score

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.score_samples(X) - self.offset_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) < 0.0, -1, 1)


def compute_expected_polytropic_pressure(
    strut_stroke_mm: np.ndarray,
    config: ProfileConfig,
) -> np.ndarray:
    """
    Compute theoretical internal nitrogen gas spring pressure P_expected(x) (bar)
    under the polytropic gas law:
        P(x) = P_0 * (L_0 / (L_0 - x))^gamma
    """
    stroke = np.asarray(strut_stroke_mm, dtype=np.float64)
    l0 = config.effective_gas_length_mm
    clamped = np.clip(stroke, 0.0, 0.94 * l0)
    return config.nominal_static_pressure_bar * np.power(l0 / (l0 - clamped), config.polytropic_gamma)


def extract_feature_matrix(
    batch: TelemetryBatch,
    config: Optional[ProfileConfig] = None,
    velocity_mm_s: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract 6D physics-informed feature matrix X for Isolation Forest training and inference.

    Features:
    0. Normalized stroke: x / x_max
    1. Normalized damper speed: |v| / v_knee
    2. Vertical G load factor: n_z
    3. Polytropic pressure ratio: P_measured / P_expected(x)
    4. Specific damping admittance: |v| / (v_knee * (|n_z - 1.0| + 0.35))
    5. Rolling 250 ms high-frequency velocity RMS normalized by v_knee
    """
    cfg = config or get_profile_config(batch.profile)
    vel = (
        np.asarray(velocity_mm_s, dtype=np.float64)
        if velocity_mm_s is not None
        else compute_damper_velocity(batch.strut_stroke_mm, batch.timestamp_ms)
    )

    expected_p = compute_expected_polytropic_pressure(batch.strut_stroke_mm, cfg)
    pressure_ratio = batch.internal_pressure_bar / np.maximum(expected_p, 1.0)

    stroke_norm = batch.strut_stroke_mm / cfg.max_stroke_mm
    speed_norm = np.abs(vel) / cfg.bump_knee_velocity_mm_s
    vertical_g = batch.vertical_g

    dynamic_g_delta = np.abs(vertical_g - 1.0) + 0.35
    damping_admittance = speed_norm / dynamic_g_delta

    # Rolling 25-sample (250 ms at 100 Hz) RMS of velocity to capture cavitation/underdamped ringing
    vel_sq = np.square(vel / cfg.bump_knee_velocity_mm_s)
    rolling_vel_rms = np.sqrt(np.maximum(0.0, uniform_filter1d(vel_sq, size=25, mode="nearest")))

    features = np.column_stack(
        [
            stroke_norm,
            speed_norm,
            vertical_g,
            pressure_ratio,
            damping_admittance,
            rolling_vel_rms,
        ]
    )
    return features, expected_p, pressure_ratio


class StrutAnomalyDetector:
    """
    Isolation Forest anomaly detector for Aerospace ATA 32 oleo-struts and F1 dampers.
    """

    def __init__(
        self,
        config: ProfileConfig,
        n_estimators: int = 120,
        contamination: float = 0.04,
        random_state: int = 42,
    ) -> None:
        self.config = config
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

        if HAS_SKLEARN and _SklearnIsolationForest is not None:
            self.model = _SklearnIsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=random_state,
            )
        else:
            self.model = _NumpyIsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=random_state,
            )
        self._is_fitted = False

    def fit(self, nominal_batches: List[TelemetryBatch] | TelemetryBatch) -> "StrutAnomalyDetector":
        """Train the Isolation Forest on one or more nominal telemetry runs."""
        if isinstance(nominal_batches, TelemetryBatch):
            batches = [nominal_batches]
        else:
            batches = list(nominal_batches)

        feature_blocks = []
        for b in batches:
            feats, _, _ = extract_feature_matrix(b, self.config)
            feature_blocks.append(feats)

        X_raw = np.vstack(feature_blocks)

        # Augment nominal training cloud with small physical jitter so the manifold has healthy thickness
        rng = np.random.default_rng(self.random_state)
        jitter_1 = X_raw + rng.normal(0.0, [0.02, 0.04, 0.03, 0.025, 0.04, 0.03], size=X_raw.shape)
        jitter_2 = X_raw + rng.normal(0.0, [0.03, 0.05, 0.04, 0.035, 0.05, 0.04], size=X_raw.shape)
        X_aug = np.vstack([X_raw, jitter_1, jitter_2])

        self._mean = np.mean(X_aug, axis=0)
        self._std = np.maximum(np.std(X_aug, axis=0), 1e-3)
        X_scaled = (X_aug - self._mean) / self._std

        self.model.fit(X_scaled)
        self._is_fitted = True
        return self

    def evaluate(
        self,
        batch: TelemetryBatch,
        velocity_mm_s: Optional[np.ndarray] = None,
    ) -> AnomalyResult:
        """
        Score a telemetry batch and diagnose specific thermo-hydraulic or structural fault modes.
        """
        if not self._is_fitted or self._mean is None or self._std is None:
            raise RuntimeError("StrutAnomalyDetector must be fitted before calling evaluate().")

        X_raw, expected_p, pressure_ratio = extract_feature_matrix(
            batch,
            self.config,
            velocity_mm_s=velocity_mm_s,
        )
        X_scaled = (X_raw - self._mean) / self._std

        decision_scores = self.model.decision_function(X_scaled)
        predictions = self.model.predict(X_scaled)

        # Map Isolation Forest decision function to an intuitive [0, 1] severity score
        # where < 0.50 is nominal inlier and >= 0.50 is anomalous
        normalized_scores = 1.0 / (1.0 + np.exp(4.5 * decision_scores))

        # Physical invariant reinforcement for severe gas pressure loss or bottoming
        severe_pressure_drop = pressure_ratio < self.config.min_healthy_pressure_ratio
        normalized_scores = np.where(
            severe_pressure_drop,
            np.maximum(normalized_scores, 0.78),
            normalized_scores,
        )

        is_anomaly = (predictions == -1) | (normalized_scores >= 0.55)
        anomalous_fraction = float(np.mean(is_anomaly)) if len(is_anomaly) > 0 else 0.0
        peak_score = float(np.max(normalized_scores)) if len(normalized_scores) > 0 else 0.0
        mean_p_ratio = float(np.mean(pressure_ratio)) if len(pressure_ratio) > 0 else 1.0

        fault_modes: List[str] = []

        # 1. Check for N2 Gas Pre-Charge Loss / Seal Blow-By
        low_pressure_frac = float(np.mean(severe_pressure_drop))
        if low_pressure_frac > 0.12 or mean_p_ratio < self.config.min_healthy_pressure_ratio:
            if self.config.profile == OperationalProfile.AEROSPACE_ATA32:
                fault_modes.append("ATA32_N2_PRECHARGE_LOSS_OR_SEAL_BLOWBY")
            else:
                fault_modes.append("F1_RESERVOIR_GAS_LOSS_BLADDER_LEAK")

        # 2. Check for Hydraulic Damping Degradation / Cavitation / Orifice Wear
        high_osc_frac = float(np.mean(X_raw[:, 5] > 0.85))
        high_admittance_frac = float(np.mean(X_raw[:, 4] > 1.65))
        if high_osc_frac > 0.15 or high_admittance_frac > 0.12:
            if self.config.profile == OperationalProfile.AEROSPACE_ATA32:
                fault_modes.append("ATA32_HYDRAULIC_ORIFICE_DAMPING_DEGRADATION")
            else:
                fault_modes.append("F1_DAMPER_HYDRAULIC_CAVITATION_SHIM_FATIGUE")

        # 3. Check for Mechanical Bottoming / Extreme Impact Excursion in Manifold
        bottoming_frac = float(np.mean(batch.strut_stroke_mm >= self.config.bottoming_stroke_threshold_mm))
        if bottoming_frac > 0.005:
            if self.config.profile == OperationalProfile.AEROSPACE_ATA32:
                fault_modes.append("ATA32_OLEO_MECHANICAL_STOP_BOTTOMING")
            else:
                fault_modes.append("F1_SKID_BLOCK_PLANK_BOTTOMING_STRIKE")

        is_degraded = anomalous_fraction >= 0.12 or len(fault_modes) > 0

        return AnomalyResult(
            anomaly_scores=np.round(normalized_scores, 4),
            raw_decision_scores=np.round(decision_scores, 4),
            is_anomaly=is_anomaly,
            expected_polytropic_pressure_bar=np.round(expected_p, 3),
            pressure_ratio=np.round(pressure_ratio, 4),
            anomalous_fraction=round(anomalous_fraction, 4),
            peak_anomaly_score=round(peak_score, 4),
            mean_pressure_ratio=round(mean_p_ratio, 4),
            detected_fault_modes=fault_modes,
            is_degraded=is_degraded,
        )
