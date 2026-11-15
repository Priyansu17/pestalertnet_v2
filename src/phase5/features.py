"""Frozen Phase 2 feature manifest helpers for Phase 5.

This module is the ONLY source of truth for the approved 25-feature order.
It performs no model loading and no evaluation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import yaml

FEATURE_MANIFEST_PATH = (
    "data/processed/feature_manifest.yaml"
)  # relative to PROJECT_ROOT

EXPECTED_N_FEATURES = 25
EXPECTED_INPUT_DAYS = 21
EXPECTED_HORIZON = 7

# Frozen, exact feature order from data/processed/feature_manifest.yaml
EXPECTED_FEATURE_NAMES = [
    "T2M",
    "RH2M",
    "PRECTOTCORR",
    "WS2M",
    "w_mask_T2M",
    "w_mask_RH2M",
    "w_mask_PRECTOTCORR",
    "w_mask_WS2M",
    "pest_surveyed",
    "pest_event",
    "pest_present",
    "T2M_roll7_mean",
    "T2M_roll21_mean",
    "RH2M_roll7_mean",
    "RH2M_roll21_mean",
    "WS2M_roll7_mean",
    "rain_sum7",
    "rain_sum21",
    "rain_days21",
    "t2m_anomaly",
    "event_count_7d",
    "event_count_21d",
    "survey_count_21d",
    "days_since_survey",
    "days_since_event",
]

# Sentinel / missing weather tokens (exact values present in the frozen CSV manifest).
WEATHER_SENTINEL_VALUES = (-999.0,)
# The approved fill policy uses NaN for missing weather; mask channels keep them separate.
MISSING_WEATHER_TRIGGER = float("nan")


@dataclass(frozen=True)
class FeatureManifest:
    """Immutable representation of the frozen Phase 2 manifest."""

    feature_names: tuple[str, ...]
    n_features: int
    input_days: int
    horizon: int
    input_dtype: str
    sentinel_values: tuple[float, ...]
    missing_weather_value: float
    feature_to_index: dict[str, int]
    description: str

    def __post_init__(self) -> None:
        if len(self.feature_names) != self.n_features:
            raise ValueError(
                f"manifest inconsistency in FeatureManifest: "
                f"n_features={self.n_features}, but feature_names has "
                f"{len(self.feature_names)} entries."
            )
        if self.n_features != EXPECTED_N_FEATURES:
            raise ValueError(
                f"frozen spec violation: expected n_features==25, got "
                f"{self.n_features}. No reordering or transformation is allowed."
            )
        if self.input_days != EXPECTED_INPUT_DAYS:
            raise ValueError(
                f"frozen spec violation: expected input_days==21, got "
                f"input_days={self.input_days}."
            )
        if self.horizon != EXPECTED_HORIZON:
            raise ValueError(
                f"frozen spec violation: expected horizon==7, got horizon={self.horizon}."
            )
        # if self.feature_names != EXPECTED_FEATURE_NAMES:
        if self.feature_names != tuple(EXPECTED_FEATURE_NAMES):  
            raise ValueError(
                "frozen manifest feature order changed. This project does not "
                "permit reordering features. Review data/processed/feature_manifest.yaml."
            )
        self._verify_feature_to_index()

    def _verify_feature_to_index(self) -> None:
        if len(self.feature_to_index) != self.n_features:
            raise ValueError(
                "manifest inconsistency: feature_to_index length does not match "
                "n_features. Manual review required."
            )
        for name, idx in self.feature_to_index.items():
            if name not in self.feature_names:
                raise ValueError(
                    f"manifest inconsistency: feature_to_index contains unknown key "
                    f"{name!r}. Manual review required."
                )
            if idx != self.feature_names.index(name):
                raise ValueError(
                    f"manifest inconsistency: feature_to_index[{name!r}]={idx}, "
                    f"but feature order has {name!r} at index "
                    f"{self.feature_names.index(name)}."
                )

    def get_index(self, name: str) -> int:
        try:
            return self.feature_to_index[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown feature {name!r}. Manifest feature order has "
                f"{len(self.feature_names)} features: "
                f"{', '.join(self.feature_names[:5])} ... "
                f"{', '.join(self.feature_names[-5:])}."
            ) from exc

    def index_to_name(self, idx: int) -> str:
        if not 0 <= idx < self.n_features:
            raise IndexError(f"feature index {idx} out of range [0, {self.n_features})")
        return self.feature_names[idx]


def _project_root() -> Path:
    """Return the PestAlertNet project root (two levels above this file)."""
    from pathlib import Path

    return Path(__file__).resolve().parents[2]


def load_feature_manifest(
    path: str | Path | None = None,
) -> FeatureManifest:
    """Load and validate the frozen Phase 2 feature manifest.

    Parameters
    ----------
    path:
        Optional path to `feature_manifest.yaml`. Defaults to the frozen
        `data/processed/feature_manifest.yaml` under the project root.

    Returns
    -------
    FeatureManifest
        Immutable, validated manifest representation.
    """
    import os
    from pathlib import Path

    if path is None:
        candidate = _project_root() / "data" / "processed" / "feature_manifest.yaml"
        if not candidate.exists():
            raise FileNotFoundError(
                f"Frozen feature manifest not found at:\n"
                f"  {candidate}\n"
                f"Expected data/processed/feature_manifest.yaml to exist unchanged."
            )
        path = candidate
    else:
        p = Path(path)
        if not p.is_absolute():
            p = _project_root() / p
        if not p.exists():
            raise FileNotFoundError(
                f"Feature manifest not found: {p}\n"
                f"Expected a frozen manifest describing the 25 approved features."
            )

    # with open(p, "r", encoding="utf-8") as fh:
    #     raw = yaml.safe_load(fh)
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise TypeError("feature_manifest.yaml root must be a mapping.")

    try:
        feature_names = raw["features"]
    except KeyError as exc:
        raise KeyError(
            "feature_manifest.yaml is missing required key 'features'."
        ) from exc

    if not isinstance(feature_names, list):
        raise TypeError("'features' must be a list of feature names.")

    if len(feature_names) != EXPECTED_N_FEATURES:
        raise ValueError(
            f"frozen spec violation: expected n_features=={EXPECTED_N_FEATURES}, "
            f"got {len(feature_names)}. No reordering or transformation is allowed."
        )

    feature_names = tuple(str(name) for name in feature_names)

    # if feature_names != EXPECTED_FEATURE_NAMES:
    if feature_names != tuple(EXPECTED_FEATURE_NAMES):
        raise ValueError(
            "frozen manifest feature order changed. This project does not "
            "permit reordering features. Review data/processed/feature_manifest.yaml."
        )

    try:
        n_features = int(raw["n_features"])
        input_days = int(raw["input_days"])
        horizon = int(raw["horizon"])
    except KeyError as exc:
        raise KeyError(
            f"feature_manifest.yaml missing required metadata key: {exc.args[0]}"
        ) from exc

    if n_features != EXPECTED_N_FEATURES:
        raise ValueError(
            f"frozen spec violation: expected n_features=={EXPECTED_N_FEATURES}, "
            f"got {n_features}."
        )
    if input_days != EXPECTED_INPUT_DAYS:
        raise ValueError(
            f"frozen spec violation: expected input_days==21, got input_days={input_days}."
        )
    if horizon != EXPECTED_HORIZON:
        raise ValueError(
            f"frozen spec violation: expected horizon==7, got horizon={horizon}."
        )

    # Sentinel / missing weather values preserved exactly as in the FE data layer.
    sentinel_values = tuple(
        float(value) for value in raw.get("sentinel_values", WEATHER_SENTINEL_VALUES)
    )
    missing_weather_value = float(raw.get("missing_weather_value", MISSING_WEATHER_TRIGGER))

    feature_to_index = {name: idx for idx, name in enumerate(feature_names)}

    return FeatureManifest(
        feature_names=feature_names,
        n_features=n_features,
        input_days=input_days,
        horizon=horizon,
        input_dtype="float32",
        sentinel_values=sentinel_values,
        missing_weather_value=missing_weather_value,
        feature_to_index=feature_to_index,
        description="Frozen asset: data/processed/feature_manifest.yaml",
    )


__all__ = [
    "FeatureManifest",
    "load_feature_manifest",
    "EXPECTED_N_FEATURES",
    "EXPECTED_INPUT_DAYS",
    "EXPECTED_HORIZON",
    "EXPECTED_FEATURE_NAMES",
    "WEATHER_SENTINEL_VALUES",
    "MISSING_WEATHER_TRIGGER",
]