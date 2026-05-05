"""Input schema handling for stress-conditioned soft-data simulation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is available in the project env.
    pd = None

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


LOGGER = logging.getLogger(__name__)


CANONICAL_COORDS = ("x", "y", "z")


@dataclass
class SchemaInfo:
    """Resolved schema information for one input table."""

    column_map: Dict[str, str]
    required_missing: Dict[str, list[str]]
    optional_missing: Dict[str, list[str]]
    facies_code_to_name: Dict[int, str]
    facies_name_to_code: Dict[str, int]
    metadata: Dict[str, Any]

    @property
    def required_ok(self) -> bool:
        """Return whether all configured required columns were found."""
        return not self.required_missing


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML or JSON pipeline configuration file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() in {".json"}:
            return json.load(f)
        if yaml is None:
            raise ImportError("PyYAML is required to read config.yaml")
        return yaml.safe_load(f)


def save_json(data: Mapping[str, Any], path: str | Path) -> None:
    """Save JSON with stable indentation and UTF-8 text."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)


def load_table(path: str | Path, columns: Optional[Iterable[str]] = None):
    """Load CSV or Parquet input as a pandas DataFrame.

    The pipeline keeps pandas as a thin ingestion layer and converts to NumPy
    arrays before grid reconstruction to avoid avoidable DataFrame copies.
    """
    if pd is None:
        raise ImportError("pandas is required for CSV/Parquet table ingestion")
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=list(columns) if columns else None)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, usecols=list(columns) if columns else None)
    raise ValueError(f"Unsupported input format for {path}. Use CSV or Parquet.")


def resolve_schema(df, config: Mapping[str, Any]) -> SchemaInfo:
    """Resolve configured aliases against actual table columns."""
    columns_cfg = config.get("columns", {})
    schema_cfg = config.get("schema", {})
    required = list(CANONICAL_COORDS) + list(schema_cfg.get("required_properties", []))
    optional = list(schema_cfg.get("optional_properties", []))

    actual = {str(c): c for c in df.columns}
    actual_lower = {str(c).lower(): c for c in df.columns}
    column_map: Dict[str, str] = {}
    missing_required: Dict[str, list[str]] = {}
    missing_optional: Dict[str, list[str]] = {}

    for canonical in list(dict.fromkeys(required + optional)):
        aliases = _aliases_for(canonical, columns_cfg)
        match = _find_column(aliases, actual, actual_lower)
        if match is None:
            if canonical in required:
                missing_required[canonical] = aliases
            elif canonical in optional:
                missing_optional[canonical] = aliases
        else:
            column_map[canonical] = str(match)

    code_to_name, name_to_code = build_facies_maps(schema_cfg.get("facies_map", {}))
    metadata = {
        "n_rows": int(len(df)),
        "input_columns": [str(c) for c in df.columns],
        "column_map": column_map,
        "required_missing": missing_required,
        "optional_missing": missing_optional,
        "facies_code_to_name": {str(k): v for k, v in code_to_name.items()},
        "property_units": schema_cfg.get("property_units", {}),
    }

    if missing_required:
        raise ValueError(
            "Missing required input columns: "
            + ", ".join(f"{k} aliases={v}" for k, v in missing_required.items())
        )
    if missing_optional:
        LOGGER.warning("Optional columns not found: %s", missing_optional)

    return SchemaInfo(
        column_map=column_map,
        required_missing=missing_required,
        optional_missing=missing_optional,
        facies_code_to_name=code_to_name,
        facies_name_to_code=name_to_code,
        metadata=metadata,
    )


def canonicalize_dataframe(df, schema: SchemaInfo):
    """Return a view-like DataFrame with canonical column names where possible."""
    rename = {source: target for target, source in schema.column_map.items()}
    needed = list(rename.keys())
    out = df.loc[:, needed].rename(columns=rename)
    if "facies" in out.columns:
        out["facies"] = encode_facies_values(
            out["facies"].to_numpy(), schema.facies_code_to_name, schema.facies_name_to_code
        )
    return out


def build_facies_maps(facies_map: Mapping[Any, Any]) -> tuple[Dict[int, str], Dict[str, int]]:
    """Build bidirectional facies maps from config values.

    Numeric keys are preserved as integer class codes. Non-numeric keys are
    assigned compact integer codes in declaration order.
    """
    if not facies_map:
        facies_map = {0: "mudstone", 1: "oil_layer", 2: "poor_oil_layer", 3: "dry_layer"}
    code_to_name: Dict[int, str] = {}
    for fallback_code, (raw_key, raw_name) in enumerate(facies_map.items()):
        try:
            code = int(raw_key)
        except (TypeError, ValueError):
            code = fallback_code
        code_to_name[code] = str(raw_name)
    name_to_code = {name: code for code, name in code_to_name.items()}
    return code_to_name, name_to_code


def encode_facies_values(
    values: np.ndarray,
    code_to_name: Mapping[int, str],
    name_to_code: Mapping[str, int],
) -> np.ndarray:
    """Encode raw facies labels into integer codes."""
    raw = np.asarray(values)
    encoded = np.full(raw.shape, -1, dtype=np.int16)
    numeric = _as_numeric_or_none(raw)
    if numeric is not None:
        rounded = np.rint(numeric).astype(np.int64)
        for code in code_to_name:
            encoded[rounded == int(code)] = int(code)
    for name, code in name_to_code.items():
        encoded[raw.astype(str) == str(name)] = int(code)
    unknown = encoded < 0
    if np.any(unknown):
        unique_unknown = sorted(set(raw[unknown].astype(str).tolist()))
        start = max(code_to_name.keys(), default=-1) + 1
        remap = {name: start + i for i, name in enumerate(unique_unknown)}
        LOGGER.warning("Encountered unknown facies labels; assigning new codes: %s", remap)
        raw_str = raw.astype(str)
        for name, code in remap.items():
            encoded[raw_str == name] = code
    return encoded


def write_metadata(schema: SchemaInfo, path: str | Path, extra: Optional[Mapping[str, Any]] = None) -> None:
    """Write schema metadata for reproducibility."""
    metadata = dict(schema.metadata)
    if extra:
        metadata.update(extra)
    save_json(metadata, path)


def _aliases_for(canonical: str, columns_cfg: Mapping[str, Any]) -> list[str]:
    aliases = columns_cfg.get(canonical, canonical)
    if aliases is None:
        return [canonical]
    if isinstance(aliases, str):
        aliases = [aliases]
    return list(dict.fromkeys([canonical, *[str(a) for a in aliases]]))


def _find_column(aliases: Iterable[str], actual: Mapping[str, Any], actual_lower: Mapping[str, Any]):
    for alias in aliases:
        if alias in actual:
            return actual[alias]
        lower = alias.lower()
        if lower in actual_lower:
            return actual_lower[lower]
    return None


def _as_numeric_or_none(values: np.ndarray) -> Optional[np.ndarray]:
    try:
        return values.astype(float)
    except (TypeError, ValueError):
        return None


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
