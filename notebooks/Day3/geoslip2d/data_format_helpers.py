"""Helpers for loading profile data in multiple text-file layouts."""

from pathlib import Path

import numpy as np
import geoslip2d as gs2d


def _detect_data_format(data_filename: Path) -> str:
    """Infer file layout from header/first-line tokens.

    Returns:
        "default" for lon/lat-first numeric files.
        "alternate" for station-first files with Ve_sig/Vn_sig/Vu_sig fields.
    """
    default_votes = 0
    alternate_votes = 0

    with open(data_filename, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # Ignore leading comment marker while preserving header tokens.
            if line.startswith("#"):
                line = line.lstrip("#").strip()
                if not line:
                    continue

            tokens = [t.strip().lower() for t in line.replace(",", " ").split()]
            if not tokens:
                continue

            # Strong alternate indicators from header names.
            if any(k in tokens for k in ("station", "ve_sig", "vn_sig", "vu_sig", "ve_pred", "vn_pred", "n_obs")):
                return "alternate"

            # Try to infer from data-like rows.
            def _is_float(tok: str) -> bool:
                try:
                    float(tok)
                    return True
                except Exception:
                    return False

            is0 = _is_float(tokens[0])
            is1 = _is_float(tokens[1]) if len(tokens) > 1 else False
            is2 = _is_float(tokens[2]) if len(tokens) > 2 else False

            # default: lon, lat, ... so first two are numeric.
            if is0 and is1:
                default_votes += 1
                continue

            # alternate: station, lat, lon, ... so first is non-numeric, next two numeric.
            if (not is0) and is1 and is2:
                alternate_votes += 1
                continue

            # Otherwise keep scanning subsequent lines.
            continue

    if alternate_votes > default_votes:
        return "alternate"
    if default_votes > alternate_votes:
        return "default"
    # Conservative fallback: default parser accepts plain numeric tables.
    return "default"
    raise ValueError(f"No readable lines found in data file: {data_filename}")


def load_profile_data_with_format(
    data_filename: Path,
    data_format: str,
    data_has_vertical: bool = True,
):
    """Load profile data in either default or alternate layout with validation."""
    fmt = data_format.strip().lower()
    if fmt not in {"default", "alternate"}:
        raise ValueError("data_format must be 'default' or 'alternate'.")

    detected = _detect_data_format(data_filename)
    if detected != fmt:
        raise ValueError(
            f"data_format='{fmt}' but file appears to be '{detected}'. "
            "Update data_format to match the file columns."
        )

    if fmt == "default":
        data = gs2d.load_profile_data(data_filename, data_has_vertical=data_has_vertical)
        return data, detected

    # Alternate format columns:
    # station, lat, lon, Ve, Ve_sig, Vn, Vn_sig, Vu, Vu_sig, Ve_pred, Vn_pred, n_obs
    # Map to GeoSlip2D expected keys: lon, lat, Ve, Vn, Vu, Sige, Sign, Sigu
    arr = np.loadtxt(data_filename, comments="#", usecols=(1, 2, 3, 5, 7, 4, 6, 8))
    arr = np.atleast_2d(arr)
    if data_has_vertical and arr.shape[1] < 8:
        raise ValueError(
            "alternate format with vertical data requires at least 8 numeric columns after mapping."
        )

    data = {
        "lon": arr[:, 1],
        "lat": arr[:, 0],
        "Ve": arr[:, 2],
        "Vn": arr[:, 3],
        "Vu": arr[:, 4] if data_has_vertical else np.array([]),
        "Sige": arr[:, 5],
        "Sign": arr[:, 6],
        "Sigu": arr[:, 7] if data_has_vertical else np.array([]),
    }
    return data, detected
