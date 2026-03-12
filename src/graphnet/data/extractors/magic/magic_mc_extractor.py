"""Extractors for MAGIC MC parquet events."""

from __future__ import annotations

from typing import Any, Dict, List

from .magic_extractor import MAGICExtractor


class MAGICMCPulseExtractor(MAGICExtractor):
    """Pulse-level extractor for MAGIC MC events."""

    def __init__(self, extractor_name: str = "MAGICPixels") -> None:
        super().__init__(extractor_name=extractor_name)

    def __call__(self, cleaned_event: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "signal": cleaned_event["signal"],
            "x_cam": cleaned_event["x_cam"],
            "y_cam": cleaned_event["y_cam"],
            "time": cleaned_event["time"],
            "tel_id": cleaned_event["tel_id"],
        }


class MAGICMCTruthExtractor(MAGICExtractor):
    """Event-level MC truth extractor."""

    def __init__(self, extractor_name: str = "truth") -> None:
        super().__init__(extractor_name=extractor_name)

    def __call__(self, cleaned_event: Dict[str, Any]) -> Dict[str, Any]:
        truth = dict(cleaned_event.get("truth", {}))
        truth["event_id"] = cleaned_event["event_id"]
        return truth


class MAGICMCGlobalExtractor(MAGICExtractor):
    """Event-level metadata/global-parameter extractor."""

    def __init__(self, extractor_name: str = "global") -> None:
        super().__init__(extractor_name=extractor_name)

    def __call__(self, cleaned_event: Dict[str, Any]) -> Dict[str, Any]:
        global_params = dict(cleaned_event.get("global_params", {}))
        global_params["event_id"] = cleaned_event["event_id"]
        # global_params["n_nodes"] = cleaned_event["n_nodes"]
        # global_params["frac_lowest"] = cleaned_event["frac_lowest"]
        return global_params


def default_magic_mc_extractors() -> List[MAGICExtractor]:
    """Default extractor set for MAGIC MC conversion."""
    return [
        MAGICMCPulseExtractor(extractor_name="MAGICPixels"),
        MAGICMCTruthExtractor(extractor_name="truth"),
        MAGICMCGlobalExtractor(extractor_name="global"),
    ]
