"""Pipelines package for scope-depth plugin."""

from scope_depth.pipelines.pipeline import DepthWebcamPipeline, Requirements
from scope_depth.pipelines.schema import DepthWebcamConfig, ModeDefaults

__all__ = ["DepthWebcamConfig", "DepthWebcamPipeline", "ModeDefaults", "Requirements"]
