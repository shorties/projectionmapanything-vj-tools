"""Plugin entry point for Daydream Scope integration."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scope.core.registry import PipelineRegistry


def register_pipelines(register: "PipelineRegistry") -> None:
    """Register the depth webcam pipeline with Scope.

    This is the main entry point for the plugin. Scope calls this function
during plugin discovery to register all pipelines provided by this plugin.

    Args:
        register: The pipeline registry provided by Scope.

    """
    from scope_depth.pipelines.pipeline import DepthWebcamPipeline

    register(DepthWebcamPipeline)
