"""I/O routines for loading meshes and exporting glTF 2.0 / GLB files."""

from animgen.io.model_input import load_model
from animgen.io.glb_output import export_glb

__all__ = [
    "load_model",
    "export_glb",
]
