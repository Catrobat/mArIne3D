"""Core data structures and base models for animgen."""

from animgen.core.armature import Armature, Bone
from animgen.core.spline import Spline, Spline as CatmullRomSpline
from animgen.core.models.model import BaseModelClass
from animgen.core.models.pipeline import Pipeline
from animgen.core.models.fish import FishModels
from animgen.core.models.serpentine import SerpentineModels

__all__ = [
    "Armature",
    "Bone",
    "Spline",
    "CatmullRomSpline",
    "BaseModelClass",
    "Pipeline",
    "FishModels",
    "SerpentineModels",
]
