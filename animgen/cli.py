"""
Command Line Interface (CLI) for animgen.
Provides terminal commands for autorigging, wave animation synthesis,
mesh straightening, and 3D asset inspection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from animgen.core.models.fish import FishModels
from animgen.core.models.model import BaseModelClass
from animgen.core.models.serpentine import SerpentineModels
from animgen.animation.straight import straighten
from animgen.io.model_input import load_model


def _print_banner() -> None:
    print(
        "\033[1;36m"
        "==========================================================\n"
        "  animgen: Procedural Animation Generation Framework      \n"
        "==========================================================\033[0m"
    )


def cmd_process(args: argparse.Namespace) -> int:
    """Runs the end-to-end autorig and procedural animation pipeline."""
    model_path = Path(args.model)
    if not model_path.exists():
        print(
            f"\033[1;31mError: Model file not found: {model_path}\033[0m",
            file=sys.stderr,
        )
        return 1

    output_path = (
        Path(args.output)
        if args.output
        else model_path.with_name(f"{model_path.stem}_{args.species}_animated.glb")
    )

    print(f"\033[1;34m[*] Loading model:\033[0m {model_path}")
    base_model = BaseModelClass(model_path)
    print(
        f"    Mesh loaded: {len(base_model.mesh.vertices):,} vertices, "
        f"{len(base_model.mesh.faces):,} faces"
    )

    prompts_embedding = (
        Path(args.prompts_embedding)
        if args.prompts_embedding
        else (
            Path("./models_cache/fixtures/sam3_text_embeddings.pt")
            if Path("./models_cache/fixtures/sam3_text_embeddings.pt").exists()
            else None
        )
    )

    prompts = args.prompts
    if prompts is None and prompts_embedding is None:
        if args.species == "fish":
            prompts = ["dorsal fin", "caudal fin"]
        else:
            prompts = ["body"]

    print(f"\033[1;34m[*] Initializing {args.species.capitalize()} pipeline...\033[0m")
    if args.species == "fish":
        kwargs = {}
        if args.no_straighten:
            kwargs["straighten_mesh"] = False
        pipeline = FishModels(
            model=base_model,
            prompts=prompts,
            prompts_embedding_path=prompts_embedding,
            **kwargs,
        )
    elif args.species == "serpentine":
        kwargs = {}
        if args.n_bones:
            kwargs["n_bones"] = args.n_bones
        pipeline = SerpentineModels(
            model=base_model,
            prompts=prompts,
            prompts_embedding_path=prompts_embedding,
            **kwargs,
        )
    else:
        print(
            f"\033[1;31mError: Unsupported species '{args.species}'. Choose 'fish' or 'serpentine'.\033[0m",
            file=sys.stderr,
        )
        return 1

    print("\033[1;34m[*] Running autorig and animation generation...\033[0m")
    pipeline.process()

    print(f"\033[1;34m[*] Exporting animated GLB:\033[0m {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported = pipeline.export(output_path)

    print(f"\033[1;32m[+] Success! Exported to: {exported}\033[0m")
    return 0


def cmd_autorig(args: argparse.Namespace) -> int:
    """Runs autorigging (segmentation + skeleton + skinning) without animations."""
    model_path = Path(args.model)
    if not model_path.exists():
        print(
            f"\033[1;31mError: Model file not found: {model_path}\033[0m",
            file=sys.stderr,
        )
        return 1

    output_path = (
        Path(args.output)
        if args.output
        else model_path.with_name(f"{model_path.stem}_{args.species}_rigged.glb")
    )

    print(f"\033[1;34m[*] Loading model:\033[0m {model_path}")
    base_model = BaseModelClass(model_path)

    prompts_embedding = (
        Path(args.prompts_embedding)
        if args.prompts_embedding
        else (
            Path("./models_cache/fixtures/sam3_text_embeddings.pt")
            if Path("./models_cache/fixtures/sam3_text_embeddings.pt").exists()
            else None
        )
    )

    prompts = args.prompts
    if prompts is None and prompts_embedding is None:
        prompts = ["dorsal fin", "caudal fin"] if args.species == "fish" else ["body"]

    if args.species == "fish":
        pipeline = FishModels(
            model=base_model,
            prompts=prompts,
            prompts_embedding_path=prompts_embedding,
        )
    elif args.species == "serpentine":
        pipeline = SerpentineModels(
            model=base_model,
            prompts=prompts,
            prompts_embedding_path=prompts_embedding,
            n_bones=args.n_bones if args.n_bones else 20,
        )
    else:
        print(
            f"\033[1;31mError: Unsupported species '{args.species}'.\033[0m",
            file=sys.stderr,
        )
        return 1

    print("\033[1;34m[*] Running autorig...\033[0m")
    pipeline.autorig()

    print(f"\033[1;34m[*] Exporting rigged GLB:\033[0m {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported = pipeline.export(output_path, animation=None)

    print(f"\033[1;32m[+] Success! Rigged model exported to: {exported}\033[0m")
    return 0


def cmd_straighten(args: argparse.Namespace) -> int:
    """Straightens a curved 3D mesh along its longitudinal axis."""
    model_path = Path(args.model)
    if not model_path.exists():
        print(
            f"\033[1;31mError: Model file not found: {model_path}\033[0m",
            file=sys.stderr,
        )
        return 1

    output_path = (
        Path(args.output)
        if args.output
        else model_path.with_name(f"{model_path.stem}_straight.glb")
    )

    print(f"\033[1;34m[*] Loading mesh:\033[0m {model_path}")
    mesh = load_model(model_path)

    print(f"\033[1;34m[*] Straightening mesh along '{args.axis}' axis...\033[0m")
    straightened_mesh = straighten(mesh, axis=args.axis)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    straightened_mesh.export(str(output_path))
    print(f"\033[1;32m[+] Success! Straightened mesh exported to: {output_path}\033[0m")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Inspects a 3D model and displays its geometry, bounding box, and skin data."""
    model_path = Path(args.model)
    if not model_path.exists():
        print(
            f"\033[1;31mError: Model file not found: {model_path}\033[0m",
            file=sys.stderr,
        )
        return 1

    _print_banner()
    print(f"\033[1;34mModel Path:\033[0m {model_path}")
    print(f"\033[1;34mFile Size:\033[0m  {model_path.stat().st_size / 1024:.1f} KB")

    mesh = load_model(model_path)
    print("\n\033[1;33m--- Geometry Overview ---\033[0m")
    print(f"  Vertices:     {len(mesh.vertices):,}")
    print(f"  Faces:        {len(mesh.faces):,}")
    print(f"  Watertight:   {mesh.is_watertight}")
    print(
        f"  Bounds Min:   [{mesh.bounds[0][0]:.3f}, {mesh.bounds[0][1]:.3f}, {mesh.bounds[0][2]:.3f}]"
    )
    print(
        f"  Bounds Max:   [{mesh.bounds[1][0]:.3f}, {mesh.bounds[1][1]:.3f}, {mesh.bounds[1][2]:.3f}]"
    )
    extents = mesh.extents
    print(f"  Extents:      [{extents[0]:.3f}, {extents[1]:.3f}, {extents[2]:.3f}]")
    print(
        f"  Center:       [{mesh.centroid[0]:.3f}, {mesh.centroid[1]:.3f}, {mesh.centroid[2]:.3f}]"
    )

    # Check for glTF specific skin / animations
    if model_path.suffix.lower() in [".glb", ".gltf"]:
        try:
            import pygltflib

            gltf = pygltflib.GLTF2().load(str(model_path))
            print("\n\033[1;33m--- glTF Articulation & Animation ---\033[0m")
            print(f"  Nodes:        {len(gltf.nodes)}")
            print(f"  Meshes:       {len(gltf.meshes)}")
            print(f"  Skins:        {len(gltf.skins)}")
            if gltf.skins:
                for i, s in enumerate(gltf.skins):
                    print(
                        f"    Skin #{i}: {len(s.joints)} joints (skeleton root={s.skeleton})"
                    )
            print(f"  Animations:   {len(gltf.animations)}")
            if gltf.animations:
                for i, anim in enumerate(gltf.animations):
                    name = anim.name or f"anim_{i}"
                    print(
                        f"    Anim #{i} ('{name}'): {len(anim.channels)} channels, {len(anim.samplers)} samplers"
                    )
        except Exception as e:
            print(f"  (glTF inspection notice: {e})")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="animgen",
        description="Procedural Animation Generation Framework for Biological Organisms",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subcommand: process
    proc = subparsers.add_parser(
        "process",
        help="Run end-to-end autorigging and procedural animation synthesis",
    )
    proc.add_argument(
        "-m", "--model", required=True, help="Input 3D model file (.glb, .obj, etc.)"
    )
    proc.add_argument(
        "-s",
        "--species",
        choices=["fish", "serpentine"],
        default="fish",
        help="Organism locomotion type (default: fish)",
    )
    proc.add_argument("-o", "--output", help="Output animated GLB destination path")
    proc.add_argument(
        "--prompts",
        nargs="+",
        help="Anatomical part prompts for SAM3 segmentation (e.g. --prompts 'dorsal fin' 'caudal fin')",
    )
    proc.add_argument(
        "--prompts-embedding",
        help="Path to precomputed text embeddings .pt file",
    )
    proc.add_argument(
        "--n-bones",
        type=int,
        help="Number of spine bones (default: 20 for serpentine, 10 for fish)",
    )
    proc.add_argument(
        "--no-straighten", action="store_true", help="Skip canonical mesh straightening"
    )

    # Subcommand: autorig
    rig = subparsers.add_parser(
        "autorig",
        help="Autorig a 3D model with skeletal armature and skin weights (no animation)",
    )
    rig.add_argument(
        "-m", "--model", required=True, help="Input 3D model file (.glb, .obj, etc.)"
    )
    rig.add_argument(
        "-s",
        "--species",
        choices=["fish", "serpentine"],
        default="fish",
        help="Organism locomotion type (default: fish)",
    )
    rig.add_argument("-o", "--output", help="Output rigged GLB destination path")
    rig.add_argument(
        "--prompts", nargs="+", help="Anatomical part prompts for SAM3 segmentation"
    )
    rig.add_argument(
        "--prompts-embedding", help="Path to precomputed text embeddings .pt file"
    )
    rig.add_argument("--n-bones", type=int, help="Number of spine bones")
    rig.add_argument(
        "--no-straighten", action="store_true", help="Skip canonical mesh straightening"
    )

    # Subcommand: straighten
    st = subparsers.add_parser(
        "straighten",
        help="Canonicalize/straighten a curved mesh using welded Bishop frames",
    )
    st.add_argument(
        "-m", "--model", required=True, help="Input 3D model file (.glb, .obj, etc.)"
    )
    st.add_argument("-o", "--output", help="Output straightened model path")
    st.add_argument(
        "--axis",
        choices=["x", "y", "z"],
        default="y",
        help="Primary spine axis (default: y)",
    )

    # Subcommand: info
    inf = subparsers.add_parser(
        "info",
        help="Inspect 3D model metadata, vertices, bounding box, skin, and animation data",
    )
    inf.add_argument("-m", "--model", required=True, help="Input 3D model file")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "process":
        return cmd_process(args)
    elif args.command == "autorig":
        return cmd_autorig(args)
    elif args.command == "straighten":
        return cmd_straighten(args)
    elif args.command == "info":
        return cmd_info(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
