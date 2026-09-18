"""
Main entry point for animgen CLI.
Execute commands directly via `python main.py <command> [options]`.
"""

import sys
from animgen.cli import main

if __name__ == "__main__":
    sys.exit(main())
