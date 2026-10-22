"""Entry point for the installed ``f1-race-engineer`` command and ``python -m ai_engineer``."""

import os
import sys


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)  # the engine's modules import each other flat
    from main import main as run

    run()


if __name__ == '__main__':
    main()
