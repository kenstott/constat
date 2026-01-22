#!/usr/bin/env python3
# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""
Interactive test environment for Constat.

Run this script to test the system with:
- Chinook database (music store)
- Northwind database (product distribution)
- Countries GraphQL API (geographic data)

Usage:
    python test_interactive.py

Or with a specific question:
    python test_interactive.py "What are the top 5 genres by revenue?"
"""

import sys
from pathlib import Path

# Ensure constat is importable
sys.path.insert(0, str(Path(__file__).parent))

from constat.repl import run_repl


def main():
    """Run the interactive REPL."""
    config_path = Path(__file__).parent / "config.yaml"

    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    # Check for initial problem from command line
    problem = None
    if len(sys.argv) > 1:
        problem = " ".join(sys.argv[1:])

    print("=" * 60)
    print("Constat Interactive Test Environment")
    print("=" * 60)
    print()
    print("Available data sources:")
    print("  - chinook: Digital music store (artists, tracks, customers)")
    print("  - northwind: Product distribution (products, orders, shipping)")
    print("  - countries API: Geographic data (continents, currencies)")
    print()
    print("Example questions to try:")
    print("  - What are the top 5 genres by revenue in Chinook?")
    print("  - Compare customer counts by country across both databases")
    print("  - Which artists have the most tracks?")
    print("  - What are the top selling products in Northwind?")
    print("  - Show revenue by continent (use Countries API)")
    print()
    print("=" * 60)

    # Run the REPL
    run_repl(
        config_path=str(config_path),
        verbose=False,
        problem=problem,
    )


if __name__ == "__main__":
    main()
