# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Visualization output utilities for constat.

This module provides helpers for saving visualizations to files
and registering them as artifacts for the React UI.
"""

from .output import VisualizationHelper, create_viz_helper

__all__ = ["VisualizationHelper", "create_viz_helper"]
