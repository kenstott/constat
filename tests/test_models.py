# Copyright (c) 2025 Kenneth Stott
# Canary: 38d1ca9b-5a69-4d97-adce-b96f15d7e23a
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests split into domain-specific files:
- test_models_plan.py — Step, StepResult, Plan, PlanDependencies
- test_models_artifact.py — ArtifactType, ARTIFACT_MIME_TYPES, Artifact, ArtifactEdgeCases
- test_models_scheduler.py — ParallelStepSchedulerComponent
"""

from __future__ import annotations
