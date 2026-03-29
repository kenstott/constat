# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Domain tier routing — determines system vs user DB for writes."""

from constat.core.config import Config


def get_domain_tier(domain_id: str | None, config: Config, user_id: str) -> str:
    """Return 'system' or 'user' for write routing.

    Args:
        domain_id: Domain identifier, or None for user-personal data.
        config: Loaded system Config.
        user_id: Current user ID.

    Returns:
        'system' if data belongs in the shared system DB,
        'user' if data belongs in the user's personal DB.
    """
    if domain_id is None or domain_id == user_id:
        return "user"
    if domain_id == "__base__":
        return "system"
    domain_cfg = config.projects.get(domain_id)
    if domain_cfg and domain_cfg.tier in ("system", "shared"):
        return "system"
    return "user"
