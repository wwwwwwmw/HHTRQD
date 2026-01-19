from __future__ import annotations

import os

from sqlalchemy.engine import Engine

from .auth import hash_password
from .constants import AHP_CRITERIA, CRITERIA_LABELS, COST_CRITERIA
from .db import create_user, get_user_by_username, upsert_default_criteria


def ensure_seed(engine: Engine) -> None:
    # Default criteria: AHP + basic filters
    defaults: list[dict] = []

    for key in AHP_CRITERIA:
        direction = "cost" if key in COST_CRITERIA else "benefit"
        defaults.append(
            {
                "key": key,
                "label": CRITERIA_LABELS.get(key, key),
                "kind": "ahp",
                "direction": direction,
                "enabled": True,
            }
        )

    # Minimal filter criteria (can be edited by admin)
    defaults.extend(
        [
            {"key": "manufacturer", "label": "Hãng xe", "kind": "filter", "direction": "none", "enabled": True},
            {"key": "year", "label": "Năm sản xuất", "kind": "filter", "direction": "none", "enabled": True},
            {"key": "price", "label": "Giá (USD)", "kind": "filter", "direction": "none", "enabled": True},
            {"key": "mileage", "label": "Số dặm", "kind": "filter", "direction": "none", "enabled": True},
        ]
    )

    upsert_default_criteria(engine, defaults)

    # Seed admin user if none exists
    admin_username = os.environ.get("ADMIN_USERNAME", "admin")
    admin_password = os.environ.get("ADMIN_PASSWORD", "admin123")

    existing = get_user_by_username(engine, admin_username)
    if existing is None:
        create_user(engine, admin_username, hash_password(admin_password), role="admin")
