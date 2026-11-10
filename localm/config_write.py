# SPDX-License-Identifier: AGPL-3.0-or-later
"""The one write path a plugin uses for a global config key.

A plugin route runs behind the plugin's own scope, which is usually not a
privileged one, so a plain config write from inside a plugin would let a key
with only that scope change a setting the schema reserves for the owner. This
module applies the same owner tier as ``PATCH /v1/config`` to such a write:
keys in ``settings_schema.admin_only_keys()`` and
``settings_schema.engine_managed_keys()`` need an owner caller.
"""

from __future__ import annotations

from typing import Any


class OwnerRequired(PermissionError):
    """The key is owner-only and the caller is not the owner."""


def owner_only_keys() -> set:
    """Config keys a non-owner caller may not write."""
    from localm.settings_schema import admin_only_keys, engine_managed_keys
    return admin_only_keys() | engine_managed_keys()


def set_global_config_key(key: str, value: Any, *, is_owner: bool) -> None:
    """Write ``config[key] = value`` for a schema setting.

    ``is_owner`` is the caller's ownership as the HTTP layer determines it
    (``Host.set_global_config`` derives it from the request); a stdio caller
    with no request passes True because the process was launched by the owner.
    Raises ``OwnerRequired`` for an owner-only key when ``is_owner`` is False
    (checked before the key or value is examined) and ``ValueError`` for a key
    that is not a config key or a value the schema rejects. The value is
    validated and coerced by the schema before it is stored.
    """
    from localm.config import update_config
    from localm.settings_schema import validate_update

    if key in owner_only_keys() and not is_owner:
        raise OwnerRequired(
            f"Changing {key} requires an owner (admin) key: it widens a trust "
            "boundary.")
    value = validate_update({key: value})[key]
    update_config(lambda cfg: cfg.__setitem__(key, value))
