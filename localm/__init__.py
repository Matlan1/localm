# SPDX-License-Identifier: AGPL-3.0-or-later
__version__ = "0.1.5rc3"

# Report, at import, when this package was loaded from a DIFFERENT source
# checkout than the working directory sits in. See localm/_srcguard.py for the
# mechanism: a console script, or a script run by path, never puts cwd on
# sys.path, so an editable install silently serves the checkout it was installed
# from and every observation made afterwards describes code the caller did not
# change. Import is the only point that also covers a throwaway repro script,
# which is precisely the thing written to decide whether a fix works.
#
# Entry points refuse outright (require_own_source); here we only report, because
# an import must never be able to take a process down. The guard is inert unless
# TWO source checkouts are in play, so a normal installation never reaches it.
try:
    from localm._srcguard import warn_if_foreign_source as _warn_if_foreign_source

    _warn_if_foreign_source()
except Exception:                                           # noqa: BLE001
    # Fail open, deliberately: a defect in the guard must not break every import
    # of localm. warn_if_foreign_source already swallows its own errors, so this
    # only covers the import of the guard module itself.
    pass
