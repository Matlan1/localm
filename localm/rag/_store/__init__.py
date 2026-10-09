# SPDX-License-Identifier: AGPL-3.0-or-later
"""The modules behind ``localm.rag.store``. Callers import from
``localm.rag.store``.

Every name these modules read through ``_st`` is looked up on
``localm.rag.store`` at call time, so a value replaced there is the one
every call site uses. Only those names are read that way: any other name
``localm.rag.store`` re-exports is a second reference to the object the
defining module here holds, so replacing it on ``localm.rag.store`` does not
reach these modules. See test_rag_store_live_lookups.
"""
