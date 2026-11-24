# SPDX-License-Identifier: AGPL-3.0-or-later
"""Replaces the files of an install, and puts them back when that fails or is
interrupted. Standard library only.

It runs outside the install it changes::

    <a python of the install's version> -I -S <id folder>/applier/update_mechanism restore <root>

``<id folder>/applier`` is made by :func:`.shared.copy_applier`. ``import localm``
raises in that process (see ``__main__``). Nothing in this package imports localm,
so it can also be imported as ``localm.update_mechanism``.

- :mod:`.job`: the record of an update, its lock and its phases; :func:`.job.restore`,
  :func:`.job.finish` and :func:`.job.cleanup`.
- :mod:`.mover`: the plan of one step and the file moves of it.
- :mod:`.packages`: the step that replaces Python packages with uv.
- :mod:`.disk`: file identities, confined paths and flushed writes.
- :mod:`.shared`: code of the build that is used here by file path.

Layout, all on the volume of the install::

    <root>/.localm-update/lock                  held for the whole update
    <root>/.localm-update/job.json              the record; present = not finished
    <root>/.localm-update/<id>/new/             the new files
    <root>/.localm-update/<id>/old/             the files they replaced
    <root>/.localm-update/<id>/plans/NN-step.json
    <root>/.localm-update/kept/<id>/            files of the user, never deleted here
    <root>/.venv/.localm-old/<id>/              the same for Python packages
"""
