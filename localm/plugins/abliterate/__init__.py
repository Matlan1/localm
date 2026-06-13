"""localm abliterate plugin — decensor a model with Heretic, then register it.

Install with:  pip install "localm[abliterate]"
Invoke via:    localm abliterate --model <id-or-path>

NOTICE: This plugin invokes Heretic (https://github.com/Matlan1/heretic-win-AMD),
licensed under AGPL-3.0-or-later, as a separate program via subprocess. localm
does not import or bundle Heretic; its source is fetched into a gitignored folder
on the user's machine on request.
"""
