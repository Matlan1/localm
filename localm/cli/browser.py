# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm setup-browser` - provision the Chromium build the automated
browser drives.

The `browser` pip extra (`pip install "localm[browser]"`) installs the
playwright driver only; the Chromium build it drives is a separate,
version-pinned download. This wraps that download as a localm-native
command, the same shape as `setup-llama` and `setup-embeddings`.
"""

import sys

import click

from ._core import console, main


@main.command("setup-browser")
@click.option("--force", is_flag=True,
              help="Reinstall even if Chromium is already present. NOTE: "
                   "playwright removes the existing build before "
                   "redownloading, so a failed --force run can leave no "
                   "Chromium installed at all.")
def setup_browser(force: bool) -> None:
    """Download the Chromium build the automated browser drives (the coder's
    browser tool, and anything else built on localm.browser).

    \b
      localm setup-browser            # install if missing
      localm setup-browser --force    # reinstall

    Respects the network policy: refused under net_mode=off unless explicit
    downloads are allowed (see `localm config net_allow_model_downloads`).
    Does nothing on the network when Chromium is already installed."""
    from rich.markup import escape

    from ..browser.provision import install_chromium

    result = install_chromium(
        force=force,
        on_progress=lambda line: console.print(line, style="dim", markup=False))
    if not result.ok:
        console.print(f"[red]{escape(result.message)}[/red]")
        sys.exit(1)
    console.print(f"[green]{escape(result.message)}[/green]")
