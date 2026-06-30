# SPDX-License-Identifier: AGPL-3.0-or-later
import click

from ._core import main


_POWERSHELL_COMPLETION = r'''# localm tab completion - add this block to your PowerShell $PROFILE
# (run: notepad $PROFILE)
Register-ArgumentCompleter -Native -CommandName localm -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)
    $words = @($commandAst.CommandElements | Select-Object -Skip 1 | ForEach-Object { $_.Extent.Text })
    if ($words.Count -eq 0 -or ($wordToComplete -eq '' -and $words[-1] -ne '')) {
        $words += ''
    }
    localm __complete @words 2>$null | Where-Object { $_ } | ForEach-Object {
        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
    }
}
'''




@main.command("__complete", hidden=True)
@click.argument("words", nargs=-1)
def _complete_hidden(words):
    """Internal: print completion candidates for the partial command line."""
    words = list(words)
    partial = words[-1] if words else ""
    prior = words[:-1]

    # Commands whose first positional argument is a registered model name
    model_cmds = {"run", "serve", "rm", "alias"}

    if prior and prior[0] in model_cmds and len(prior) == 1:
        from ..config import load_registry
        candidates = sorted(load_registry())
    elif not prior:
        candidates = sorted(
            cmd for cmd, obj in main.commands.items()
            if not getattr(obj, "hidden", False)
        )
    else:
        candidates = []

    for c in candidates:
        if c.startswith(partial):
            click.echo(c)




@main.command("completion")
@click.argument("shell", type=click.Choice(["powershell", "bash", "zsh", "fish"]))
def completion(shell):
    """Print shell tab-completion setup for SHELL.

    \b
    PowerShell:  localm completion powershell >> $PROFILE
    bash:        localm completion bash   (prints the one-liner to add)
    zsh / fish:  same, using Click's built-in completion support
    """
    if shell == "powershell":
        click.echo(_POWERSHELL_COMPLETION)
    elif shell == "bash":
        click.echo('# Add to ~/.bashrc:\neval "$(_LOCALM_COMPLETE=bash_source localm)"')
    elif shell == "zsh":
        click.echo('# Add to ~/.zshrc:\neval "$(_LOCALM_COMPLETE=zsh_source localm)"')
    elif shell == "fish":
        click.echo('# Add to ~/.config/fish/completions/localm.fish:\n'
                   '_LOCALM_COMPLETE=fish_source localm | source')
