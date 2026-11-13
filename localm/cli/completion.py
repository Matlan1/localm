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
        $text = $_
        if ($text -match '[\s''"`$;,(){}@&|<>#]') { $text = "'" + ($text -replace "'", "''") + "'" }
        [System.Management.Automation.CompletionResult]::new($text, $_, 'ParameterValue', $_)
    }
}
'''




@main.command("__complete", hidden=True,
              context_settings={"ignore_unknown_options": True,
                                "help_option_names": []})
@click.argument("words", nargs=-1, type=click.UNPROCESSED)
def _complete_hidden(words):
    """Internal: print completion candidates for the partial command line,
    one per line, from Click's own completion (the one bash, zsh and fish use):
    commands, options, and each argument's own candidates such as model names."""
    from click.shell_completion import ShellComplete

    words = list(words)
    partial = words[-1] if words else ""
    prior = words[:-1]
    complete = ShellComplete(main, {}, "localm", "_LOCALM_COMPLETE")
    for item in complete.get_completions(prior, partial):
        if item.type == "plain" and item.value:
            click.echo(item.value)




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
