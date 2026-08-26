# SPDX-License-Identifier: AGPL-3.0-or-later
"""NEW-COMFY-STATUS-OWN-MEANS-TWO-THINGS: `comfy status` used "own" for two
opposite targets on adjacent lines. With comfy_target at its default and no
managed install (the exact reproduction below - a fresh install with nothing
configured):

    Preferred target  : own
    Target now        : your own ComfyUI (http://127.0.0.1:8188)

The config value "own" means localm's OWN MANAGED ComfyUI; "your own
ComfyUI" means the opposite - the USER's separate install. The two lines
read as agreeing while naming opposite targets.
"""

import localm.cli.comfy as comfy_cli


class TestComfyStatusTargetWording:
    def test_default_config_no_managed_install_lines_do_not_collide(
            self, cli_runner):
        """The ticket's own repro: default comfy_target, nothing installed."""
        result = cli_runner.invoke(comfy_cli.comfy_status, ["--no-ping"])
        assert result.exit_code == 0, result.output
        lines = result.output.splitlines()
        preferred = next(l for l in lines if "Preferred target" in l)
        target_now = next(l for l in lines if "Target now" in l)

        assert "own" in preferred
        assert "your own ComfyUI" in target_now
        # The fix: "Preferred target" must spell out the SAME phrase "Target
        # now" uses for the managed instance, so a reader cannot mistake the
        # bare word "own" for the user's own separate install.
        assert "localm's managed ComfyUI" in preferred, preferred

    def test_user_target_is_labelled_with_the_same_phrase_target_now_uses(
            self, cli_runner):
        from localm.config import update_config
        update_config(lambda cfg: cfg.__setitem__("comfy_target", "user"))

        result = cli_runner.invoke(comfy_cli.comfy_status, ["--no-ping"])
        assert result.exit_code == 0, result.output
        preferred = next(l for l in result.output.splitlines()
                        if "Preferred target" in l)
        assert "your own ComfyUI" in preferred, preferred
