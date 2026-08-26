# SPDX-License-Identifier: AGPL-3.0-or-later
"""Media-plugin config resolution: the opt-in "use config from" share-config,
its cycle-prevention, and source-unavailable fallback. Exercises
localm.plugins.media_config dict logic only."""

from localm.plugins import media_config as mc


def _cfg(plugins, enabled=None):
    # The active set is passed explicitly to resolve_config(active=...); this
    # only builds the installed/enabled config shape.
    return {"plugins": plugins, "plugins_enabled": enabled or list(plugins)}


def test_own_block_when_no_sharing():
    cfg = _cfg({"image": {"backend": "comfy", "comfy": {"api_url": "http://i"}}})
    block, warn = mc.resolve_config("image", cfg)
    assert warn is None
    assert block["comfy"]["api_url"] == "http://i"


def test_share_resolves_to_source_block():
    cfg = _cfg({
        "image": {"comfy": {"api_url": "http://image"}},
        "music": {"use_config_from": "image", "comfy": {"api_url": "http://own-music"}},
    })
    block, warn = mc.resolve_config("music", cfg, active={"image", "music"})
    assert warn is None
    # music reflects image's backend, not its own
    assert block["comfy"]["api_url"] == "http://image"
    assert block["use_config_from"] == "image"   # marker preserved


def test_share_never_mutates_sharer_own_block():
    plugins = {
        "image": {"comfy": {"api_url": "http://image"}},
        "music": {"use_config_from": "image", "comfy": {"api_url": "http://own-music"}},
    }
    cfg = _cfg(plugins)
    mc.resolve_config("music", cfg, active={"image", "music"})
    # the stored block is untouched
    assert plugins["music"]["comfy"]["api_url"] == "http://own-music"


def test_returned_block_is_deep_copy():
    """Mutating the resolved block, nested sub-dicts included, must not corrupt
    the stored config."""
    plugins = {"image": {"comfy": {"api_url": "http://image"}}}
    cfg = _cfg(plugins)
    block, _ = mc.resolve_config("image", cfg)
    block["comfy"]["api_url"] = "http://MUTATED"
    block["comfy"]["new_key"] = "x"
    assert plugins["image"]["comfy"] == {"api_url": "http://image"}


def test_resolved_share_block_is_deep_copy_of_source():
    """When music shares image, mutating music's resolved block must not corrupt
    image's stored config."""
    plugins = {
        "image": {"comfy": {"api_url": "http://image"}},
        "music": {"use_config_from": "image", "comfy": {"api_url": "http://own-music"}},
    }
    cfg = _cfg(plugins)
    block, _ = mc.resolve_config("music", cfg, active={"image", "music"})
    block["comfy"]["api_url"] = "http://MUTATED"
    assert plugins["image"]["comfy"]["api_url"] == "http://image"


def test_source_installed_but_disabled_falls_back():
    """A source that is installed but DISABLED is not active, so the sharer falls
    back to its own config."""
    cfg = _cfg({
        "image": {"comfy": {"api_url": "http://image"}},
        "music": {"use_config_from": "image", "comfy": {"api_url": "http://own-music"}},
    })
    block, warn = mc.resolve_config("music", cfg, active={"music"})  # image not active
    assert warn and "image" in warn and "not active" in warn
    assert block["comfy"]["api_url"] == "http://own-music"


def test_source_missing_falls_back_with_warning():
    cfg = _cfg({"music": {"use_config_from": "image",
                          "comfy": {"api_url": "http://own-music"}}})
    block, warn = mc.resolve_config("music", cfg, active={"music"})
    assert warn is not None
    assert block["comfy"]["api_url"] == "http://own-music"


def test_direct_cycle_is_broken():
    cfg = _cfg({
        "image": {"use_config_from": "music", "comfy": {"api_url": "http://i"}},
        "music": {"use_config_from": "image", "comfy": {"api_url": "http://m"}},
    })
    # neither recurses; each falls back to its own with a warning
    bi, wi = mc.resolve_config("image", cfg, active={"image", "music"})
    bm, wm = mc.resolve_config("music", cfg, active={"image", "music"})
    assert wi and wm
    assert bi["comfy"]["api_url"] == "http://i"
    assert bm["comfy"]["api_url"] == "http://m"


def test_three_way_cycle_is_broken():
    """A transitive cycle image->music->video->image must not loop; each falls
    back to its own block with a warning."""
    cfg = _cfg({
        "image": {"use_config_from": "music", "comfy": {"api_url": "http://i"}},
        "music": {"use_config_from": "video", "comfy": {"api_url": "http://m"}},
        "video": {"use_config_from": "image", "comfy": {"api_url": "http://v"}},
    })
    active = {"image", "music", "video"}
    for name, url in (("image", "http://i"), ("music", "http://m"), ("video", "http://v")):
        block, warn = mc.resolve_config(name, cfg, active=active)
        assert warn and "cycle" in warn
        assert block["comfy"]["api_url"] == url


def test_would_cycle_detects_transitive():
    cfg = _cfg({
        "music": {"use_config_from": "video"},
        "video": {"use_config_from": "image"},
    })
    # setting image.use_config_from=music would close image->music->video->image
    assert mc.would_cycle("image", "music", cfg) is True


def test_would_cycle_detects_direct_and_self():
    cfg = _cfg({"image": {}, "music": {"use_config_from": "image"}})
    assert mc.would_cycle("image", "image", cfg) is True     # self
    assert mc.would_cycle("image", "music", cfg) is True      # music->image already
    assert mc.would_cycle("video", "image", cfg) is False     # video->image is fine


def test_self_reference_ignored():
    cfg = _cfg({"image": {"use_config_from": "image", "comfy": {"api_url": "http://i"}}})
    block, warn = mc.resolve_config("image", cfg)
    assert warn is None
    assert block["comfy"]["api_url"] == "http://i"


def test_non_media_source_ignored():
    cfg = _cfg({"image": {"use_config_from": "coder", "comfy": {"api_url": "http://i"}}})
    block, warn = mc.resolve_config("image", cfg)
    assert warn is None                       # coder is not a media plugin
    assert block["comfy"]["api_url"] == "http://i"
