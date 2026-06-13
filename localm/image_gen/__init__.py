"""
localm.image_gen - local image generation via ComfyUI FLUX.

Public API::

    from localm.image_gen.comfy import generate_image
    ok, message = generate_image("a sunset over the ocean", Path("out.png"))
"""

from .comfy import generate_image

__all__ = ["generate_image"]
