from pathlib import Path
from localm.image_gen.comfy import generate_image

def main():
    prompt = "A cyberpunk crossover portrait of a man with white hair, a short beard, and yellow eyes, metallic cybernetic implants on his face, wearing a red leather jacket with embedded light strips. Shot at night on a wet street, realistic skin texture with visible pores and wrinkles, cinematic lighting, raw photo."
    output_file = "a_geralt.png"
    guidance = 3.0  # Set your guidance scale here (e.g., 2.5 - 3.0 for photorealism, 3.5 for default)
    
    # LoRA settings — place file in your ComfyUI loras directory
    lora_name = "aidmaNSFWunlock-FLUX-V0.2.safetensors"
    lora_strength = 0.6
    
    print("Connecting to ComfyUI and generating image...")
    print(f"Prompt: {prompt}")
    print(f"Guidance Scale: {guidance}")
    if lora_name:
        print(f"LoRA: {lora_name} (Strength: {lora_strength})")
    print(f"Target Output: {output_file}\n")
    
    cwd = Path(__file__).parent
    out_p = cwd / output_file
    ok, message = generate_image(
        prompt,
        out_p,
        guidance=guidance,
        lora_name=lora_name,
        lora_strength=lora_strength,
    )

    print("\n--- Result ---")
    if ok:
        print("SUCCESS!")
        print(message)
        if out_p.exists():
            print(f"Verified: Output file exists at {out_p.resolve()}")
    else:
        print("FAILURE!")
        print(message)

if __name__ == "__main__":
    main()
