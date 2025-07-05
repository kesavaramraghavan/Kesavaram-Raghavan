import os
import re
import spacy
from huggingface_hub import InferenceClient
from PIL import Image
from dotenv import load_dotenv

#Load token
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not found in environment")

#Inference client
client = InferenceClient(model="black-forest-labs/FLUX.1-dev", token=HF_TOKEN)

#spaCy NLP setup
nlp = spacy.load("en_core_web_sm")

def make_short_name(prompt: str) -> str:
    doc = nlp(prompt)
    # 1) Try noun chunks
    chunks = [chunk.root.lemma_.lower() for chunk in doc.noun_chunks]
    if chunks:
        return "_".join(chunks[:2])
    # 2) Fallback to first 2 significant tokens
    tokens = [
        tok.lemma_.lower() for tok in doc
        if tok.is_alpha and not tok.is_stop and len(tok) > 3
    ]
    return "_".join(tokens[:2]) or "output"

#Your prompts
prompts = [
    "Picture a gargantuan, age-old tree rising out of a foggy valley as the first light of day paints everything gold. Rope bridges strung with glowing lanterns stretch from its mighty limbs to nearby cliffs, while miniature wooden homes nestle among the tree’s enormous roots. The whole scene feels like a majestic fantasy painting brought to life.",
    "Imagine an astronaut in a sleek suit rocking out on Mars—red dust spirals around pulsing, luminous amplifiers while neon ribbons of light dance across the sky. The colors explode in a retro-futuristic, synthwave vibe.",
    "Envision a storybook forest clearing where rabbits, foxes, and owls sit on giant mushroom stools, chatting happily. Teacups drift in mid-air, pouring glittering tea into delicate saucers, all rendered in gentle, dreamy watercolor tones.",
    "See a streamlined dragon made of silvery, flowing metal slicing through black storm clouds over a roiling sea. Flashes of lightning shimmer across its mirror-like scales, as if captured in a blockbuster film still.",
    "Step into a smoky detective’s office straight from the roaring twenties—only here, glowing neon signs flicker outside the blinds and holographic evidence hovers above an old wooden desk. Shadows and light play dramatically across the scene, blending classic noir with edgy cyberpunk flair."
]


#Generate images & save
for prompt in prompts:
    print(f"▶ Prompt: {prompt[:60]}…")
    img: Image.Image = client.text_to_image(prompt, guidance_scale=9, num_inference_steps=30)
    short = make_short_name(prompt)
    filename = f"{short}.png"
    img.save(filename)
    print(f"✅ Saved as: {filename}\n")

print("🎉 All done!")
