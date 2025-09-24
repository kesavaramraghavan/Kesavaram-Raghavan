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
    "A futuristic AI brain made of glowing circuits and neural networks, floating in a dark space with digital code and data streams surrounding it, ultra-detailed, hyper-realistic, high-definition 8K, cinematic lighting, sci-fi atmosphere, holographic interface elements, vibrant colors."
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
