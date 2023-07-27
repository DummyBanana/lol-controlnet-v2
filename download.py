# This file runs during container build time to get model weights built into the container

# In this example: A Huggingface Stable Diffusion custom model
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

import torch

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    # this should match the model load used in app.py's init function

    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    model = StableDiffusionControlNetPipeline.from_pretrained(
        "nitrosocke/Arcane-Diffusion", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
    )

if __name__ == "__main__":
    download_model()
