from io import BytesIO
import numpy as np
from PIL import Image
import cv2
import base64
import torch
import base64
from diffusers.utils import load_image
from diffusers import DDPMScheduler, DiffusionPipeline,UniPCMultistepScheduler, StableDiffusionControlNetPipeline, ControlNetModel
from potassium import Potassium, Request, Response

# create a new Potassium app
app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    model = StableDiffusionControlNetPipeline.from_pretrained(
        "DummyBanana/lol-diffusions", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
    ).to("cuda")
   
    context = {
        "model": model,
    }

    return context

# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    
   #prompt = request.json.get("prompt")
    model = context.get("model")
    model_inputs = request.json
   # outputs = model(prompt)

    #return Response(
    #    json = {"outputs": outputs}, 
    #    status=200
    #)
    prompt = model_inputs.get('prompt', None)
    negative_prompt = model_inputs.get('negative_prompt', None)
    num_images_per_prompt = model_inputs.get('num_images_per_prompt',1)
    num_inference_steps = model_inputs.get('num_inference_steps', 20)
    image_data = model_inputs.get('image_data', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB") 
    image = np.array(image)
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)

    canny_image = Image.fromarray(image)
    buffered = BytesIO()
    canny_image.save(buffered,format="JPEG")
    canny_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    model.scheduler = UniPCMultistepScheduler.from_config(model.scheduler.config)
    model.enable_model_cpu_offload()
    model.enable_xformers_memory_efficient_attention()
    output = model(
        prompt,
        canny_image,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
        num_inference_steps=num_inference_steps
    )
    restuls_arr = []
    image = output.images[0]
    for image in output.images:
        buffered = BytesIO()
        image.save(buffered,format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        restuls_arr.append(image_base64)
    
    # Return the results as a dictionary
    #return {
    #    'canny_base64': canny_base64,
    #    'image_base64': restuls_arr
    #}
    return Response(
        json = {
        'canny_base64': canny_base64,
        'image_base64': restuls_arr
    }, 
        status=200
    )

if __name__ == "__main__":
    app.serve()
