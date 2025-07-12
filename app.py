import gradio as gr
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import os
import uuid

# --- Configuration ---
DATA_DIR = "/app/data"
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
# Set environment variables to use a writable cache directory on our persistent volume
os.environ['HF_HOME'] = os.path.join(DATA_DIR, "huggingface_cache")
os.environ['MPLCONFIGDIR'] = "/tmp/matplotlib"

# --- Model Configuration ---
device = "cuda"
dtype = torch.float16
base_model_id = "emilianJR/epiCRealism"
repo_id = "ByteDance/AnimateDiff-Lightning"

# --- Global variable for the loaded pipeline ---
pipe = None

# --- Startup: Load the model into memory ---
def load_pipeline(steps):
    global pipe
    print(f"Loading AnimateDiff Lightning {steps}-step model...")
    
    try:
        adapter_ckpt = f"animatediff_lightning_{steps}step_diffusers.safetensors"
        motion_adapter_path = hf_hub_download(repo_id, adapter_ckpt)
        
        adapter = MotionAdapter.from_pretrained(repo_id, torch_dtype=dtype)
        adapter.load_state_dict(load_file(motion_adapter_path, device=device))
        
        pipe = AnimateDiffPipeline.from_pretrained(
            base_model_id,
            motion_adapter=adapter,
            torch_dtype=dtype
        ).to(device)
        
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear"
        )
        print("Model loaded successfully.")
        return f"Model loaded: {steps}-step"
    except Exception as e:
        print(f"❌ FAILED TO LOAD MODEL: {e}")
        # Use raise gr.Error() to display the error in the Gradio UI
        raise gr.Error(f"Failed to load model: {e}")

# --- Main Generation Function ---
def generate_animation(prompt, steps, guidance_scale):
    if pipe is None:
        raise gr.Error("Model is not loaded. Please select a model version from the dropdown first.")
    if not prompt:
        raise gr.Error("Prompt cannot be empty.")

    print(f"Generating video for prompt: '{prompt}' with {steps} steps.")
    
    try:
        output = pipe(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps
        )
        frames = output.frames[0]

        os.makedirs(VIDEOS_DIR, exist_ok=True)
        filename = f"{uuid.uuid4()}.gif"
        output_path = os.path.join(VIDEOS_DIR, filename)
        
        export_to_gif(frames, output_path)
        
        print(f"Animation saved to persistent storage at: {output_path}")
        return output_path
    except Exception as e:
        print(f"An error occurred during generation: {e}")
        raise gr.Error(f"Failed to generate animation. Error: {e}")

# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("<div style='text-align: center; font-size: 32px; font-weight: bold;'>AnimateDiff Lightning</div>")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Configuration")
            
            model_steps = gr.Dropdown(
                label="Model Version (Steps)",
                choices=[1, 2, 4, 8],
                value=4,
                info="Select the model version. Higher steps can improve quality but may be slower."
            )
            
            status_box = gr.Textbox(label="Model Status", value="Not loaded", interactive=False)

            prompt_input = gr.Textbox(
                label="Prompt", 
                placeholder="A girl smiling",
                lines=3
            )
            
            with gr.Accordion("Advanced Options", open=False):
                guidance_scale = gr.Slider(
                    label="Guidance Scale", 
                    minimum=1.0, 
                    maximum=5.0, 
                    value=1.0, 
                    step=0.1,
                    info="Lower values give the model more creative freedom."
                )

            submit_btn = gr.Button("Generate Animation", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### Generated Animation")
            output_video = gr.Video(label="Result", interactive=False, height=512)

    # When the dropdown changes, load the corresponding model
    model_steps.change(
        fn=load_pipeline,
        inputs=[model_steps],
        outputs=[status_box]
    )
    
    # When the button is clicked, generate the animation
    submit_btn.click(
        fn=generate_animation,
        inputs=[prompt_input, model_steps, guidance_scale],
        outputs=[output_video]
    )
    
    # Load the default model when the app starts
    demo.load(
        fn=load_pipeline,
        inputs=[model_steps],
        outputs=[status_box]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False, server_port=7860)
