import gradio as gr
import torch
import os
import sys
import gc
import uuid
from huggingface_hub import snapshot_download

# The 'wan' library is now in the same directory, so we can import it directly.
import wan
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS

# --- Configuration ---
# We use a single data directory on our persistent volume
DATA_DIR = "/app/data"
MODELS_DIR = os.path.join(DATA_DIR, "models")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_ID.split('/')[-1])

# --- Global variable for the loaded model ---
wan_i2v_model = None

# --- Startup: Download model if it doesn't exist ---
def download_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}, downloading from Hugging Face...")
        os.makedirs(MODEL_PATH, exist_ok=True)
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=MODEL_PATH,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.pt", "*.pth", "*i2v-1.3B*"]
        )
        print("Model download complete.")
    else:
        print(f"Model found at {MODEL_PATH}.")

# --- Startup: Load the model into memory ---
def load_model():
    global wan_i2v_model
    if wan_i2v_model is not None:
        print("Model is already loaded.")
        return

    print("Loading Wan-AI 14B-720P model...")
    try:
        cfg = WAN_CONFIGS['i2v-14B']
        wan_i2v_model = wan.WanI2V(
            config=cfg,
            checkpoint_dir=MODEL_PATH,
            device_id=0, rank=0, t5_fsdp=False, dit_fsdp=False, use_usp=False
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"❌ FAILED TO LOAD MODEL: {e}")
        raise

# --- Main Generation Function ---
def i2v_generation(img2vid_image, img2vid_prompt, n_prompt, sd_steps, guide_scale, shift_scale, seed):
    if wan_i2v_model is None:
        raise gr.Error("Model is not loaded. Please wait for the application to start.")
    if img2vid_image is None:
        raise gr.Error("Please upload an input image.")

    print(f"Generating video for prompt: '{img2vid_prompt}'")
    
    try:
        video_tensor = wan_i2v_model.generate(
            img2vid_prompt,
            img2vid_image,
            max_area=MAX_AREA_CONFIGS['720*1280'],
            shift=shift_scale,
            sampling_steps=sd_steps,
            guide_scale=guide_scale,
            n_prompt=n_prompt,
            seed=seed,
            offload_model=True
        )

        os.makedirs(VIDEOS_DIR, exist_ok=True)
        filename = f"{uuid.uuid4()}.mp4"
        output_path = os.path.join(VIDEOS_DIR, filename)
        
        wan.utils.utils.cache_video(
            tensor=video_tensor[None],
            save_file=output_path,
            fps=16,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )
        print(f"Video saved to persistent storage at: {output_path}")
        return output_path
    except Exception as e:
        print(f"An error occurred during generation: {e}")
        raise gr.Error(f"Failed to generate video. Error: {e}")

# --- Gradio Interface ---
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("<div style='text-align: center; font-size: 32px; font-weight: bold;'>Wan2.1 (I2V-14B)</div>")
        
        with gr.Row():
            with gr.Column():
                img2vid_image = gr.Image(type="pil", label="Upload Input Image")
                img2vid_prompt = gr.Textbox(label="Prompt", placeholder="Describe the video you want to generate")
                
                with gr.Accordion("Advanced Options", open=True):
                    n_prompt = gr.Textbox(label="Negative Prompt", value="worst quality, low quality, blurry")
                    sd_steps = gr.Slider(label="Diffusion Steps", minimum=1, maximum=100, value=50, step=1)
                    guide_scale = gr.Slider(label="Guide Scale", minimum=0, maximum=20, value=5.0, step=0.1)
                    shift_scale = gr.Slider(label="Shift Scale", minimum=0, maximum=10, value=5.0, step=0.1)
                    seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=-1)
                
                run_i2v_button = gr.Button("Generate Video")

            with gr.Column():
                result_gallery = gr.Video(label='Generated Video', interactive=False, height=600)

        run_i2v_button.click(
            fn=i2v_generation,
            inputs=[img2vid_image, img2vid_prompt, n_prompt, sd_steps, guide_scale, shift_scale, seed],
            outputs=[result_gallery]
        )
    return demo

# --- Main Execution Block ---
if __name__ == '__main__':
    download_model()
    load_model()
    demo = gradio_interface()
    demo.launch(server_name="0.0.0.0", share=False, server_port=7860)
