from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from processingpaligemma import PaliGemmaProcessor  # Import from your local files
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os

def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, any]:
    # Load the tokenizer using the processor
    try:
        # First, try to load the processor which includes the tokenizer
        num_image_tokens = 729  # Default for PaliGemma, adjust based on your config
        image_size = 224  # Default for PaliGemma
        
        # Load tokenizer directly from the model path
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
        
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        # Fallback: Create a simple tokenizer or use a different approach
        raise ValueError(f"Could not load tokenizer from {model_path}. Please ensure tokenizer files exist.")
    
    print(f"Tokenizer loaded successfully. Padding side: {tokenizer.padding_side}")
    
    # Find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    
    # Load the model's config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)
    
    # Create the model using the configuration
    model = PaliGemmaForConditionalGeneration(config).to(device)
    
    # Load the state dict of the model
    model.load_state_dict(tensors, strict=False)
    
    # Tie weights
    model.tie_weights()
    
    return (model, tokenizer)