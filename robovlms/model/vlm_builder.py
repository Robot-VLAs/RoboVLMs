import copy
import transformers
import torch

from robovlms.utils.model_utils import build_tokenizer


def build_vlm(vlm_config, tokenizer_config, precision="bf16"):
    vlm_config = copy.deepcopy(vlm_config)
    model_path = vlm_config.get("pretrained_model_name_or_path")
    model_name = vlm_config.get("name")
    model_type = vlm_config.get("type", "AutoModel")
    
    print(f"Debug - Model name: {model_name}, Model path: {model_path}, Model type: {model_type}")
    
    if model_name == "paligemma":
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            revision="bfloat16",
        )
        tokenizer = AutoProcessor.from_pretrained(model_path)
    elif model_name == "llava":
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        model_base = None  # default is None
        model_path = vlm_config.get("pretrained_model_name_or_path")
        model_family_name = get_model_name_from_path(model_path)
        tokenizer, model, _, __ = load_pretrained_model(
            model_path,
            model_base,
            model_family_name,
            use_flash_attn=False,
            device_map="cpu",
        )
    elif model_name.lower() == "qwen":
        print(f"Loading Qwen VL model from {model_path}")
        
        # Try loading as Qwen2-VL first (regardless of path name)
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            print("Attempting to load as Qwen2-VL model...")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                trust_remote_code=True,
            )
            tokenizer = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            print("Successfully loaded as Qwen2-VL!")
            return tokenizer, model
        except Exception as e:
            print(f"Failed to load as Qwen2-VL: {e}")
            print("Falling back to original Qwen-VL loading method...")
            
            # Fall back to original Qwen-VL
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                trust_remote_code=True,
            )
            tokenizer = build_tokenizer(tokenizer_config)
            
        return tokenizer, model
    else:
        print(f"Building {model_name} from {model_path} with {model_type}")
        model = getattr(transformers, model_type).from_pretrained(
            model_path, trust_remote_code=True
        )
        tokenizer = build_tokenizer(tokenizer_config)

    return tokenizer, model
