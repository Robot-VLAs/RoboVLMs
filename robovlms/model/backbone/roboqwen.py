import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import inspect


from robovlms.model.backbone.base_backbone import BaseRoboVLM, load_config, deep_update

logger = logging.getLogger(__name__)

class RoboQwen(BaseRoboVLM):
    @property
    def image_processor(self):
        return self.processor.image_processor

    @property
    def model(self):
        return self.backbone

    @property
    def hidden_size(self):
        return self.model.config.hidden_size

    @property
    def word_embedding(self):
        return self.model.get_input_embeddings()

    @property
    def text_tower(self):
        return self.model

    @property
    def vision_tower(self):
        # Return our stored reference to the vision model
        if hasattr(self, '_vision_model') and self._vision_model is not None:
            return self._vision_model
        
        # If we don't have a reference yet, try to find it now
        logger.warning("No vision model reference found, attempting to locate it now")
        
        # Try various possible locations in the model
        if hasattr(self.model, 'vision_model'):
            return self.model.vision_model
        elif hasattr(self.model, 'visual'):
            return self.model.visual
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'vision_tower'):
            return self.model.model.vision_tower
        
        # Last resort - check for vision in other components
        for attr_name in dir(self.model):
            if 'vis' in attr_name.lower() and not attr_name.startswith('__'):
                logger.info(f"Found potential vision component: {attr_name}")
                return getattr(self.model, attr_name)
        
        raise AttributeError("Could not find a vision component in the model!")

    @property
    def start_image_token_id(self):
        # Check multiple possible locations for the token ID
        if hasattr(self.model.config, 'vision_start_token_id'):
            return torch.LongTensor([self.model.config.vision_start_token_id])
        elif hasattr(self.model.config, 'visual') and 'image_start_id' in self.model.config.visual:
            return torch.LongTensor([self.model.config.visual['image_start_id']])
        else:
            logger.warning("Using default start image token ID (9390)")
            return torch.LongTensor([9390])  # Common default

    @property
    def end_image_token_id(self):
        if hasattr(self.model.config, 'vision_end_token_id'):
            return torch.LongTensor([self.model.config.vision_end_token_id])
        elif hasattr(self.model.config, 'visual') and 'image_start_id' in self.model.config.visual:
            return torch.LongTensor([self.model.config.visual['image_start_id'] + 1])
        else:
            logger.warning("Using default end image token ID (9391)")
            return torch.LongTensor([9391])  # Common default

    def _init_backbone(self):
        """Initialize the Qwen2-VL model and tokenizer"""
        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
        
        logger.info("Initializing Qwen2-VL backbone...")
        
        # Initialize the processor for image processing
        self.processor = Qwen2VLProcessor.from_pretrained(
            self.configs["vlm"]["pretrained_model_name_or_path"],
            trust_remote_code=True,
        )
        
        # Get the tokenizer from the processor
        tokenizer = self.processor.tokenizer
        
        # Initialize the model using Qwen2VLForConditionalGeneration
        backbone = Qwen2VLForConditionalGeneration.from_pretrained(
            self.configs["vlm"]["pretrained_model_name_or_path"],
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        # Store references to vision components
        # Print model structure to debug
        logger.info(f"Backbone attributes: {list(backbone.__dict__.keys())}")
        logger.info(f"Backbone modules: {[name for name, _ in backbone.named_modules()]}")
        
        # Attempt to locate vision components
        if hasattr(backbone, 'vision_model'):
            self._vision_model = backbone.vision_model
            logger.info("Found vision_model attribute")
        elif hasattr(backbone, 'model') and hasattr(backbone.model, 'vision_tower'):
            self._vision_model = backbone.model.vision_tower
            logger.info("Found model.vision_tower attribute")
        elif hasattr(backbone, 'vision'):
            self._vision_model = backbone.vision
            logger.info("Found vision attribute")
        else:
            # Last resort: search through all attributes
            vision_candidates = ['vision_tower', 'vision_model', 'visual', 'vision', 'image_encoder']
            found = False
            for module_name, module in backbone.named_modules():
                logger.info(f"Checking module: {module_name}, type: {type(module).__name__}")
                for candidate in vision_candidates:
                    if candidate in module_name.lower():
                        self._vision_model = module
                        logger.info(f"Found vision component at {module_name}")
                        found = True
                        break
                if found:
                    break
            
            if not found:
                logger.warning("Could not find vision component! Using a placeholder.")
                # Create a placeholder that logs access
                self._vision_model = None
        
        return tokenizer, backbone

    def encode_images(self, images):
        """Process images with Qwen-VL vision tower"""
      
        # input: images: list of b,c,h,w or b,t,c,h,w
        # output: image_features: list of bx[txn, d]
        if images.ndim == 4:
            images = images.unsqueeze(1)
        bs, seq_len = images.shape[:2]
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            images = torch.cat([image for image in images], dim=0)
        image_features = self.image_processor(images)
        image_features = image_features.view(
            bs, seq_len, -1, image_features[0].shape[-1]
        )
        # import pdb; pdb.set_trace()
        if self.use_vision_resampler:
            ### downsample at token num dim: b, s, n, d -> b, s, v d
            # b T F v d -> b, T, n, d
            image_features = self.vision_resampler(
                image_features.unsqueeze(2)
            )  # downsample v_tok per image to n_tok

        return image_features

if __name__ == "__main__":
    configs = load_config(
        "configs/calvin_finetune/qwen.json"
    )
    use_hand_rgb = False  # True
    model = RoboQwen(
        configs=configs,
        train_setup_configs=configs["train_setup"],
        fwd_head_configs=None,
        window_size=configs["window_size"],
        use_hand_rgb=use_hand_rgb,
        act_head_configs=configs["act_head"],
        fwd_pred_next_n=configs["fwd_pred_next_n"],
        use_state=True,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Qwen Model Parameters: {total_params / 1000000:.2f}M")
    bs, seq_len = 3, 8
    device = "cuda:0"
    model = model.to(device)
    # device = 'cpu'
    img_size = 448
    fwd_next_n = configs["fwd_pred_next_n"]
    vision_x = torch.zeros(
        (bs, seq_len, 3, img_size, img_size), dtype=torch.float16
    ).to(device)
    print("\n[TEST] Vision input shape:", vision_x.shape)  # <-- Add this
    print("[TEST] Vision input dimensions:", vision_x.ndim)  # <-- Add this
    vision_gripper = torch.zeros(
        (bs, seq_len, 3, img_size, img_size), dtype=torch.float16
    ).to(device)
    lang_x = torch.ones((bs, 10), dtype=torch.long).to(device) * 100
    attention_mask = torch.ones((bs, 10)).bool().to(device)
    action_lables = (
        torch.randn(bs, seq_len, fwd_next_n, 6).to(device),
        torch.zeros(bs, seq_len, fwd_next_n).to(device),
    )
    model = model.to(device).to(torch.float16)
    test_res = model(
        vision_x,
        lang_x,
        attention_mask=attention_mask,
        position_ids=None,
        use_cached_vision_x=False,
        action_labels=action_lables,
        action_mask=None,
        caption_labels=None,
        caption_mask=None,
        past_key_values=None,
        use_cache=False,
        vision_gripper=vision_gripper,
        fwd_rgb_labels=None,
        fwd_hand_rgb_labels=None,
        fwd_mask=None,
        data_source="calvin_action",
    )

    print(test_res)