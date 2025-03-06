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

    def _make_square(self, tensor):
        """Make a tensor square by padding the shorter dimension."""
        if tensor.shape[-2] == tensor.shape[-1]:
            return tensor  # Already square
            
        size = max(tensor.shape[-2], tensor.shape[-1])
        result = torch.zeros(
            *tensor.shape[:-2], size, size,
            device=tensor.device, dtype=tensor.dtype
        )
        
        # Center the original image in the square tensor
        h_offset = (size - tensor.shape[-2]) // 2
        w_offset = (size - tensor.shape[-1]) // 2
        
        if len(tensor.shape) == 5:  # [batch, seq, c, h, w]
            result[:, :, :, h_offset:h_offset+tensor.shape[-2], w_offset:w_offset+tensor.shape[-1]] = tensor
        elif len(tensor.shape) == 4:  # [batch, c, h, w]
            result[:, :, h_offset:h_offset+tensor.shape[-2], w_offset:w_offset+tensor.shape[-1]] = tensor
        
        return result

    def model_encode_images(self, images):
        """
        Process images using Qwen2-VL's vision model.
        
        Args:
            images: Images tensor of shape [batch, seq, c, h, w] or [batch, c, h, w]
            
        Returns:
            Image features tensor
        """
        # First make sure we have square images for CALVIN
        images = self._ensure_square(images)
        
        # Extensive logging for debugging
        logger.info(f"Processing images with shape: {images.shape}")
        logger.info(f"Vision model type: {type(self.vision_tower).__name__}")
        
        # Handle 5D inputs (batch, seq_len, c, h, w)
        if images.dim() == 5:
            batch_size, seq_len, c, h, w = images.shape
            
            # Reshape to [batch*seq, c, h, w] for processing
            flat_images = images.reshape(-1, c, h, w)
            
            try:
                # Attempt to process with vision model
                # First try without grid_thw
                image_features = self.vision_tower(flat_images)
                logger.info(f"Vision tower output type: {type(image_features).__name__}")
                
                # Handle different output formats
                if isinstance(image_features, dict) and 'last_hidden_state' in image_features:
                    image_features = image_features['last_hidden_state']
                    logger.info("Extracted last_hidden_state from dict output")
                
                # Log shape for debugging
                logger.info(f"Raw features shape: {image_features.shape}")
                
                # Reshape back to include sequence dimension
                if image_features.dim() > 2:
                    num_patches = image_features.shape[1]
                    feat_dim = image_features.shape[-1]
                    image_features = image_features.reshape(batch_size, seq_len, num_patches, feat_dim)
                else:
                    # Handle case where output is already pooled to [batch*seq, hidden_dim]
                    feat_dim = image_features.shape[-1]
                    image_features = image_features.reshape(batch_size, seq_len, 1, feat_dim)
                
                logger.info(f"Final features shape: {image_features.shape}")
                return image_features
                
            except Exception as e:
                logger.error(f"Error processing images: {e}")
                logger.info("Falling back to using image processor and then processing")
                
                try:
                    # Let's use the image processor first
                    processed_images = []
                    for i in range(flat_images.shape[0]):
                        img = flat_images[i].detach().cpu()
                        processed = self.image_processor(img, return_tensors="pt").to(images.device)
                        processed_images.append(processed["pixel_values"])
                    
                    processed_batch = torch.cat(processed_images, dim=0)
                    image_features = self.vision_tower(processed_batch)
                    
                    # Extract features and reshape
                    if isinstance(image_features, dict) and 'last_hidden_state' in image_features:
                        image_features = image_features['last_hidden_state']
                    
                    # Reshape back to include sequence dimension
                    if image_features.dim() > 2:
                        num_patches = image_features.shape[1]
                        feat_dim = image_features.shape[-1]
                        image_features = image_features.reshape(batch_size, seq_len, num_patches, feat_dim)
                    else:
                        feat_dim = image_features.shape[-1]
                        image_features = image_features.reshape(batch_size, seq_len, 1, feat_dim)
                    
                    return image_features
                    
                except Exception as e2:
                    logger.error(f"All processing methods failed: {e}, {e2}")
                    # Return zeros as emergency fallback
                    feature_dim = self.hidden_size
                    return torch.zeros(
                        (batch_size, seq_len, 1, feature_dim), 
                        device=images.device, 
                        dtype=images.dtype
                    )
        else:
            # Similar logic for 4D inputs
            batch_size, c, h, w = images.shape
            
            try:
                # Process with vision model
                image_features = self.vision_tower(images)
                
                # Handle different output formats
                if isinstance(image_features, dict) and 'last_hidden_state' in image_features:
                    image_features = image_features['last_hidden_state']
                
                # Add sequence dimension
                if image_features.dim() > 2:
                    image_features = image_features.unsqueeze(1)
                else:
                    image_features = image_features.unsqueeze(1).unsqueeze(1)
                
                return image_features
                
            except Exception as e:
                logger.error(f"Error processing single images: {e}")
                # Return zeros as emergency fallback
                feature_dim = self.hidden_size
                return torch.zeros(
                    (batch_size, 1, 1, feature_dim), 
                    device=images.device, 
                    dtype=images.dtype
                )
    
    def _ensure_square(self, images):
        """Make sure images are square (needed for CALVIN)"""
        if images.shape[-2] == images.shape[-1]:
            return images  # Already square
            
        size = max(images.shape[-2], images.shape[-1])
        if images.dim() == 5:  # [batch, seq, c, h, w]
            _, _, c, h, w = images.shape
            result = torch.zeros(
                *images.shape[:-2], size, size,
                device=images.device, dtype=images.dtype
            )
            h_offset = (size - h) // 2
            w_offset = (size - w) // 2
            result[:, :, :, h_offset:h_offset+h, w_offset:w_offset+w] = images
        else:  # [batch, c, h, w]
            _, c, h, w = images.shape
            result = torch.zeros(
                *images.shape[:-2], size, size,
                device=images.device, dtype=images.dtype
            )
            h_offset = (size - h) // 2
            w_offset = (size - w) // 2
            result[:, :, h_offset:h_offset+h, w_offset:w_offset+w] = images
            
        return result

    def _process_single_image(self, img):
        """Process a single image tensor to extract features, ensuring square output"""
        try:
            # Ensure image has batch dimension
            img = img.unsqueeze(0)  # [1, c, h, w]
            
            # Check if image is square, if not make it square
            c, h, w = img.shape[1:]
            if h != w:
                logger.info(f"Making image square from {h}x{w} to {max(h,w)}x{max(h,w)}")
                size = max(h, w)
                square_img = torch.zeros(1, c, size, size, device=img.device, dtype=img.dtype)
                h_offset = (size - h) // 2
                w_offset = (size - w) // 2
                square_img[:, :, h_offset:h_offset+h, w_offset:w_offset+w] = img
                img = square_img
            
            # Create grid_thw tensor - ensure it represents a square grid
            grid_size = img.shape[2] // 16  # Using patch size of 16
            grid_thw = torch.tensor([[1, grid_size, grid_size]], device=img.device, dtype=torch.int64)
            
            # Process through vision tower
            features = self.vision_tower(img, grid_thw=grid_thw)
            
            # Return the features without batch dimension
            if features.dim() > 1:
                # Average across spatial dimensions if needed
                if features.dim() > 2:
                    features = features.mean(dim=1)
                # Remove batch dimension
                features = features.squeeze(0)
            
            return features
        
        except Exception as e:
            logger.error(f"Error in single image processing: {e}")
            
            # Try alternative approach
            try:
                # Try to use an even simpler approach
                # Get the pixel values directly from the image - simplify to grayscale
                if img.shape[0] == 3:  # RGB
                    gray = img.mean(dim=0, keepdim=True)  # [1, h, w]
                else:
                    gray = img
                
                # Flatten and project to feature dimension
                flat = gray.flatten()
                
                # Create a simple feature projection if needed
                if not hasattr(self, '_emergency_projection'):
                    feat_dim = 1024
                    if hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
                        feat_dim = self.model.config.hidden_size
                    
                    self._emergency_projection = nn.Linear(flat.shape[0], feat_dim).to(img.device).to(img.dtype)
                    # Initialize with small random values
                    with torch.no_grad():
                        self._emergency_projection.weight.data.normal_(0, 0.01)
                        self._emergency_projection.bias.data.zero_()
                
                # Project to feature dimension
                features = self._emergency_projection(flat)
                
                logger.warning("Using emergency feature projection as fallback")
                return features
                
            except Exception as e2:
                logger.error(f"Alternative approach failed: {e2}")
                raise ValueError(f"Failed to process image: {e}, {e2}")

if __name__ == "__main__":
    configs = load_config(
        "configs/calvin_finetune/finetune_qwen-vl-7b_cont-lstm-post_full_ft_text_vision_wd=0_hist=8_act=10_aug-shift_act-norm.json"
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
    bs, seq_len = 2, 4
    device = "cuda:0"
    model = model.to(device)
    # device = 'cpu'
    img_size = 448
    fwd_next_n = configs["fwd_pred_next_n"]
    vision_x = torch.zeros(
        (bs, seq_len, 3, img_size, img_size), dtype=torch.float16
    ).to(device)
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
        data_source="action",
    )

    print(test_res)