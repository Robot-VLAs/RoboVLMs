import torch
import logging
import argparse
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def inspect_qwen2vl_structure(model_path):
    """Inspect the structure of Qwen2-VL model to find the vision tower"""
    logger.info(f"Loading Qwen2-VL model from: {model_path}")
    
    try:
        # Load processor
        logger.info("Loading processor...")
        processor = Qwen2VLProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        logger.info(f"Processor type: {type(processor)}")
        logger.info(f"Image processor type: {type(processor.image_processor)}")
        
        # Load model using Qwen2VLForConditionalGeneration instead of Qwen2VLModel
        logger.info("Loading model using Qwen2VLForConditionalGeneration...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        # Inspect model structure
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Model config: {model.config}")
        logger.info(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
        
        # Check for visual component
        if hasattr(model, 'visual'):
            logger.info(f"Found 'visual' attribute: {type(model.visual)}")
            logger.info(f"Visual attributes: {[attr for attr in dir(model.visual) if not attr.startswith('_')]}")
            
            # Test if it's callable
            try:
                dummy_input = torch.randn(1, 3, 224, 224, device=model.device)
                output = model.visual(dummy_input)
                logger.info(f"Visual component is callable! Output shape: {output.shape}")
            except Exception as e:
                logger.error(f"Error calling visual component: {str(e)}")
        else:
            logger.warning("No 'visual' attribute found!")
            
            # Try alternative names
            for attr_name in ['vision_model', 'vision_tower', 'vision_encoder']:
                if hasattr(model, attr_name):
                    logger.info(f"Found '{attr_name}' attribute: {type(getattr(model, attr_name))}")
        
        # Store a direct reference to the vision model
        vision_tower = model.visual if hasattr(model, 'visual') else None
        logger.info(f"Vision tower reference: {vision_tower is not None}")
        
        # Test the vision tower with a dummy input
        if vision_tower is not None:
            try:
                dummy_input = torch.randn(1, 3, 224, 224, device=model.device)
                output = vision_tower(dummy_input)
                logger.info(f"Vision tower output shape: {output.shape}")
            except Exception as e:
                logger.error(f"Error using vision tower: {str(e)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during inspection: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Qwen2-VL model structure")
    parser.add_argument("--model_path", type=str, default=".vlms/Qwen-VL", help="Path to the Qwen2-VL model")
    args = parser.parse_args()
    
    success = inspect_qwen2vl_structure(args.model_path)
    if success:
        logger.info("Inspection completed successfully")
    else:
        logger.error("Inspection failed")