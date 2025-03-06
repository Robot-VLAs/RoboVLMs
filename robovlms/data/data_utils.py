import os
from typing import List, Literal
import sys
from io import BytesIO
import base64
from PIL import Image
from tqdm import tqdm
import json
import csv
from einops import rearrange, repeat
import numpy as np
import logging
import logging.handlers
import requests
import random

import torch
import torch.nn as nn
from torch.utils.data import default_collate
import torch.nn.functional as F
from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)


class RandomShiftsSingleAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(1, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift = shift.repeat(n, 1, 1, 1)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    @torch.no_grad()
    def forward(self, x):
        assert isinstance(x, torch.Tensor) and len(x.size()) == 4
        x = x.float()
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class RandomShiftsSingleAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(1, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift = shift.repeat(n, 1, 1, 1)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


def collate_with_none(batch):
    assert isinstance(batch[0], dict)

    delete_keys = set()
    data_type = None
    for k in batch[0]:
        if batch[0][k] is None:
            delete_keys.add(k)
        elif "data_type" in batch[0]:
            data_type = batch[0]["data_type"]

    delete_keys.add("data_type")
    for k in delete_keys:
        for d in batch:
            d.pop(k, None)

    collated = default_collate(batch)
    for k in delete_keys:
        collated[k] = None
    collated["data_type"] = data_type

    return collated


def list_files(folders: List[str]) -> List[str]:
    files = []
    for folder in folders:
        if os.path.isdir(folder):
            files.extend([os.path.join(folder, d) for d in os.listdir(folder)])
        elif os.path.isfile(folder):
            files.append(folder)
        else:
            print("Path {} is invalid".format(folder))
            sys.stdout.flush()
    return files


def list_all_files(dirs, verbose=False):
    sub_dirs = list_files(dirs)
    all_files = []
    all_dirs = []

    if verbose:
        _iter = tqdm(sub_dirs)
    else:
        _iter = sub_dirs

    for d in _iter:
        if os.path.isdir(d):
            all_dirs.append(d)
        else:
            all_files.append(d)

    if all_dirs:
        all_files.extend(list_all_files(all_dirs))
    return all_files


def list_dir_with_cache(data_dir, cache_dir=None, verbose=True):
    from robovlms.utils.dist_train import get_rank

    data_dir = data_dir.rstrip("/")

    if cache_dir is None:
        _parent_dir = os.path.dirname(data_dir)
        _base_name = os.path.basename(data_dir)
        _cache_file = os.path.join(_parent_dir, _base_name + f"_filelist.json")
    else:
        max_name_length = os.pathconf("/", "PC_NAME_MAX")
        _cache_name = data_dir.strip("/").replace("/", "_") + ".json"
        _cache_name = _cache_name[-max_name_length:]
        os.makedirs(cache_dir, exist_ok=True)
        _cache_file = os.path.join(cache_dir, _cache_name)

    if os.path.exists(_cache_file):
        if get_rank() == 0 and verbose:
            print(f"Loading data list from {_cache_file}...")

        with open(_cache_file) as f:
            return json.load(f)

    else:
        verbose = get_rank() == 0 and verbose
        data_list = list_all_files([data_dir], verbose=verbose)
        _temp_cache = _cache_file + f".rank{str(get_rank())}"
        max_name_length = os.pathconf("/", "PC_NAME_MAX")
        _temp_cache = _temp_cache[-max_name_length:]
        with open(_temp_cache, "w") as f:
            json.dump(data_list, f)
        if not os.path.exists(_cache_file):
            import shutil

            shutil.move(_temp_cache, _cache_file)

    return data_list


def grouping(data_list, num_group):
    groups = [[] for _ in range(num_group)]
    for i, d in enumerate(data_list):
        groups[i % num_group].append(d)
    return groups


def b64_2_img(data):
    buff = BytesIO(base64.b64decode(data))
    return Image.open(buff)


def read_csv(rpath, encoding=None, **kwargs):
    if rpath.startswith("hdfs"):
        raise NotImplementedError
    cfg_args = dict(delimiter=",")
    cfg_args.update(kwargs)
    try:
        data = []
        with open(rpath, encoding=encoding) as csv_file:
            csv_reader = csv.reader(csv_file, **cfg_args)
            columns = next(csv_reader)
            for row in csv_reader:
                data.append(dict(zip(columns, row)))
        return data
    except:
        return []


def claw_matrix(n, k, device="cpu"):
    upper_triangle_matrix = torch.triu(torch.ones(n, n), diagonal=0).to(device)
    lower_triangle_matrix = torch.tril(torch.ones(n, n), diagonal=k).to(device)

    claw = upper_triangle_matrix * lower_triangle_matrix

    return claw


def generate_chunck_data(data, window_size, chunk_size):
    if data is None:
        return None
    bs, seq_len = data.shape[:2]
    raw_data_shape = data.shape[2:]
    data_flatten = data.flatten().view(bs, seq_len, -1)
    assert (
        seq_len == window_size + chunk_size
    ), f"The sequence length should be {window_size + chunk_size}"
    data_flatten = repeat(data_flatten, "b s d -> b w s d", w=window_size)

    mask = claw_matrix(seq_len, chunk_size - 1, data_flatten.device)
    # mask = mask - torch.diag_embed(mask.diag()) # set current obs mask to 0
    mask = mask[:window_size].bool()

    mask = repeat(mask, "w s -> b w s d", b=bs, d=data_flatten.shape[-1])
    data_flatten = torch.masked_select(data_flatten, mask)

    # data_flatten = data_flatten.view(bs, window_size, chunk_size, *raw_data_shape)
    data_flatten = data_flatten.view(bs, window_size, chunk_size, *raw_data_shape)

    return data_flatten


def get_text_function(tokenizer, tokenizer_type, max_length=256):
    import functools

    if tokenizer_type == "flamingo":

        def preprocess_text_flamingo(sample, tokenizer):
            tokenizer.padding_side = "right"
            sample = [
                (f"<image>{s.strip()}<|endofchunk|>{tokenizer.eos_token}")
                for s in sample
            ]
            text = tokenizer(
                sample,
                max_length=max_length,
                padding="longest",
                truncation="only_first",
                return_tensors="pt",
            )
            return text["input_ids"], text["attention_mask"]

        return functools.partial(preprocess_text_flamingo, tokenizer=tokenizer)
    elif tokenizer_type == "llava":
        DEFAULT_IMAGE_TOKEN = "<image>"

        def preprocess_text_llava(sample, tokenizer):
            # tokenizer.padding_side = "right"
            # sample = [
            #     (f"{tokenizer.bos_token}{DEFAULT_IMAGE_TOKEN}{s.strip()}{tokenizer.eos_token}") for s in sample
            # ]
            sample = [(f"{s.strip()}") for s in sample]
            text = tokenizer(
                sample,
                max_length=2048,
                padding="longest",
                return_tensors="pt",
            )
            return text["input_ids"], text["attention_mask"]

        return functools.partial(preprocess_text_llava, tokenizer=tokenizer)
    elif tokenizer_type == "llava_orig":
        DEFAULT_IMAGE_TOKEN = "<image>"

        def preprocess_text_llava(sample, tokenizer):
            system_prompt = "You are a helpful language and vision assistant. \
            You are able to understand the visual content that the user provides, \
            and assist the user with a variety of tasks using natural language."
            rgb_prompt = "This is the static camera observation: "
            gripper_prompt = "This is the egocentric girpper camera observation: "
            tokenizer.padding_side = "right"
            sample = [
                (
                    f"{tokenizer.bos_token}{system_prompt}{rgb_prompt}{DEFAULT_IMAGE_TOKEN}{gripper_prompt}{DEFAULT_IMAGE_TOKEN}{s.strip()}{tokenizer.eos_token}"
                )
                for s in sample
            ]
            text = tokenizer(
                sample,
                max_length=max_length,
                padding="longest",
                truncation="only_first",
                return_tensors="pt",
            )

            return text["input_ids"], text["attention_mask"]

        return functools.partial(preprocess_text_llava, tokenizer=tokenizer)
    elif tokenizer_type == "qwen":

        def preprocess_text_qwen(texts, tokenizer):
            """
            Preprocess text for Qwen2-VL model
            """
            # Fix for Qwen2TokenizerFast which doesn't have eod_id
            if hasattr(tokenizer, 'eod_id'):
                tokenizer.pad_token_id = tokenizer.eod_id
            else:
                # For Qwen2TokenizerFast, use eos_token_id instead
                tokenizer.pad_token_id = tokenizer.eos_token_id
            
            # Process all the texts
            encodings_dict = tokenizer(
                texts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=512,
            )

            input_ids = encodings_dict["input_ids"]
            attention_mask = encodings_dict["attention_mask"]
            
            return input_ids, attention_mask

        return functools.partial(preprocess_text_qwen, tokenizer=tokenizer)
    elif tokenizer_type == "kosmos":

        def preprocess_text_kosmos(sample, tokenizer):
            tokenizer.padding_side = "right"
            sample = [(f"<grounding>An image of a robot {s.strip()}") for s in sample]
            text = tokenizer(
                sample,
                truncation="only_first",
                return_tensors="pt",
                padding="longest",
                max_length=512,
            )
            return text["input_ids"], text["attention_mask"]

        return functools.partial(preprocess_text_kosmos, tokenizer=tokenizer)
    elif tokenizer_type == "moondream":

        def preprocess_text_moondream(sample, tokenizer):
            tokenizer.padding_side = "right"
            sample = [(f"<|endoftext|>{s.strip()}") for s in sample]
            text = tokenizer(
                sample,
                truncation="only_first",
                return_tensors="pt",
                padding="longest",
                max_length=512,
                add_special_tokens=True,
            )
            return text["input_ids"], text["attention_mask"]

        return functools.partial(preprocess_text_moondream, tokenizer=tokenizer)
    elif tokenizer_type == "uform":

        def preprocess_text_uform(sample, tokenizer):
            tokenizer.padding_side = "right"
            sample = [(f"<image> {s.strip()}") for s in sample]
            text = tokenizer(
                sample,
                truncation="only_first",
                return_tensors="pt",
                padding="longest",
                max_length=512,
                add_special_tokens=True,
            )
            return text["input_ids"], text["attention_mask"]

        return functools.partial(preprocess_text_uform, tokenizer=tokenizer)
    elif tokenizer_type == "paligemma":

        def preprocess_text_paligemma(sample, tokenizer):
            tokenizer.padding_side = "right"
            sample = [(f"{tokenizer.eos_token}{s.strip()}\n") for s in sample]
            text = tokenizer(
                sample,
                truncation="only_first",
                return_tensors="pt",
                padding="longest",
                max_length=512,
                add_special_tokens=False,
            )
            return text["input_ids"], text["attention_mask"]

        return functools.partial(preprocess_text_paligemma, tokenizer=tokenizer)
    else:

        def preprocess_text_default(sample, tokenizer):
            tokenizer.padding_side = "right"
            sample = [(f"<|endoftext|>{s.strip()}") for s in sample]
            text = tokenizer(
                sample,
                truncation="only_first",
                return_tensors="pt",
                padding="longest",
                max_length=512,
                add_special_tokens=True,
            )
            return text["input_ids"], text["attention_mask"]

        return functools.partial(preprocess_text_default, tokenizer=tokenizer)


def preprocess_image(sample, image_processor, model_type=None, **kwargs):
    """
    Preprocess images with the given image processor.
    
    Args:
        sample: List of images to process
        image_processor: Image processor from the model
        model_type: Model type (for compatibility with existing code)
        **kwargs: Additional arguments for flexibility
        
    Returns:
        Processed image tensors compatible with the expected format
    """
    import torch
    import logging
    import inspect
    from transformers.image_processing_utils import BatchFeature
    
    logger = logging.getLogger(__name__)
    
    # Ensure sample is in the right format
    if not isinstance(sample, list):
        sample = [sample]
    
    # Check if this is being called from CALVIN dataset
    is_calvin = False
    for frame in inspect.stack():
        if 'calvin' in frame.filename.lower():
            is_calvin = True
            break
    
    try:
        # Determine the processor type by checking specific attributes
        processor_name = type(image_processor).__name__
        is_qwen2 = 'Qwen2' in processor_name
        
        if is_qwen2:
            logger.debug(f"Using Qwen2-VL processing with processor: {processor_name}")
            
            # Qwen2-VL requires return_tensors="pt" to get tensor outputs
            processed_images = []
            for img in sample:
                # Process with explicit tensor return
                result = image_processor(img, return_tensors="pt")
                
                # Extract pixel_values from BatchFeature
                if isinstance(result, BatchFeature) and "pixel_values" in result:
                    pixel_values = result["pixel_values"]
                    
                    # Calculate grid_thw for Qwen2-VL vision model
                    # Note: This is needed for the forward pass of the vision model
                    # The grid_thw should be calculated based on the image size and patch size
                    # Typically patch size is 16 for vision transformers
                    img_h, img_w = pixel_values.shape[-2], pixel_values.shape[-1]
                    patch_size = 16  # Standard patch size for most vision models
                    grid_h, grid_w = img_h // patch_size, img_w // patch_size
                    
                    # Store the grid_thw information with the tensor
                    pixel_values.grid_thw = torch.tensor(
                        [[1, grid_h, grid_w]], 
                        dtype=torch.int64, 
                        device=pixel_values.device
                    )
                    
                    processed_images.append(pixel_values)
                elif isinstance(result, dict) and "pixel_values" in result:
                    pixel_values = result["pixel_values"]
                    
                    # Same grid_thw calculation for dict result
                    img_h, img_w = pixel_values.shape[-2], pixel_values.shape[-1]
                    patch_size = 16
                    grid_h, grid_w = img_h // patch_size, img_w // patch_size
                    
                    pixel_values.grid_thw = torch.tensor(
                        [[1, grid_h, grid_w]], 
                        dtype=torch.int64, 
                        device=pixel_values.device
                    )
                    
                    processed_images.append(pixel_values)
                else:
                    logger.warning(f"Unexpected result from processor: {type(result)}")
                    # Try to handle as is if it's already a tensor
                    if torch.is_tensor(result):
                        tensor = result.unsqueeze(0) if result.dim() == 3 else result
                        
                        # Add grid_thw information
                        img_h, img_w = tensor.shape[-2], tensor.shape[-1]
                        patch_size = 16
                        grid_h, grid_w = img_h // patch_size, img_w // patch_size
                        
                        tensor.grid_thw = torch.tensor(
                            [[1, grid_h, grid_w]], 
                            dtype=torch.int64, 
                            device=tensor.device
                        )
                        
                        processed_images.append(tensor)
            
            # Combine results
            if len(processed_images) > 1:
                output = torch.cat(processed_images, dim=0)
                
                # Preserve grid_thw information
                if hasattr(processed_images[0], 'grid_thw'):
                    # For batched images, we need to adjust the first dimension of grid_thw
                    grid_thw = processed_images[0].grid_thw.clone()
                    grid_thw[:, 0] = len(processed_images)  # Update batch size
                    output.grid_thw = grid_thw
            else:
                output = processed_images[0]
            
            # Ensure CALVIN gets the right format
            if is_calvin and output.dim() == 4:  # [batch, c, h, w]
                # For CALVIN, add sequence and camera dimensions
                output = output.unsqueeze(1)  # [batch, 1, c, h, w]
                output = output.unsqueeze(2)  # [batch, 1, 1, c, h, w]
                
                # Preserve grid_thw
                if hasattr(output, 'grid_thw'):
                    output.grid_thw = output.grid_thw
            
            logger.debug(f"Processed images with shape: {output.shape}")
            if hasattr(output, 'grid_thw'):
                logger.debug(f"With grid_thw: {output.grid_thw}")
                
            return output
            
        else:
            # Original Qwen-VL approach
            logger.debug("Using original Qwen-VL image processing approach")
            try:
                # Original processor returns tensor directly
                processed = [image_processor(s).unsqueeze(0) for s in sample]
                
                if len(processed) > 1:
                    output = torch.cat(processed, dim=0)
                else:
                    output = processed[0]
                
                # Also handle CALVIN format for original Qwen-VL
                if is_calvin and output.dim() == 4:
                    output = output.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, c, h, w]
                    
                return output
                
            except Exception as e:
                logger.error(f"Original method failed with: {e}")
                raise
                
    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        
        # Create an emergency fallback tensor
        if is_calvin:
            return torch.zeros((len(sample), 1, 1, 3, 224, 224), dtype=torch.float32)
        else:
            return torch.zeros((len(sample), 3, 224, 224), dtype=torch.float32)

def order_pick_k(lst, k):
    if len(lst) <= k:
        return lst
    rng = np.random.random(len(lst))
    index = np.argsort(rng)[:k]
    index_sort = sorted(index)
    new_lst = [lst[i] for i in index_sort]
    print(f"WARNING: total file: {len(lst)}, random pick: {k}." f" (ignored)")
    return new_lst


def build_logger(logger_name, logger_filename):
    global handler
    from robovlms.data.vid_llava_constants import LOGDIR

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when="D", utc=True, encoding="UTF-8"
        )
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == "\n":
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != "":
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ""


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
    }
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


def get_llava_image_processor(name):
    from transformers import CLIPImageProcessor

    return CLIPImageProcessor.from_pretrained(name)


def get_llava_video_processor(tokenizer, video_decode_backend="decord", num_frames=8):
    from robovlms.data.llava_vid_processor import LanguageBindVideoProcessor

    return LanguageBindVideoProcessor(video_decode_backend, num_frames, tokenizer)


def get_prompt_builder(model_name, eos=None, bos=None):
    model_family = "openvla"
    import robovlms.data.prompting

    if "vicuna" in model_name.lower():
        return data.prompting.VicunaV15ChatPromptBuilder(model_family, eos=eos, bos=bos)
    elif "mistral" in model_name.lower():
        return data.prompting.MistralInstructPromptBuilder(
            model_family, eos=eos, bos=bos
        )
    elif "llama" in model_name.lower():
        return data.prompting.LLaMa2ChatPromptBuilder(model_family, eos=eos, bos=bos)
    elif "mpt" in model_name.lower():
        return data.prompting.PhiPromptBuilder(model_family, eos=eos, bos=bos)
    elif "qwen" in model_name.lower():
        return data.prompting.QwenPromptBuilder(model_family, eos=eos, bos=bos)
    else:
        return data.prompting.PhiPromptBuilder(model_family, eos=eos, bos=bos)


def mu_law_companding(x, mu=255, maintain_last=True):
    """Applies μ-law companding to the input array."""
    last_val = x[-1]
    res = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    if maintain_last:
        res[-1] = last_val
    return res


def inverse_mu_law_companding(y, mu=255, maintain_last=True):
    """Applies the inverse of μ-law companding to the input array."""
    last_val = y[-1]
    res = np.sign(y) * (np.expm1(np.abs(y) * np.log1p(mu)) / mu)
    if maintain_last:
        res[-1] = last_val
    return res


def regularize_action(x, x_mean, x_std, eps=1e-6, maintain_last=True):
    # return a value ~ N(0, 1)
    last_val = x[-1]
    res = (x - x_mean) / (x_std + eps)
    if maintain_last:
        res[-1] = last_val
    return res


def unregularize_action(x, x_mean, x_std, eps=1e-6, maintain_last=True):
    last_val = x[-1]
    res = x * (x_std + eps) + x_mean
    if maintain_last:
        res[-1] = last_val
    return res


class PatchMask(nn.Module):
    def __init__(self, patch_size=16, mask_ratio=0.35):
        super(PatchMask, self).__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # Generate random mask coordinates.
        mask_coords = []
        for i in range(batch_size):
            for j in range(0, height, self.patch_size):
                for k in range(0, width, self.patch_size):
                    if random.random() < self.mask_ratio:
                        mask_coords.append((i, j, k))

        # Mask out the patches.
        masked_x = x.clone()
        for i, j, k in mask_coords:
            masked_x[i, :, j : j + self.patch_size, k : k + self.patch_size] = 0.0

        return masked_x


def normalize_action(action, action_min=-1, action_max=1, maintain_last=False):
    last_val = action[..., -1]
    action = np.clip(action, a_min=float(action_min), a_max=float(action_max))
    res = 2 * (action - action_min) / (action_max - action_min) - 1
    if maintain_last:
        res[..., -1] = last_val
    return res


def unnoramalize_action(action, action_min=-1, action_max=1, maintain_last=False):
    last_val = action[..., -1]
    res = 0.5 * (action + 1) * (action_max - action_min) + action_min
    if maintain_last:
        res[..., -1] = last_val
    return res


def get_chunked_episode(
    window_sample: Literal["sliding", "range"],
    left_pad: bool,
    window_size: int,
    fwd_pred_next_n: int,
    episode_idx_range: np.ndarray,
):
    if window_sample == "range":
        window_range = np.arange(window_size)
        chunk_range = np.arange(window_size + fwd_pred_next_n)
        left_pad_mask = window_range[:, None] <= chunk_range[None, :]
    else:
        left_pad_mask = np.ones((window_size, window_size + fwd_pred_next_n))

    traj_len = len(episode_idx_range)
    chunk_indices = np.broadcast_to(
        np.arange(-window_size + 1, fwd_pred_next_n + 1),
        [traj_len, window_size + fwd_pred_next_n],
    ) + np.broadcast_to(
        np.arange(traj_len)[:, None],
        [traj_len, window_size + fwd_pred_next_n],
    )
    chunk_mask = (chunk_indices >= 0) & (chunk_indices < traj_len)
    chunk_indices = np.clip(chunk_indices, 0, traj_len - 1)
    left_index = 0 if left_pad else window_size - 1
    chunk_indices = chunk_indices[left_index:]
    chunk_mask = chunk_mask[left_index:]
    if window_sample == "range":
        tile_times = chunk_indices.shape[0]
        chunk_indices = np.repeat(chunk_indices, repeats=window_size, axis=0)
        chunk_mask = np.repeat(chunk_mask, repeats=window_size, axis=0)
        chunk_mask = chunk_mask & np.tile(left_pad_mask, (tile_times, 1))

    return episode_idx_range[chunk_indices], chunk_mask


def permute_tensor_last_dim(x: torch.Tensor, insert_dim: int):
    old_permutation = list(range(x.ndim))
    new_permutation = (
        old_permutation[:insert_dim]
        + [old_permutation[-1]]
        + old_permutation[insert_dim:-1]
    )
    return x.permute(new_permutation).contiguous()


def get_tensor_chunk(x: torch.Tensor, fwd_pred_next_n: int):
    chunk_x = x.unfold(0, fwd_pred_next_n, 1)
    chunk_x = permute_tensor_last_dim(chunk_x, 1)
    return chunk_x


def pad_sequences(sequences: List[torch.Tensor], padding_value):
    # 找出最后一维的最大长度
    max_len = max(tensor.shape[-1] for tensor in sequences)

    # 对每个 tensor 在最后一维进行 padding
    padded_tensors = [
        F.pad(
            tensor,
            (0, max_len - tensor.shape[-1]),
            mode="constant",
            value=padding_value,
        )
        for tensor in sequences
    ]

    # 将 list of tensor 堆叠为一个 tensor
    return torch.stack(padded_tensors)


def world_to_tcp_frame(action, robot_obs):
    # from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles
    from .pose_transforms import euler_angles_to_matrix, matrix_to_euler_angles

    with autocast(dtype=torch.float32):
        flag = False
        if len(action.shape) == 4:
            flag = True
            b, s, f, _ = action.shape
            action = action.reshape(b, s * f, -1)
            robot_obs = robot_obs.reshape(b, s * f, -1)
        b, s, _ = action.shape
        world_T_tcp = (
            euler_angles_to_matrix(robot_obs[..., 3:6], convention="XYZ")
            .float()
            .reshape(-1, 3, 3)
        )
        tcp_T_world = torch.inverse(world_T_tcp)
        pos_w_rel = action[..., :3].reshape(-1, 3, 1)
        pos_tcp_rel = tcp_T_world @ pos_w_rel
        # downscaling is necessary here to get pseudo infinitesimal rotation
        orn_w_rel = action[..., 3:6] * 0.01
        world_T_tcp_new = (
            euler_angles_to_matrix(robot_obs[..., 3:6] + orn_w_rel, convention="XYZ")
            .float()
            .reshape(-1, 3, 3)
        )
        tcp_new_T_tcp_old = torch.inverse(world_T_tcp_new) @ world_T_tcp
        orn_tcp_rel = matrix_to_euler_angles(
            tcp_new_T_tcp_old, convention="XYZ"
        ).float()
        orn_tcp_rel = torch.where(
            orn_tcp_rel < -np.pi, orn_tcp_rel + 2 * np.pi, orn_tcp_rel
        )
        orn_tcp_rel = torch.where(
            orn_tcp_rel > np.pi, orn_tcp_rel - 2 * np.pi, orn_tcp_rel
        )
        # upscaling again
        orn_tcp_rel *= 100
        action_tcp = torch.cat(
            [
                pos_tcp_rel.reshape(b, s, -1),
                orn_tcp_rel.reshape(b, s, -1),
                action[..., -1:],
            ],
            dim=-1,
        )
        if flag:
            action_tcp = action_tcp.reshape(b, s, -1, action_tcp.shape[-1])
        assert not torch.any(action_tcp.isnan())
    return action_tcp


def tcp_to_world_frame(action, robot_obs):
    from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

    with autocast(dtype=torch.float32):
        flag = False
        if len(action.shape) == 4:
            flag = True
            b, s, f, _ = action.shape
            action = action.reshape(b, s * f, -1)
            robot_obs = robot_obs.reshape(b, s * f, -1)
        # import pdb; pdb.set_trace()
        b, s, _ = action.shape
        world_T_tcp = (
            euler_angles_to_matrix(robot_obs[..., 3:6], convention="XYZ")
            .float()
            .reshape(-1, 3, 3)
        )
        pos_tcp_rel = action[..., :3].reshape(-1, 3, 1)
        pos_w_rel = world_T_tcp @ pos_tcp_rel
        # downscaling is necessary here to get pseudo infinitesimal rotation
        orn_tcp_rel = action[..., 3:6] * 0.01
        tcp_new_T_tcp_old = (
            euler_angles_to_matrix(orn_tcp_rel, convention="XYZ")
            .float()
            .reshape(-1, 3, 3)
        )
        world_T_tcp_new = world_T_tcp @ torch.inverse(tcp_new_T_tcp_old)

        orn_w_new = matrix_to_euler_angles(world_T_tcp_new, convention="XYZ").float()
        if torch.any(orn_w_new.isnan()):
            logger.warning("NaN value in euler angles.")
            orn_w_new = matrix_to_euler_angles(
                quaternion_to_matrix(matrix_to_quaternion(world_T_tcp_new)),
                convention="XYZ",
            ).float()
        orn_w_rel = orn_w_new - robot_obs[..., 3:6].reshape(-1, 3)
        orn_w_rel = torch.where(orn_w_rel < -np.pi, orn_w_rel + 2 * np.pi, orn_w_rel)
        orn_w_rel = torch.where(orn_w_rel > np.pi, orn_w_rel - 2 * np.pi, orn_w_rel)
        # upscaling again
        orn_w_rel *= 100
        action_w = torch.cat(
            [
                pos_w_rel.reshape(b, s, -1),
                orn_w_rel.reshape(b, s, -1),
                action[..., -1:],
            ],
            dim=-1,
        )
        if flag:
            action_w = action_w.reshape(b, s, -1, action_w.shape[-1])
        assert not torch.any(action_w.isnan())
    return action_w


if __name__ == "__main__":
    # print(claw_matrix(5, 1))
    bs = 2
    seq_len = 10
    window_size = 9
    chunk_size = seq_len - window_size
    data = torch.randn(bs, seq_len, 5)
    print(data)
    print("-" * 100)
    print(generate_chunck_data(data, window_size, chunk_size))
