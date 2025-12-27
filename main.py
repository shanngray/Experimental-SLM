"""Main entry point for training the transformer language model.

This script orchestrates all Phase 1 components (tokenizer, dataset, dataloader,
model, trainer, checkpointing, evaluation, sampling) into a complete training pipeline.

Usage:
    # Train from scratch with defaults
    uv run python main.py

    # Resume from checkpoint
    uv run python main.py --resume checkpoints/checkpoint_step_1000

    # Use custom config and data
    uv run python main.py --config configs/experiment.yaml --data data/custom.txt

    # Override max steps
    uv run python main.py --max-steps 5000

    # Import model from HuggingFace
    uv run python main.py import-model Qwen/Qwen-0.5B

    # List available models
    uv run python main.py list-models

    # Show model info
    uv run python main.py model-info qwen-0.5b-base
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml
from huggingface_hub import snapshot_download, hf_hub_download, login
from transformers import AutoConfig, AutoTokenizer
from tqdm import tqdm

from src.config import TrainingConfig
from src.dataloader import DataLoader
from src.dataset import WindowDataset, split_corpus
from src.model.registry import ModelRegistry
from src.model.transformer import Transformer
from src.model.adapters.base import BaseAdapter
from src.model.adapters.custom_transformer import CustomTransformerAdapter
from src.model.adapters.qwen import QwenAdapter
from src.tokenizer import Tokenizer
from src.training.trainer import Trainer, create_optimizer


def load_config_from_yaml(config_path: str) -> TrainingConfig:
    """Load training configuration from a YAML file.
    
    Args:
        config_path: Path to YAML configuration file.
    
    Returns:
        TrainingConfig instance with values from YAML.
    
    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config file is invalid.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        if config_dict is None:
            config_dict = {}
        
        return TrainingConfig.from_dict(config_dict)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file {config_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error loading config from {config_path}: {e}")


def load_data_file(data_path: str) -> str:
    """Load text data from file.
    
    Args:
        data_path: Path to text data file.
    
    Returns:
        Text content as string.
    
    Raises:
        FileNotFoundError: If data file doesn't exist.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise IOError(f"Error reading data file {data_path}: {e}")


def find_default_data_file() -> Optional[str]:
    """Find default data file in data/ directory.
    
    Returns:
        Path to first .txt file in data/ directory, or None if not found.
    """
    data_dir = Path("data")
    if not data_dir.exists():
        return None
    
    # Look for uni-alg-int.txt first (mentioned in proposal)
    preferred_file = data_dir / "uni-alg-int.txt"
    if preferred_file.exists():
        return str(preferred_file)
    
    # Otherwise, find first .txt file
    txt_files = list(data_dir.glob("*.txt"))
    if txt_files:
        return str(txt_files[0])
    
    return None


def sanitize_model_name(model_id: str) -> str:
    """Generate a sanitized model name from HuggingFace model ID.
    
    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen-0.5B")
    
    Returns:
        Sanitized model name (e.g., "qwen-0.5b-base")
    """
    # Replace slashes and special characters with hyphens
    name = model_id.replace('/', '-').replace('_', '-').lower()
    # Remove any remaining special characters
    name = re.sub(r'[^a-z0-9-]', '', name)
    # Ensure it doesn't start or end with hyphen
    name = name.strip('-')
    return name


def get_supported_architectures() -> list[str]:
    """Get list of supported model architectures.
    
    Returns:
        List of supported architecture type strings.
    """
    return ['qwen']


def compute_model_checksum(model_dir: Path) -> str:
    """Compute SHA256 checksum of model directory.
    
    Computes checksum by hashing all model files (excluding metadata files).
    This is used for integrity validation.
    
    Args:
        model_dir: Path to model directory.
    
    Returns:
        SHA256 checksum as hex string.
    """
    sha256 = hashlib.sha256()
    
    # Files to include in checksum (model weights, configs)
    include_patterns = ['*.pt', '*.safetensors', '*.bin', 'config.json', 'tokenizer*.json', '*.model']
    exclude_patterns = ['metadata.json', '*.md', '*.txt']
    
    files_to_hash = []
    for pattern in include_patterns:
        files_to_hash.extend(model_dir.rglob(pattern))
    
    # Sort for deterministic hashing
    files_to_hash.sort()
    
    for file_path in files_to_hash:
        # Skip excluded patterns
        if any(file_path.match(excl) for excl in exclude_patterns):
            continue
        
        if file_path.is_file():
            try:
                with open(file_path, 'rb') as f:
                    # Hash file path relative to model_dir and content
                    rel_path = file_path.relative_to(model_dir)
                    sha256.update(str(rel_path).encode('utf-8'))
                    while chunk := f.read(8192):
                        sha256.update(chunk)
            except Exception:
                # Skip files that can't be read
                continue
    
    return sha256.hexdigest()


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size string.
    
    Args:
        size_bytes: Size in bytes.
    
    Returns:
        Formatted string (e.g., "1.5 GB").
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def check_architecture_support(model_id: str) -> tuple[bool, Optional[str]]:
    """Check if a model architecture is supported.
    
    Args:
        model_id: HuggingFace model ID.
    
    Returns:
        Tuple of (is_supported, architecture_type). architecture_type is None
        if model is not supported.
    """
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model_type = getattr(config, 'model_type', '').lower()
        
        supported = get_supported_architectures()
        for arch in supported:
            if arch in model_type:
                return True, arch
        
        return False, None
    except Exception as e:
        # If we can't determine architecture, assume unsupported
        return False, None


def handle_import_model(args) -> int:
    """Handle import-model subcommand.
    
    Args:
        args: Parsed arguments for import-model command.
    
    Returns:
        Exit code (0 for success, 1 for error).
    """
    model_id = args.model_id
    custom_name = args.name
    
    print(f"Importing model: {model_id}")
    
    # Check architecture support
    is_supported, architecture_type = check_architecture_support(model_id)
    if not is_supported:
        print(f"Error: Model architecture is not supported.", file=sys.stderr)
        print(f"Supported architectures: {', '.join(get_supported_architectures())}", file=sys.stderr)
        return 1
    
    print(f"Detected architecture: {architecture_type}")
    
    # Generate model name
    if custom_name:
        model_name = custom_name
    else:
        model_name = sanitize_model_name(model_id)
    
    # Check if model name already exists
    registry = ModelRegistry()
    if model_name in registry.models:
        print(f"Error: Model '{model_name}' already exists in registry.", file=sys.stderr)
        print(f"Use --name to specify a different name, or delete the existing model first.", file=sys.stderr)
        return 1
    
    # Check for HuggingFace authentication if needed
    hf_token = os.environ.get('HUGGINGFACE_HUB_TOKEN')
    if hf_token:
        try:
            login(token=hf_token)
        except Exception as e:
            print(f"Warning: Failed to login with token: {e}", file=sys.stderr)
    
    # Determine model directory
    project_root = Path(__file__).parent
    models_dir = project_root / "models"
    model_dir = models_dir / model_name
    
    # Check disk space and estimate model size
    estimated_size_gb = None
    try:
        # Get model info to estimate size
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        # Better estimate: check actual config parameters
        num_params = getattr(config, 'num_parameters', None)
        if num_params is None:
            # Fallback: estimate from architecture
            hidden_size = getattr(config, 'hidden_size', getattr(config, 'd_model', 0))
            num_layers = getattr(config, 'num_hidden_layers', getattr(config, 'n_layers', 0))
            vocab_size = getattr(config, 'vocab_size', 0)
            if hidden_size and num_layers:
                # Rough estimate: embedding + transformer layers
                num_params = vocab_size * hidden_size + num_layers * (12 * hidden_size * hidden_size)
        
        if num_params:
            # Estimate: assume 2 bytes per parameter for float16/bfloat16
            estimated_size_bytes = num_params * 2
            estimated_size_gb = estimated_size_bytes / (1024 ** 3)
            
            # Check available space
            stat = shutil.disk_usage(models_dir)
            available_bytes = stat.free
            available_gb = available_bytes / (1024 ** 3)
            
            # Improved warnings for large models
            if estimated_size_gb > 10.0:
                print(f"⚠️  Warning: Large model detected (estimated size: {format_size(estimated_size_bytes)})", file=sys.stderr)
                print(f"   This may take significant time to download and require substantial disk space.", file=sys.stderr)
                print(f"   Available space: {format_size(available_bytes)}", file=sys.stderr)
                if available_gb < estimated_size_gb * 1.5:  # 1.5x safety margin
                    print(f"   ⚠️  CRITICAL: Estimated size exceeds available space!", file=sys.stderr)
                    response = input("Continue anyway? (y/N): ")
                    if response.lower() != 'y':
                        return 1
            elif available_gb < estimated_size_gb * 1.5:
                print(f"⚠️  Warning: Estimated model size ({format_size(estimated_size_bytes)}) may exceed available space ({format_size(available_bytes)})", file=sys.stderr)
                response = input("Continue anyway? (y/N): ")
                if response.lower() != 'y':
                    return 1
    except Exception as e:
        print(f"Warning: Could not estimate model size: {e}", file=sys.stderr)
    
    # Check if model already exists in HuggingFace cache
    from huggingface_hub import try_to_load_from_cache
    cache_path = try_to_load_from_cache(repo_id=model_id, filename="config.json")
    if cache_path and Path(cache_path).exists():
        print(f"Found model in HuggingFace cache, will use cached files when possible...")
    
    # Download model with progress indication
    print(f"Downloading model from HuggingFace...")
    try:
        # Use snapshot_download with progress callback
        # Note: huggingface_hub doesn't expose direct progress callbacks, but tqdm
        # can be enabled via environment variable or we can wrap it
        # For now, we'll use a simple approach: check if files exist before download
        
        # Check if model directory already exists (from previous partial download)
        if model_dir.exists():
            print(f"Found existing directory at {model_dir}, resuming download...")
        
        # Download with progress (huggingface_hub will show progress if HF_HUB_ENABLE_HF_TRANSFER=1)
        # Enable progress bars in huggingface_hub if possible
        os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')
        
        cache_dir = snapshot_download(
            repo_id=model_id,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        # Verify download completed
        if not model_dir.exists() or not any(model_dir.iterdir()):
            raise RuntimeError("Download completed but model directory is empty")
        
        # Count downloaded files for user feedback
        num_files = sum(1 for _ in model_dir.rglob('*') if _.is_file())
        total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
        print(f"✓ Model downloaded successfully ({num_files} files, {format_size(total_size)})")
    except Exception as e:
        print(f"Error downloading model: {e}", file=sys.stderr)
        # Clean up partial download
        if model_dir.exists():
            shutil.rmtree(model_dir)
        return 1
    
    # Get license information
    license_info = "Unknown"
    try:
        readme_path = model_dir / "README.md"
        if readme_path.exists():
            # Try to extract license from README (simplified)
            with open(readme_path, 'r', encoding='utf-8') as f:
                readme_content = f.read()
                # Look for license field
                license_match = re.search(r'license[:\s]+([^\n]+)', readme_content, re.IGNORECASE)
                if license_match:
                    license_info = license_match.group(1).strip()
    except Exception:
        pass
    
    print(f"\nLicense: {license_info}")
    print("Please review the model's license terms before using.")
    response = input("Acknowledge license terms? (y/N): ")
    if response.lower() != 'y':
        print("Import cancelled.", file=sys.stderr)
        shutil.rmtree(model_dir)
        return 1
    
    # Validate model can be loaded
    print("Validating model...")
    try:
        from src.model.adapters.qwen import QwenAdapter
        # Optimize loading: use lazy loading if available
        adapter = QwenAdapter(str(model_dir))
        num_params = adapter.get_num_parameters()
        print(f"✓ Model validated successfully. Parameters: {num_params:,}")
    except Exception as e:
        print(f"Error validating model: {e}", file=sys.stderr)
        shutil.rmtree(model_dir)
        return 1
    
    # Compute checksum for integrity validation
    print("Computing checksum...")
    try:
        checksum = compute_model_checksum(model_dir)
        print(f"✓ Checksum computed: {checksum[:16]}...")
    except Exception as e:
        print(f"Warning: Could not compute checksum: {e}", file=sys.stderr)
        checksum = None
    
    # Save metadata
    metadata = {
        'license': license_info,
        'model_size': num_params,
        'architecture_type': architecture_type,
        'checksum': checksum
    }
    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Register in registry
    try:
        registry.add_model(
            model_name=model_name,
            model_id=model_id,
            architecture_type=architecture_type,
            local_path=f"models/{model_name}",
            source="huggingface",
            metadata=metadata
        )
        print(f"\nModel imported successfully!")
        print(f"Model name: {model_name}")
        print(f"Use this name in your config file: model_name: \"{model_name}\"")
        return 0
    except Exception as e:
        print(f"Error registering model: {e}", file=sys.stderr)
        shutil.rmtree(model_dir)
        return 1


def handle_list_models(args) -> int:
    """Handle list-models subcommand.
    
    Args:
        args: Parsed arguments for list-models command.
    
    Returns:
        Exit code (0 for success, 1 for error).
    """
    registry = ModelRegistry()
    
    models = registry.list_models(
        architecture_type=args.architecture,
        source=args.source
    )
    
    if not models:
        print("No models found in registry.")
        if args.architecture or args.source:
            print("Try removing filters to see all models.")
        return 0
    
    print(f"\nFound {len(models)} model(s):\n")
    print(f"{'Model Name':<30} {'Model ID':<30} {'Architecture':<15} {'Source':<15} {'Created':<20}")
    print("-" * 110)
    
    for model in models:
        created = model.get('created_at', 'Unknown')
        # Format date for display
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
            created = dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            pass
        
        print(f"{model['model_name']:<30} {model['model_id']:<30} {model['architecture_type']:<15} {model['source']:<15} {created:<20}")
    
    return 0


def handle_model_info(args) -> int:
    """Handle model-info subcommand.
    
    Args:
        args: Parsed arguments for model-info command.
    
    Returns:
        Exit code (0 for success, 1 for error).
    """
    registry = ModelRegistry()
    
    info = registry.get_model_info(args.model_name)
    if info is None:
        print(f"Error: Model '{args.model_name}' not found in registry.", file=sys.stderr)
        return 1
    
    print(f"\nModel Information: {args.model_name}\n")
    print(f"Model ID: {info['model_id']}")
    print(f"Architecture: {info['architecture_type']}")
    print(f"Source: {info['source']}")
    print(f"Local Path: {info['local_path']}")
    print(f"Created: {info['created_at']}")
    
    if 'fine_tuned_from' in info:
        print(f"Fine-tuned from: {info['fine_tuned_from']}")
    
    if 'total_size_mb' in info:
        print(f"Size: {info['total_size_mb']:.2f} MB")
    
    if 'metadata' in info and info['metadata']:
        print(f"\nMetadata:")
        for key, value in info['metadata'].items():
            if key == 'checksum' and value:
                # Truncate checksum for display
                print(f"  {key}: {value[:16]}...")
            else:
                print(f"  {key}: {value}")
    
    # Show fine-tuning children if any
    children = registry.get_fine_tuning_children(args.model_name)
    if children:
        print(f"\nFine-tuned models ({len(children)}):")
        for child in children:
            print(f"  - {child['model_name']} (created: {child['created_at']})")
    
    return 0


def handle_delete_model(args) -> int:
    """Handle delete-model subcommand.
    
    Args:
        args: Parsed arguments for delete-model command.
    
    Returns:
        Exit code (0 for success, 1 for error).
    """
    registry = ModelRegistry()
    
    model_name = args.model_name
    
    # Check if model exists
    if model_name not in registry.models:
        print(f"Error: Model '{model_name}' not found in registry.", file=sys.stderr)
        return 1
    
    # Get model info
    model_info = registry.get_model(model_name)
    if model_info is None:
        print(f"Error: Could not retrieve model info for '{model_name}'.", file=sys.stderr)
        return 1
    
    # Check for fine-tuning children
    children = registry.get_fine_tuning_children(model_name)
    if children:
        print(f"Warning: Model '{model_name}' has {len(children)} fine-tuned child model(s):")
        for child in children:
            print(f"  - {child['model_name']}")
        response = input(f"\nDelete '{model_name}' anyway? This will break lineage tracking. (y/N): ")
        if response.lower() != 'y':
            print("Deletion cancelled.")
            return 0
    
    # Confirm deletion
    local_path = model_info.get('local_path')
    project_root = Path(__file__).parent
    print(f"\nModel to delete: {model_name}")
    print(f"  Model ID: {model_info.get('model_id')}")
    print(f"  Local path: {local_path}")
    if local_path:
        model_path = project_root / local_path if not Path(local_path).is_absolute() else Path(local_path)
        if model_path.exists():
            # Estimate size with better formatting
            try:
                total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
                print(f"  Disk usage: {format_size(total_size)}")
            except Exception as e:
                print(f"  Warning: Could not calculate disk usage: {e}")
    
    response = input(f"\nDelete model '{model_name}' from registry? (y/N): ")
    if response.lower() != 'y':
        print("Deletion cancelled.")
        return 0
    
    # Delete from registry
    try:
        registry.delete_model(model_name)
        print(f"Model '{model_name}' deleted from registry.")
        
        # Optionally delete files
        if local_path:
            model_path = project_root / local_path if not Path(local_path).is_absolute() else Path(local_path)
            if model_path.exists() and args.delete_files:
                response = input(f"Also delete model files at {model_path}? (y/N): ")
                if response.lower() == 'y':
                    try:
                        import shutil
                        shutil.rmtree(model_path)
                        print(f"Deleted model files from {model_path}")
                    except Exception as e:
                        print(f"Warning: Could not delete model files: {e}", file=sys.stderr)
            elif model_path.exists():
                print(f"Model files still exist at {model_path} (use --delete-files to remove them)")
        
        return 0
    except Exception as e:
        print(f"Error deleting model: {e}", file=sys.stderr)
        return 1


def handle_validate_model(args) -> int:
    """Handle validate-model subcommand.
    
    Args:
        args: Parsed arguments for validate-model command.
    
    Returns:
        Exit code (0 for success, 1 for error).
    """
    registry = ModelRegistry()
    
    model_name = args.model_name
    
    # Check if model exists
    if model_name not in registry.models:
        print(f"Error: Model '{model_name}' not found in registry.", file=sys.stderr)
        return 1
    
    model_info = registry.get_model(model_name)
    if model_info is None:
        print(f"Error: Could not retrieve model info for '{model_name}'.", file=sys.stderr)
        return 1
    
    print(f"Validating model: {model_name}\n")
    
    # Check 1: Registry entry integrity
    print("1. Checking registry entry...")
    required_fields = ['model_name', 'model_id', 'architecture_type', 'local_path', 'source', 'created_at']
    missing_fields = [field for field in required_fields if field not in model_info]
    if missing_fields:
        print(f"   ❌ Missing required fields: {', '.join(missing_fields)}")
        return 1
    else:
        print("   ✅ Registry entry is valid")
    
    # Check 2: Local path exists
    print("2. Checking local files...")
    local_path = model_info.get('local_path')
    project_root = Path(__file__).parent
    model_path = project_root / local_path if not Path(local_path).is_absolute() else Path(local_path)
    
    if not model_path.exists():
        print(f"   ❌ Model directory not found: {model_path}")
        return 1
    else:
        print(f"   ✅ Model directory exists: {model_path}")
    
    # Check 3: Checksum validation (if available)
    metadata = model_info.get('metadata', {})
    stored_checksum = metadata.get('checksum')
    if stored_checksum:
        print("3. Checking integrity (checksum)...")
        try:
            current_checksum = compute_model_checksum(model_path)
            if current_checksum == stored_checksum:
                print(f"   ✅ Checksum verified: {current_checksum[:16]}...")
            else:
                print(f"   ⚠️  Warning: Checksum mismatch!", file=sys.stderr)
                print(f"      Stored:   {stored_checksum[:16]}...", file=sys.stderr)
                print(f"      Current:  {current_checksum[:16]}...", file=sys.stderr)
                print(f"      Model files may have been modified or corrupted.", file=sys.stderr)
                response = input("   Continue validation anyway? (y/N): ")
                if response.lower() != 'y':
                    return 1
        except Exception as e:
            print(f"   ⚠️  Warning: Could not verify checksum: {e}", file=sys.stderr)
    else:
        print("3. Skipping checksum (not available for this model)")
    
    # Check 4: Required files exist
    print("4. Checking required files...")
    architecture_type = model_info.get('architecture_type')
    required_files = []
    
    if architecture_type == 'qwen':
        # Qwen models need config.json and model files
        required_files = ['config.json']
        # Check for model files (could be model.safetensors, pytorch_model.bin, etc.)
        model_files = list(model_path.glob('model*.safetensors')) + list(model_path.glob('pytorch_model*.bin'))
        if not model_files:
            print("   ❌ No model weight files found (expected model.safetensors or pytorch_model.bin)")
            return 1
        else:
            print(f"   ✅ Found model weight file: {model_files[0].name}")
    elif architecture_type == 'custom-transformer':
        required_files = ['config.json', 'model.pt']
    
    for req_file in required_files:
        file_path = model_path / req_file
        if not file_path.exists():
            print(f"   ❌ Missing required file: {req_file}")
            return 1
        else:
            print(f"   ✅ Found required file: {req_file}")
    
    # Check 5: Try to load model
    print("5. Testing model loading...")
    try:
        if architecture_type == 'qwen':
            adapter = QwenAdapter(str(model_path))
        elif architecture_type == 'custom-transformer':
            # Load config
            config_path = model_path / "config.json"
            with open(config_path, 'r') as f:
                model_config = json.load(f)
            adapter = CustomTransformerAdapter(**{k: v for k, v in model_config.items() if k != 'seed'})
            adapter.load_checkpoint(str(model_path))
        else:
            print(f"   ❌ Unsupported architecture type: {architecture_type}")
            return 1
        
        num_params = adapter.get_num_parameters()
        print(f"   ✅ Model loaded successfully ({num_params:,} parameters)")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return 1
    
    # Check 6: Test forward pass
    print("6. Testing forward pass...")
    try:
        # Create dummy input
        batch_size = 2
        seq_len = 10
        vocab_size = adapter.get_config().get('vocab_size', 1000)
        dummy_input = torch.randint(0, min(vocab_size, 1000), (batch_size, seq_len))
        
        adapter.eval()
        with torch.no_grad():
            output = adapter(dummy_input)
        
        expected_shape = (batch_size, seq_len, vocab_size)
        if output.shape != expected_shape:
            print(f"   ❌ Output shape mismatch: expected {expected_shape}, got {output.shape}")
            return 1
        else:
            print(f"   ✅ Forward pass successful (output shape: {output.shape})")
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        return 1
    
    print(f"\n✅ Model '{model_name}' validation passed!")
    return 0


def load_model_adapter(
    config: TrainingConfig,
    vocab_size: int,
    project_root: Optional[Path] = None
) -> tuple[BaseAdapter, Optional[object]]:
    """Load model adapter based on config.model_name.
    
    If config.model_name is None, creates a CustomTransformerAdapter with architecture
    params from config. Otherwise, loads model from registry and selects appropriate
    adapter based on architecture_type.
    
    Args:
        config: Training configuration.
        vocab_size: Vocabulary size (used for custom Transformer).
        project_root: Project root directory. Defaults to main.py's parent.
    
    Returns:
        Tuple of (adapter, tokenizer). Tokenizer is None for custom Transformer
        (uses default Tokenizer), or the model's native tokenizer for imported models.
    
    Raises:
        ValueError: If model_name is specified but not found in registry, or if
            model files are missing, or if architecture_type is unsupported.
    """
    if project_root is None:
        project_root = Path(__file__).parent
    
    # Case 1: Custom Transformer (backward compatible)
    if config.model_name is None:
        print("Using custom Transformer architecture")
        adapter = CustomTransformerAdapter(
            vocab_size=vocab_size,
            max_seq_len=config.max_seq_len,
            n_layers=config.n_layers,
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            seed=config.seed
        )
        return adapter, None  # None means use default Tokenizer
    
    # Case 2: Load from registry
    print(f"Loading model from registry: {config.model_name}")
    registry = ModelRegistry()
    
    model_entry = registry.get_model(config.model_name)
    if model_entry is None:
        available_models = registry.list_models()
        available_names = [m['model_name'] for m in available_models]
        raise ValueError(
            f"Model '{config.model_name}' not found in registry.\n"
            f"Available models: {', '.join(available_names) if available_names else '(none)'}\n"
            f"Import a model with: python main.py import-model <model-id>"
        )
    
    # Get model path
    local_path = model_entry.get('local_path')
    if not local_path:
        raise ValueError(
            f"Model '{config.model_name}' has no local_path in registry"
        )
    
    # Resolve path
    model_path = Path(local_path)
    if not model_path.is_absolute():
        model_path = project_root / model_path
    
    if not model_path.exists():
        raise ValueError(
            f"Model files not found at {model_path}\n"
            f"Model may have been moved or deleted. Try re-importing with:\n"
            f"python main.py import-model {model_entry.get('model_id', '<model-id>')}"
        )
    
    # Select adapter based on architecture_type
    architecture_type = model_entry.get('architecture_type')
    if architecture_type == 'qwen':
        print(f"Loading Qwen adapter from {model_path}")
        adapter = QwenAdapter(str(model_path))
        tokenizer = adapter.get_tokenizer()
        return adapter, tokenizer
    elif architecture_type == 'custom-transformer':
        # This shouldn't normally happen (custom-transformer shouldn't be in registry),
        # but handle it for completeness
        print(f"Loading custom Transformer adapter from {model_path}")
        # Load config from model directory
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise ValueError(f"Model config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        
        adapter = CustomTransformerAdapter(
            vocab_size=model_config.get('vocab_size', vocab_size),
            max_seq_len=model_config.get('max_seq_len', config.max_seq_len),
            n_layers=model_config.get('n_layers', config.n_layers),
            d_model=model_config.get('d_model', config.d_model),
            n_heads=model_config.get('n_heads', config.n_heads),
            d_ff=model_config.get('d_ff', config.d_ff),
            dropout=model_config.get('dropout', config.dropout),
            seed=model_config.get('seed', config.seed)
        )
        
        # Load weights if available
        weights_path = model_path / "model.pt"
        if weights_path.exists():
            adapter.load_checkpoint(str(model_path))
        
        return adapter, None
    else:
        raise ValueError(
            f"Unsupported architecture type: {architecture_type}\n"
            f"Supported types: qwen, custom-transformer"
        )


def parse_args():
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train a transformer language model or manage models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from scratch
  python main.py

  # Resume from checkpoint
  python main.py --resume checkpoints/checkpoint_step_1000

  # Import model from HuggingFace
  python main.py import-model Qwen/Qwen-0.5B

  # List available models
  python main.py list-models

  # Show model info
  python main.py model-info qwen-0.5b-base
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command (default, backward compatible)
    train_parser = parser.add_argument_group('Training arguments')
    train_parser.add_argument(
        '--resume',
        type=str,
        metavar='PATH',
        help='Resume training from checkpoint directory'
    )
    
    train_parser.add_argument(
        '--config',
        type=str,
        metavar='PATH',
        help='Load configuration from YAML file'
    )
    
    train_parser.add_argument(
        '--data',
        type=str,
        metavar='PATH',
        help='Path to training data file (default: data/uni-alg-int.txt)'
    )
    
    train_parser.add_argument(
        '--max-steps',
        type=int,
        metavar='N',
        help='Maximum training steps (overrides config)'
    )
    
    # Import model command
    import_parser = subparsers.add_parser(
        'import-model',
        help='Import a model from HuggingFace'
    )
    import_parser.add_argument(
        'model_id',
        type=str,
        help='HuggingFace model ID (e.g., Qwen/Qwen-0.5B)'
    )
    import_parser.add_argument(
        '--name',
        type=str,
        help='Custom model name (default: auto-generated from model ID)'
    )
    
    # List models command
    list_parser = subparsers.add_parser(
        'list-models',
        help='List all available models in registry'
    )
    list_parser.add_argument(
        '--architecture',
        type=str,
        help='Filter by architecture type (e.g., qwen)'
    )
    list_parser.add_argument(
        '--source',
        type=str,
        help='Filter by source (e.g., huggingface)'
    )
    
    # Model info command
    info_parser = subparsers.add_parser(
        'model-info',
        help='Show detailed information about a model'
    )
    info_parser.add_argument(
        'model_name',
        type=str,
        help='Model name from registry'
    )
    
    # Delete model command
    delete_parser = subparsers.add_parser(
        'delete-model',
        help='Delete a model from registry'
    )
    delete_parser.add_argument(
        'model_name',
        type=str,
        help='Model name to delete from registry'
    )
    delete_parser.add_argument(
        '--delete-files',
        action='store_true',
        help='Also delete model files from disk (prompts for confirmation)'
    )
    
    # Validate model command
    validate_parser = subparsers.add_parser(
        'validate-model',
        help='Validate model integrity and test loading'
    )
    validate_parser.add_argument(
        'model_name',
        type=str,
        help='Model name to validate'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse command-line arguments
    args = parse_args()
    
    # Handle subcommands
    if args.command == 'import-model':
        sys.exit(handle_import_model(args))
    elif args.command == 'list-models':
        sys.exit(handle_list_models(args))
    elif args.command == 'model-info':
        sys.exit(handle_model_info(args))
    elif args.command == 'delete-model':
        sys.exit(handle_delete_model(args))
    elif args.command == 'validate-model':
        sys.exit(handle_validate_model(args))
    elif args.command is not None:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)
    
    # Default: training mode (backward compatible)
    # Load configuration
    if args.config:
        try:
            config = load_config_from_yaml(args.config)
            print(f"Loaded configuration from {args.config}")
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        config = TrainingConfig()
        print("Using default configuration")
    
    # Override max_steps if provided (CLI takes precedence over config)
    max_steps = args.max_steps if args.max_steps is not None else config.max_steps
    if args.max_steps is not None:
        print(f"max_steps overridden via CLI: {max_steps} (config had {config.max_steps})")
    
    # Log key training hyperparameters
    print(f"\nTraining configuration:")
    print(f"  max_steps: {max_steps}")
    print(f"  checkpoint_cadence: {config.checkpoint_cadence}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  train_ratio: {config.train_ratio}")
    
    # Determine data file path
    if args.data:
        data_path = args.data
    else:
        data_path = find_default_data_file()
        if data_path is None:
            print("Error: No data file specified and no default data file found in data/", 
                  file=sys.stderr)
            print("Please specify data file with --data argument", file=sys.stderr)
            sys.exit(1)
    
    # Load text data
    try:
        print(f"Loading data from {data_path}...")
        text_data = load_data_file(data_path)
        print(f"Loaded {len(text_data)} characters")
    except (FileNotFoundError, IOError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Determine tokenizer and tokenize corpus
    # For imported models, we need to load adapter first to get tokenizer
    # For custom Transformer, we tokenize first to get vocab_size
    adapter = None
    tokenizer = None
    corpus = None
    vocab_size = None
    
    if config.model_name is None:
        # Custom Transformer: tokenize first, then create adapter
        print("Initializing tokenizer...")
        tokenizer = Tokenizer()
        print("Tokenizing corpus...")
        corpus = tokenizer.encode(text_data)
        vocab_size = len(tokenizer.char_to_id)
        print(f"Tokenized corpus: {len(corpus)} tokens, vocab_size: {vocab_size}")
        
        # Create custom Transformer adapter
        print("Creating custom Transformer adapter...")
        adapter = CustomTransformerAdapter(
            vocab_size=vocab_size,
            max_seq_len=config.max_seq_len,
            n_layers=config.n_layers,
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            seed=config.seed
        )
    else:
        # Imported model: load adapter first to get tokenizer
        print("Loading model adapter...")
        try:
            adapter, model_tokenizer = load_model_adapter(config, vocab_size=256)  # Temporary, will be ignored
        except ValueError as e:
            print(f"Error loading model: {e}", file=sys.stderr)
            sys.exit(1)
        
        # Use model's tokenizer
        if model_tokenizer is not None:
            print("Using model's native tokenizer")
            tokenizer = model_tokenizer
        else:
            print("Warning: Model adapter returned no tokenizer, using default")
            tokenizer = Tokenizer()
        
        # Tokenize corpus
        print("Tokenizing corpus...")
        if hasattr(tokenizer, 'encode'):
            # Our Tokenizer class
            corpus = tokenizer.encode(text_data)
            vocab_size = len(tokenizer.char_to_id)
        else:
            # HuggingFace tokenizer
            # For HuggingFace tokenizers, encode the full text
            encoded = tokenizer(text_data, return_tensors=None, add_special_tokens=False)
            if isinstance(encoded, dict) and 'input_ids' in encoded:
                corpus = encoded['input_ids']
            elif isinstance(encoded, list):
                if len(encoded) > 0 and isinstance(encoded[0], list):
                    # Flatten nested list
                    corpus = [item for sublist in encoded for item in sublist]
                else:
                    corpus = encoded
            else:
                raise ValueError(f"Unexpected tokenizer output format: {type(encoded)}")
            
            # Get vocab_size from tokenizer
            if hasattr(tokenizer, 'vocab_size'):
                vocab_size = tokenizer.vocab_size
            else:
                # Fallback: estimate from corpus
                vocab_size = max(corpus) + 1 if corpus else 256
        
        print(f"Tokenized corpus: {len(corpus)} tokens, vocab_size: {vocab_size}")
    
    # Split corpus into train/val
    train_ratio = config.train_ratio
    val_ratio = 1.0 - train_ratio
    print(f"Splitting corpus into train/val ({train_ratio*100:.1f}%/{val_ratio*100:.1f}%)...")
    train_corpus, val_corpus = split_corpus(corpus, train_ratio=train_ratio, seed=config.seed or 42)
    print(f"Train: {len(train_corpus)} tokens, Val: {len(val_corpus)} tokens")
    
    # Create datasets
    print(f"Creating datasets with context_length={config.max_seq_len}...")
    train_dataset = WindowDataset(train_corpus, context_length=config.max_seq_len)
    val_dataset = WindowDataset(val_corpus, context_length=config.max_seq_len)
    print(f"Train dataset: {len(train_dataset)} windows, Val dataset: {len(val_dataset)} windows")
    
    # Create dataloaders
    print(f"Creating dataloaders with batch_size={config.batch_size}...")
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, seed=config.seed)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, seed=config.seed)
    print(f"Train batches: {len(train_dataloader)}, Val batches: {len(val_dataloader)}")
    
    # Check if resuming from checkpoint
    if args.resume:
        try:
            print(f"Loading checkpoint from {args.resume}...")
            checkpoint_path = Path(args.resume)
            
            # Load checkpoint metadata to determine model type
            metadata_path = checkpoint_path / "metadata.json"
            checkpoint_model_name = None
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    checkpoint_model_name = metadata.get('model_name')
            
            # If checkpoint has different model_name than config, warn user
            if checkpoint_model_name is not None and checkpoint_model_name != config.model_name:
                print(f"Warning: Checkpoint was saved with model_name='{checkpoint_model_name}', "
                      f"but config specifies model_name='{config.model_name}'")
                print("Loading checkpoint model instead...")
                # Update config to match checkpoint
                config.model_name = checkpoint_model_name
                # Reload adapter with correct model_name
                adapter, model_tokenizer = load_model_adapter(config, vocab_size=vocab_size)
                if model_tokenizer is not None:
                    tokenizer = model_tokenizer
            
            # Load model weights using adapter's load_checkpoint
            try:
                adapter.load_checkpoint(str(checkpoint_path))
                print("Loaded model weights from checkpoint")
            except Exception as e:
                print(f"Warning: Could not load model weights from checkpoint: {e}")
                print("Continuing with initialized model...")
            
            # Create optimizer
            optimizer = create_optimizer(adapter, config)
            
            # Load optimizer state if available
            optimizer_path = checkpoint_path / "optimizer.pt"
            if optimizer_path.exists():
                try:
                    optimizer_state = torch.load(optimizer_path, map_location='cpu')
                    optimizer.load_state_dict(optimizer_state)
                    print("Loaded optimizer state from checkpoint")
                except Exception as e:
                    print(f"Warning: Could not load optimizer state: {e}")
            
            # Create trainer
            trainer = Trainer(
                model=adapter,
                optimizer=optimizer,
                config=config,
                val_dataloader=val_dataloader,
                tokenizer=tokenizer
            )
            
            # Load step counter from metadata
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    trainer.step = metadata.get('step', 0)
                    print(f"Resumed from step {trainer.step}")
            else:
                print("No metadata found in checkpoint, starting from step 0")
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            print(f"Error loading checkpoint: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Model adapter already created above
        print("Model adapter ready")
        num_params = adapter.get_num_parameters()
        print(f"Model parameters: {num_params:,}")
        print(f"Architecture: {adapter.get_architecture_type()}")
        
        print("Creating optimizer...")
        optimizer = create_optimizer(adapter, config)
        
        # Create trainer
        trainer = Trainer(
            model=adapter,
            optimizer=optimizer,
            config=config,
            val_dataloader=val_dataloader,
            tokenizer=tokenizer
        )
        print("Trainer initialized")
    
    # Helper function to get model metadata for checkpointing
    def get_model_metadata_for_checkpoint():
        """Get model metadata from registry for checkpointing."""
        model_name = config.model_name
        model_id = None
        model_source = None
        fine_tuned_from = None
        
        # If resuming from checkpoint, this is fine-tuning
        # The fine_tuned_from should be the model_name from the checkpoint we're resuming from
        if args.resume:
            checkpoint_path = Path(args.resume)
            metadata_path = checkpoint_path / "metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        checkpoint_metadata = json.load(f)
                        # Use checkpoint's model_name as fine_tuned_from
                        checkpoint_model_name = checkpoint_metadata.get('model_name')
                        if checkpoint_model_name:
                            fine_tuned_from = checkpoint_model_name
                except Exception:
                    pass  # Ignore errors reading checkpoint metadata
        
        # Get model metadata from registry if model_name is set
        if model_name:
            try:
                registry = ModelRegistry()
                model_entry = registry.get_model(model_name)
                if model_entry:
                    model_id = model_entry.get('model_id')
                    model_source = model_entry.get('source', 'custom')
                    # Only use registry's fine_tuned_from if we're not resuming (not fine-tuning)
                    if fine_tuned_from is None:
                        fine_tuned_from = model_entry.get('fine_tuned_from')
            except Exception:
                pass  # Ignore errors, use None values
        else:
            # Custom Transformer - source is "custom"
            model_source = "custom"
        
        return model_name, model_id, model_source, fine_tuned_from
    
    # Training loop
    print(f"\nStarting training for {max_steps} steps...")
    if config.checkpoint_cadence is not None:
        print(f"Checkpoints will be saved every {config.checkpoint_cadence} steps")
    else:
        print("Periodic checkpointing is disabled (checkpoint_cadence is None)")
    print("Press Ctrl+C to interrupt and save checkpoint\n")
    
    # Get model metadata once at start
    model_name_meta, model_id_meta, model_source_meta, fine_tuned_from_meta = get_model_metadata_for_checkpoint()
    
    try:
        step_count = 0
        while trainer.step < max_steps:
            for batch in train_dataloader:
                if trainer.step >= max_steps:
                    break
                
                # Training step
                loss = trainer.training_step(batch)
                step_count += 1
                
                # Save checkpoint periodically
                if (config.checkpoint_cadence is not None and 
                    trainer.step > 0 and 
                    trainer.step % config.checkpoint_cadence == 0):
                    checkpoint_path = trainer.save_checkpoint(
                        tokenizer,
                        model_name=model_name_meta,
                        model_id=model_id_meta,
                        model_source=model_source_meta,
                        fine_tuned_from=fine_tuned_from_meta
                    )
                    print(f"Checkpoint saved to {checkpoint_path}")
            
            # If we've completed an epoch, continue from start
            if trainer.step < max_steps:
                continue
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint before exit...")
        try:
            checkpoint_path = trainer.save_checkpoint(
                tokenizer,
                model_name=model_name_meta,
                model_id=model_id_meta,
                model_source=model_source_meta,
                fine_tuned_from=fine_tuned_from_meta
            )
            print(f"Checkpoint saved to {checkpoint_path}")
            print("Training can be resumed with: python main.py --resume", checkpoint_path)
        except Exception as e:
            print(f"Error saving checkpoint: {e}", file=sys.stderr)
            sys.exit(1)
        sys.exit(0)
    
    # Training completed
    print(f"\nTraining completed! Final step: {trainer.step}")
    
    # Save final checkpoint
    print("Saving final checkpoint...")
    checkpoint_path = trainer.save_checkpoint(
        tokenizer,
        checkpoint_name="final",
        model_name=model_name_meta,
        model_id=model_id_meta,
        model_source=model_source_meta,
        fine_tuned_from=fine_tuned_from_meta
    )
    print(f"Final checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
