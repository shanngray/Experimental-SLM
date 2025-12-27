"""Adapter for Qwen architecture models.

This adapter wraps HuggingFace Qwen models to provide the BaseAdapter interface,
enabling fine-tuning and inference with Qwen models imported from HuggingFace.
"""

import json
from pathlib import Path
from typing import Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from src.model.adapters.base import BaseAdapter


class QwenAdapter(BaseAdapter):
    """Adapter for Qwen architecture models.
    
    Wraps HuggingFace Qwen models to provide the BaseAdapter interface,
    handling Qwen-specific details like layer naming and tokenizer integration.
    
    Attributes:
        model: The underlying HuggingFace Qwen model.
        tokenizer: The Qwen tokenizer.
        model_path: Path to the model directory.
        architecture_type: Always "qwen".
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None
    ):
        """Initialize QwenAdapter from local model directory.
        
        Args:
            model_path: Path to model directory containing weights, config, and tokenizer.
            device: Device to load model on ("cpu", "cuda", etc.). Defaults to "cpu".
        
        Raises:
            FileNotFoundError: If model directory or required files don't exist.
            ValueError: If model files are invalid or architecture is not Qwen.
        """
        super().__init__()
        
        if device is None:
            device = "cpu"
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        # Load model config
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")
        
        # Load HuggingFace config
        try:
            hf_config = AutoConfig.from_pretrained(
                str(self.model_path),
                trust_remote_code=True
            )
        except Exception as e:
            raise ValueError(f"Failed to load model config: {e}") from e
        
        # Verify this is a Qwen model
        model_type = getattr(hf_config, 'model_type', '').lower()
        if 'qwen' not in model_type:
            raise ValueError(
                f"Model type '{model_type}' is not a Qwen model. "
                f"QwenAdapter only supports Qwen architectures."
            )
        
        # Load model weights with optimizations
        try:
            # Use low_cpu_mem_usage for faster loading and lower memory footprint
            # Use safetensors if available (faster and safer)
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float32,
                device_map=device,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_safetensors=True  # Prefer safetensors format
            )
        except Exception as e:
            # Fallback if safetensors not available
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    torch_dtype=torch.float32,
                    device_map=device,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            except Exception as e2:
                raise ValueError(f"Failed to load Qwen model: {e2}") from e2
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        except Exception as e:
            raise ValueError(f"Failed to load Qwen tokenizer: {e}") from e
        
        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self._architecture_type = "qwen"
        self._device = device
        self._hf_config = hf_config
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through the Qwen model.
        
        Args:
            input_ids: Input tensor of shape [batch_size, seq_len] with token IDs.
            attention_mask: Optional attention mask tensor of shape [batch_size, seq_len].
                Values should be 0 for masked positions and 1 for unmasked positions.
        
        Returns:
            Logits tensor of shape [batch_size, seq_len, vocab_size].
        """
        # Qwen models expect input_ids and optionally attention_mask
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Extract logits from outputs
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        return logits
    
    def get_config(self) -> Dict:
        """Get model configuration.
        
        Returns:
            Dictionary containing model configuration parameters.
        """
        # Convert HuggingFace config to dict
        config_dict = self._hf_config.to_dict()
        
        # Add adapter-specific metadata
        config_dict['architecture_type'] = self._architecture_type
        config_dict['model_path'] = str(self.model_path)
        
        return config_dict
    
    def save_checkpoint(self, path: str) -> None:
        """Save model state to disk.
        
        Args:
            path: Directory path where checkpoint should be saved.
        """
        checkpoint_dir = Path(path)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model weights using HuggingFace save_pretrained
        self.model.save_pretrained(str(checkpoint_dir))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(str(checkpoint_dir))
        
        # Save adapter metadata
        metadata = {
            'architecture_type': self._architecture_type,
            'model_path': str(self.model_path),
            'config': self.get_config()
        }
        metadata_path = checkpoint_dir / "adapter_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_checkpoint(self, path: str) -> None:
        """Load model state from disk.
        
        Args:
            path: Directory path where checkpoint is stored.
        """
        checkpoint_dir = Path(path)
        
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        # Load model weights using HuggingFace from_pretrained
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                str(checkpoint_dir),
                torch_dtype=torch.float32,
                device_map=self._device,
                trust_remote_code=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model checkpoint: {e}") from e
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer checkpoint: {e}") from e
        
        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def get_architecture_type(self) -> str:
        """Get architecture type identifier.
        
        Returns:
            "qwen"
        """
        return self._architecture_type
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters in the model.
        
        Returns:
            Total parameter count as integer.
        """
        return sum(p.numel() for p in self.model.parameters())
    
    def get_tokenizer(self):
        """Get the tokenizer associated with this model.
        
        Returns:
            Qwen tokenizer object.
        """
        return self.tokenizer
    
    def parameters(self):
        """Get model parameters for optimizer.
        
        Returns:
            Iterator over model parameters.
        """
        return self.model.parameters()
    
    def train(self, mode: bool = True):
        """Set training mode.
        
        Args:
            mode: If True, set to training mode; if False, set to eval mode.
        
        Returns:
            Self for method chaining.
        """
        self.model.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode.
        
        Returns:
            Self for method chaining.
        """
        return self.train(False)

