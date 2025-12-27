"""Tests for architecture adapters."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import torch

from src.model.adapters.base import BaseAdapter
from src.model.adapters.custom_transformer import CustomTransformerAdapter
from src.model.transformer import Transformer


class TestBaseAdapter:
    """Test BaseAdapter interface."""
    
    def test_base_adapter_is_abstract(self):
        """Test that BaseAdapter cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseAdapter()
    
    def test_base_adapter_requires_forward(self):
        """Test that adapters must implement forward."""
        class IncompleteAdapter(BaseAdapter):
            pass
        
        with pytest.raises(TypeError):
            IncompleteAdapter()


class TestCustomTransformerAdapter:
    """Test CustomTransformerAdapter."""
    
    def test_adapter_initialization(self):
        """Test adapter initialization."""
        vocab_size = 100
        adapter = CustomTransformerAdapter(vocab_size=vocab_size)
        
        assert adapter.get_architecture_type() == "custom-transformer"
        assert adapter.get_config()['vocab_size'] == vocab_size
    
    def test_adapter_forward_pass(self):
        """Test adapter forward pass."""
        vocab_size = 100
        adapter = CustomTransformerAdapter(vocab_size=vocab_size)
        
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        logits = adapter.forward(input_ids)
        
        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert torch.is_tensor(logits)
    
    def test_adapter_forward_with_attention_mask(self):
        """Test adapter forward pass with attention mask (should work but not used)."""
        vocab_size = 100
        adapter = CustomTransformerAdapter(vocab_size=vocab_size)
        
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Should not raise error (even though custom Transformer doesn't use it)
        logits = adapter.forward(input_ids, attention_mask=attention_mask)
        assert logits.shape == (batch_size, seq_len, vocab_size)
    
    def test_adapter_get_config(self):
        """Test getting adapter configuration."""
        vocab_size = 100
        max_seq_len = 512
        n_layers = 6
        d_model = 512
        n_heads = 8
        d_ff = 2048
        dropout = 0.2
        seed = 42
        
        adapter = CustomTransformerAdapter(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            seed=seed
        )
        
        config = adapter.get_config()
        assert config['vocab_size'] == vocab_size
        assert config['max_seq_len'] == max_seq_len
        assert config['n_layers'] == n_layers
        assert config['d_model'] == d_model
        assert config['n_heads'] == n_heads
        assert config['d_ff'] == d_ff
        assert config['dropout'] == dropout
        assert config['seed'] == seed
    
    def test_adapter_save_checkpoint(self):
        """Test saving checkpoint."""
        vocab_size = 100
        adapter = CustomTransformerAdapter(vocab_size=vocab_size)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint"
            
            adapter.save_checkpoint(str(checkpoint_path))
            
            # Check files were created
            assert (checkpoint_path / "model.pt").exists()
            assert (checkpoint_path / "config.json").exists()
    
    def test_adapter_load_checkpoint(self):
        """Test loading checkpoint."""
        vocab_size = 100
        adapter1 = CustomTransformerAdapter(vocab_size=vocab_size)
        
        # Get initial weights
        initial_params = {k: v.clone() for k, v in adapter1.model.named_parameters()}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint"
            adapter1.save_checkpoint(str(checkpoint_path))
            
            # Create new adapter and load
            adapter2 = CustomTransformerAdapter(vocab_size=vocab_size)
            adapter2.load_checkpoint(str(checkpoint_path))
            
            # Check weights match
            for name, param in adapter2.model.named_parameters():
                assert torch.allclose(param, initial_params[name])
    
    def test_adapter_load_checkpoint_not_found(self):
        """Test loading non-existent checkpoint raises error."""
        vocab_size = 100
        adapter = CustomTransformerAdapter(vocab_size=vocab_size)
        
        with pytest.raises(FileNotFoundError):
            adapter.load_checkpoint("/nonexistent/path")
    
    def test_adapter_get_num_parameters(self):
        """Test getting parameter count."""
        vocab_size = 100
        adapter = CustomTransformerAdapter(vocab_size=vocab_size)
        
        num_params = adapter.get_num_parameters()
        assert isinstance(num_params, int)
        assert num_params > 0
        
        # Compare with direct model
        direct_count = sum(p.numel() for p in adapter.model.parameters())
        assert num_params == direct_count
    
    def test_adapter_parameters(self):
        """Test getting parameters for optimizer."""
        vocab_size = 100
        adapter = CustomTransformerAdapter(vocab_size=vocab_size)
        
        params = list(adapter.parameters())
        assert len(params) > 0
        assert all(isinstance(p, torch.Tensor) for p in params)
    
    def test_adapter_train_mode(self):
        """Test setting training mode."""
        vocab_size = 100
        adapter = CustomTransformerAdapter(vocab_size=vocab_size)
        
        # Should start in training mode
        assert adapter.model.training
        
        # Set to eval mode
        adapter.eval()
        assert not adapter.model.training
        
        # Set back to training mode
        adapter.train()
        assert adapter.model.training
    
    def test_adapter_backward_compatibility(self):
        """Test that adapter maintains backward compatibility with Transformer."""
        vocab_size = 100
        adapter = CustomTransformerAdapter(vocab_size=vocab_size)
        
        # Should be able to access underlying model
        assert isinstance(adapter.model, Transformer)
        
        # Set to eval mode to disable dropout for deterministic comparison
        adapter.eval()
        
        # Should have same forward behavior
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        adapter_logits = adapter.forward(input_ids)
        direct_logits = adapter.model(input_ids)
        
        assert torch.allclose(adapter_logits, direct_logits)
    
    def test_adapter_get_tokenizer(self):
        """Test getting tokenizer (default returns None)."""
        vocab_size = 100
        adapter = CustomTransformerAdapter(vocab_size=vocab_size)
        
        # CustomTransformerAdapter doesn't override get_tokenizer, so should return None
        tokenizer = adapter.get_tokenizer()
        assert tokenizer is None


class TestAdapterIntegration:
    """Test adapter integration scenarios."""
    
    def test_adapter_with_optimizer(self):
        """Test adapter works with PyTorch optimizer."""
        vocab_size = 100
        adapter = CustomTransformerAdapter(vocab_size=vocab_size)
        
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4)
        
        # Do a forward and backward pass
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        logits = adapter.forward(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Should complete without error
        assert True
    
    def test_adapter_checkpoint_roundtrip(self):
        """Test save/load checkpoint roundtrip."""
        vocab_size = 100
        adapter1 = CustomTransformerAdapter(vocab_size=vocab_size)
        
        # Do some forward passes to change internal state
        input_ids = torch.randint(0, vocab_size, (2, 10))
        _ = adapter1.forward(input_ids)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint"
            adapter1.save_checkpoint(str(checkpoint_path))
            
            # Create new adapter and load
            adapter2 = CustomTransformerAdapter(vocab_size=vocab_size)
            adapter2.load_checkpoint(str(checkpoint_path))
            
            # Set both to eval mode to disable dropout for deterministic comparison
            adapter1.eval()
            adapter2.eval()
            
            # Forward pass should produce same results
            logits1 = adapter1.forward(input_ids)
            logits2 = adapter2.forward(input_ids)
            
            assert torch.allclose(logits1, logits2)


class TestQwenAdapter:
    """Test QwenAdapter (with mocked HuggingFace dependencies)."""
    
    @pytest.fixture
    def mock_qwen_model(self):
        """Create a mock Qwen model."""
        model = Mock()
        model.parameters.return_value = [torch.randn(10, 10)]
        
        # Mock forward pass
        def forward(input_ids, attention_mask=None, return_dict=True):
            batch_size, seq_len = input_ids.shape
            vocab_size = 100
            logits = torch.randn(batch_size, seq_len, vocab_size)
            mock_output = Mock()
            mock_output.logits = logits
            return mock_output
        
        model.return_value = None
        model.side_effect = forward
        model.__call__ = forward
        
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "<|endoftext|>"
        return tokenizer
    
    @pytest.fixture
    def mock_model_dir(self, tmp_path):
        """Create a temporary model directory with required files."""
        model_dir = tmp_path / "qwen-model"
        model_dir.mkdir()
        
        # Create config.json
        config = {
            "model_type": "qwen2",
            "vocab_size": 100,
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4
        }
        with open(model_dir / "config.json", 'w') as f:
            json.dump(config, f)
        
        return model_dir
    
    @patch('src.model.adapters.qwen.AutoConfig')
    @patch('src.model.adapters.qwen.AutoModelForCausalLM')
    @patch('src.model.adapters.qwen.AutoTokenizer')
    def test_qwen_adapter_initialization(
        self,
        mock_tokenizer_class,
        mock_model_class,
        mock_config_class,
        mock_model_dir,
        mock_qwen_model,
        mock_tokenizer
    ):
        """Test QwenAdapter initialization."""
        # Setup mocks
        mock_config = Mock()
        mock_config.model_type = "qwen2"
        mock_config_class.from_pretrained.return_value = mock_config
        
        mock_model_class.from_pretrained.return_value = mock_qwen_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        from src.model.adapters.qwen import QwenAdapter
        
        adapter = QwenAdapter(str(mock_model_dir))
        
        assert adapter.get_architecture_type() == "qwen"
        assert adapter.model == mock_qwen_model
        assert adapter.tokenizer == mock_tokenizer
    
    @patch('src.model.adapters.qwen.AutoConfig')
    @patch('src.model.adapters.qwen.AutoModelForCausalLM')
    @patch('src.model.adapters.qwen.AutoTokenizer')
    def test_qwen_adapter_forward_pass(
        self,
        mock_tokenizer_class,
        mock_model_class,
        mock_config_class,
        mock_model_dir,
        mock_qwen_model,
        mock_tokenizer
    ):
        """Test QwenAdapter forward pass."""
        # Setup mocks
        mock_config = Mock()
        mock_config.model_type = "qwen2"
        mock_config.to_dict.return_value = {"model_type": "qwen2"}
        mock_config_class.from_pretrained.return_value = mock_config
        
        # Mock forward to return proper logits
        def forward(input_ids, attention_mask=None, return_dict=True):
            batch_size, seq_len = input_ids.shape
            vocab_size = 100
            logits = torch.randn(batch_size, seq_len, vocab_size)
            mock_output = Mock()
            mock_output.logits = logits
            return mock_output
        
        mock_qwen_model.return_value = None
        mock_qwen_model.side_effect = forward
        mock_qwen_model.__call__ = forward
        
        mock_model_class.from_pretrained.return_value = mock_qwen_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        from src.model.adapters.qwen import QwenAdapter
        
        adapter = QwenAdapter(str(mock_model_dir))
        
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        
        logits = adapter.forward(input_ids)
        
        assert logits.shape == (batch_size, seq_len, 100)
        assert torch.is_tensor(logits)
    
    @patch('src.model.adapters.qwen.AutoConfig')
    @patch('src.model.adapters.qwen.AutoModelForCausalLM')
    @patch('src.model.adapters.qwen.AutoTokenizer')
    def test_qwen_adapter_forward_with_attention_mask(
        self,
        mock_tokenizer_class,
        mock_model_class,
        mock_config_class,
        mock_model_dir,
        mock_qwen_model,
        mock_tokenizer
    ):
        """Test QwenAdapter forward pass with attention mask."""
        # Setup mocks
        mock_config = Mock()
        mock_config.model_type = "qwen2"
        mock_config.to_dict.return_value = {"model_type": "qwen2"}
        mock_config_class.from_pretrained.return_value = mock_config
        
        def forward(input_ids, attention_mask=None, return_dict=True):
            batch_size, seq_len = input_ids.shape
            vocab_size = 100
            logits = torch.randn(batch_size, seq_len, vocab_size)
            mock_output = Mock()
            mock_output.logits = logits
            return mock_output
        
        mock_qwen_model.return_value = None
        mock_qwen_model.side_effect = forward
        mock_qwen_model.__call__ = forward
        
        mock_model_class.from_pretrained.return_value = mock_qwen_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        from src.model.adapters.qwen import QwenAdapter
        
        adapter = QwenAdapter(str(mock_model_dir))
        
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        logits = adapter.forward(input_ids, attention_mask=attention_mask)
        
        assert logits.shape == (batch_size, seq_len, 100)
    
    @patch('src.model.adapters.qwen.AutoConfig')
    @patch('src.model.adapters.qwen.AutoModelForCausalLM')
    @patch('src.model.adapters.qwen.AutoTokenizer')
    def test_qwen_adapter_get_config(
        self,
        mock_tokenizer_class,
        mock_model_class,
        mock_config_class,
        mock_model_dir,
        mock_qwen_model,
        mock_tokenizer
    ):
        """Test getting QwenAdapter configuration."""
        # Setup mocks
        mock_config = Mock()
        mock_config.model_type = "qwen2"
        mock_config.to_dict.return_value = {"model_type": "qwen2", "vocab_size": 100}
        mock_config_class.from_pretrained.return_value = mock_config
        
        mock_model_class.from_pretrained.return_value = mock_qwen_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        from src.model.adapters.qwen import QwenAdapter
        
        adapter = QwenAdapter(str(mock_model_dir))
        
        config = adapter.get_config()
        assert config['architecture_type'] == "qwen"
        assert config['model_path'] == str(mock_model_dir)
        assert 'model_type' in config
    
    @patch('src.model.adapters.qwen.AutoConfig')
    @patch('src.model.adapters.qwen.AutoModelForCausalLM')
    @patch('src.model.adapters.qwen.AutoTokenizer')
    def test_qwen_adapter_get_tokenizer(
        self,
        mock_tokenizer_class,
        mock_model_class,
        mock_config_class,
        mock_model_dir,
        mock_qwen_model,
        mock_tokenizer
    ):
        """Test getting Qwen tokenizer."""
        # Setup mocks
        mock_config = Mock()
        mock_config.model_type = "qwen2"
        mock_config.to_dict.return_value = {"model_type": "qwen2"}
        mock_config_class.from_pretrained.return_value = mock_config
        
        mock_model_class.from_pretrained.return_value = mock_qwen_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        from src.model.adapters.qwen import QwenAdapter
        
        adapter = QwenAdapter(str(mock_model_dir))
        
        tokenizer = adapter.get_tokenizer()
        assert tokenizer == mock_tokenizer
    
    @patch('src.model.adapters.qwen.AutoConfig')
    @patch('src.model.adapters.qwen.AutoModelForCausalLM')
    @patch('src.model.adapters.qwen.AutoTokenizer')
    def test_qwen_adapter_get_num_parameters(
        self,
        mock_tokenizer_class,
        mock_model_class,
        mock_config_class,
        mock_model_dir,
        mock_qwen_model,
        mock_tokenizer
    ):
        """Test getting parameter count."""
        # Setup mocks
        mock_config = Mock()
        mock_config.model_type = "qwen2"
        mock_config.to_dict.return_value = {"model_type": "qwen2"}
        mock_config_class.from_pretrained.return_value = mock_config
        
        # Mock model with known parameter count
        mock_qwen_model.parameters.return_value = [
            torch.randn(10, 10),  # 100 params
            torch.randn(5, 5)     # 25 params
        ]
        
        mock_model_class.from_pretrained.return_value = mock_qwen_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        from src.model.adapters.qwen import QwenAdapter
        
        adapter = QwenAdapter(str(mock_model_dir))
        
        num_params = adapter.get_num_parameters()
        assert num_params == 125  # 100 + 25
    
    @patch('src.model.adapters.qwen.AutoConfig')
    def test_qwen_adapter_invalid_architecture(self, mock_config_class, mock_model_dir):
        """Test that QwenAdapter rejects non-Qwen models."""
        # Setup mock config with non-Qwen model type
        mock_config = Mock()
        mock_config.model_type = "llama"
        mock_config_class.from_pretrained.return_value = mock_config
        
        from src.model.adapters.qwen import QwenAdapter
        
        with pytest.raises(ValueError, match="not a Qwen model"):
            QwenAdapter(str(mock_model_dir))
    
    @patch('src.model.adapters.qwen.AutoConfig')
    @patch('src.model.adapters.qwen.AutoModelForCausalLM')
    @patch('src.model.adapters.qwen.AutoTokenizer')
    def test_qwen_adapter_save_checkpoint(
        self,
        mock_tokenizer_class,
        mock_model_class,
        mock_config_class,
        mock_model_dir,
        mock_qwen_model,
        mock_tokenizer
    ):
        """Test saving checkpoint."""
        # Setup mocks
        mock_config = Mock()
        mock_config.model_type = "qwen2"
        mock_config.to_dict.return_value = {"model_type": "qwen2"}
        mock_config_class.from_pretrained.return_value = mock_config
        
        mock_model_class.from_pretrained.return_value = mock_qwen_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        from src.model.adapters.qwen import QwenAdapter
        
        adapter = QwenAdapter(str(mock_model_dir))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint"
            
            adapter.save_checkpoint(str(checkpoint_path))
            
            # Check that save_pretrained was called
            mock_qwen_model.save_pretrained.assert_called_once()
            mock_tokenizer.save_pretrained.assert_called_once()
            
            # Check metadata file was created
            assert (checkpoint_path / "adapter_metadata.json").exists()

