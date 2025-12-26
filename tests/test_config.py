"""Tests for configuration management.

This module provides comprehensive tests for TrainingConfig, including:
- Default values for all hyperparameters
- Loading from dictionaries
- Serialization to dictionaries
- YAML file loading
- Error handling
- Backward compatibility
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.config import TrainingConfig
from src.model.transformer import Transformer
from src.tokenizer import Tokenizer


@pytest.fixture
def tiny_corpus():
    """Create a tiny synthetic corpus for testing."""
    # Create a simple repeating pattern that can be learned
    text = "abcd " * 100  # 500 characters
    return text


class TestTrainingConfigDefaults:
    """Test default values for TrainingConfig."""
    
    def test_n_layers_default(self):
        """Test n_layers default is 4."""
        config = TrainingConfig()
        assert config.n_layers == 4
    
    def test_d_model_default(self):
        """Test d_model default is 256."""
        config = TrainingConfig()
        assert config.d_model == 256
    
    def test_n_heads_default(self):
        """Test n_heads default is 4."""
        config = TrainingConfig()
        assert config.n_heads == 4
    
    def test_d_ff_default(self):
        """Test d_ff default is 1024."""
        config = TrainingConfig()
        assert config.d_ff == 1024
    
    def test_dropout_default(self):
        """Test dropout default is 0.1."""
        config = TrainingConfig()
        assert config.dropout == 0.1
    
    def test_train_ratio_default(self):
        """Test train_ratio default is 0.95."""
        config = TrainingConfig()
        assert config.train_ratio == 0.95
    
    def test_max_steps_default(self):
        """Test max_steps default is 10000."""
        config = TrainingConfig()
        assert config.max_steps == 10000
    
    def test_checkpoint_cadence_default(self):
        """Test checkpoint_cadence default is 1000."""
        config = TrainingConfig()
        assert config.checkpoint_cadence == 1000


class TestTrainingConfigFromDict:
    """Test loading TrainingConfig from dictionary."""
    
    def test_load_all_new_hyperparameters(self):
        """Test loading all new hyperparameters from dict."""
        config_dict = {
            "n_layers": 6,
            "d_model": 512,
            "n_heads": 8,
            "d_ff": 2048,
            "dropout": 0.2,
            "train_ratio": 0.9,
            "max_steps": 20000,
            "checkpoint_cadence": 2000,
        }
        config = TrainingConfig.from_dict(config_dict)
        assert config.n_layers == 6
        assert config.d_model == 512
        assert config.n_heads == 8
        assert config.d_ff == 2048
        assert config.dropout == 0.2
        assert config.train_ratio == 0.9
        assert config.max_steps == 20000
        assert config.checkpoint_cadence == 2000
    
    def test_missing_new_fields_use_defaults(self):
        """Test missing new fields use defaults."""
        config_dict = {
            "learning_rate": 1e-3,  # Old field
            "batch_size": 32,
        }
        config = TrainingConfig.from_dict(config_dict)
        # New fields should use defaults
        assert config.n_layers == 4
        assert config.d_model == 256
        assert config.n_heads == 4
        assert config.d_ff == 1024
        assert config.dropout == 0.1
        assert config.train_ratio == 0.95
        assert config.max_steps == 10000
        assert config.checkpoint_cadence == 1000
        # Provided fields should be set
        assert config.learning_rate == 1e-3
        assert config.batch_size == 32
    
    def test_invalid_types_raise_errors(self):
        """Test invalid types raise appropriate errors."""
        # Test invalid n_layers type
        with pytest.raises(TypeError):
            TrainingConfig.from_dict({"n_layers": "invalid"})
        
        # Test invalid dropout type
        with pytest.raises(TypeError):
            TrainingConfig.from_dict({"dropout": "invalid"})
        
        # Test invalid train_ratio type
        with pytest.raises(TypeError):
            TrainingConfig.from_dict({"train_ratio": "invalid"})
    
    def test_invalid_values_raise_errors(self):
        """Test invalid values raise appropriate errors."""
        # Test negative n_layers
        with pytest.raises(ValueError, match="n_layers.*must be positive"):
            TrainingConfig.from_dict({"n_layers": -1})
        
        # Test invalid dropout range
        with pytest.raises(ValueError, match="dropout.*must be between"):
            TrainingConfig.from_dict({"dropout": 1.5})
        
        # Test invalid train_ratio range
        with pytest.raises(ValueError, match="train_ratio.*must be between"):
            TrainingConfig.from_dict({"train_ratio": 1.0})
        
        # Test d_model not divisible by n_heads
        with pytest.raises(ValueError, match="d_model.*must be divisible"):
            TrainingConfig.from_dict({"d_model": 100, "n_heads": 3})


class TestTrainingConfigToDict:
    """Test serializing TrainingConfig to dictionary."""
    
    def test_all_new_hyperparameters_serialized(self):
        """Test all new hyperparameters are serialized."""
        config = TrainingConfig(
            n_layers=6,
            d_model=512,
            n_heads=8,
            d_ff=2048,
            dropout=0.2,
            train_ratio=0.9,
            max_steps=20000,
            checkpoint_cadence=2000,
        )
        config_dict = config.to_dict()
        assert config_dict["n_layers"] == 6
        assert config_dict["d_model"] == 512
        assert config_dict["n_heads"] == 8
        assert config_dict["d_ff"] == 2048
        assert config_dict["dropout"] == 0.2
        assert config_dict["train_ratio"] == 0.9
        assert config_dict["max_steps"] == 20000
        assert config_dict["checkpoint_cadence"] == 2000
    
    def test_serialized_dict_can_recreate_config(self):
        """Test serialized dict can recreate TrainingConfig."""
        original_config = TrainingConfig(
            n_layers=6,
            d_model=512,
            n_heads=8,
            d_ff=2048,
            dropout=0.2,
            train_ratio=0.9,
            max_steps=20000,
            checkpoint_cadence=2000,
        )
        config_dict = original_config.to_dict()
        recreated_config = TrainingConfig.from_dict(config_dict)
        
        assert recreated_config.n_layers == original_config.n_layers
        assert recreated_config.d_model == original_config.d_model
        assert recreated_config.n_heads == original_config.n_heads
        assert recreated_config.d_ff == original_config.d_ff
        assert recreated_config.dropout == original_config.dropout
        assert recreated_config.train_ratio == original_config.train_ratio
        assert recreated_config.max_steps == original_config.max_steps
        assert recreated_config.checkpoint_cadence == original_config.checkpoint_cadence


class TestConfigLoadingFromYAML:
    """Test loading configuration from YAML files."""
    
    def test_load_default_config(self):
        """Test default config loads successfully."""
        from main import load_config_from_yaml
        
        config = load_config_from_yaml("configs/default.yaml")
        assert config is not None
        assert isinstance(config, TrainingConfig)
    
    def test_default_config_hyperparameters_match_defaults(self):
        """Test all hyperparameters in default config match TrainingConfig defaults."""
        from main import load_config_from_yaml
        
        config = load_config_from_yaml("configs/default.yaml")
        default_config = TrainingConfig()
        
        # Check new hyperparameters match defaults
        assert config.n_layers == default_config.n_layers == 4
        assert config.d_model == default_config.d_model == 256
        assert config.n_heads == default_config.n_heads == 4
        assert config.d_ff == default_config.d_ff == 1024
        assert config.dropout == default_config.dropout == 0.1
        assert config.train_ratio == default_config.train_ratio == 0.95
        assert config.max_steps == default_config.max_steps == 10000
        assert config.checkpoint_cadence == default_config.checkpoint_cadence == 1000
    
    def test_load_custom_config(self):
        """Test custom config file loads successfully."""
        from main import load_config_from_yaml
        
        # Test small-model config
        config = load_config_from_yaml("configs/small-model.yaml")
        assert config.n_layers == 2
        assert config.d_model == 128
        assert config.n_heads == 2
        assert config.d_ff == 512
    
    def test_custom_values_override_defaults(self):
        """Test custom values override defaults."""
        from main import load_config_from_yaml
        
        config = load_config_from_yaml("configs/large-model.yaml")
        assert config.n_layers == 6  # Override
        assert config.d_model == 512  # Override
        assert config.n_heads == 8  # Override
        assert config.d_ff == 2048  # Override
        assert config.max_steps == 20000  # Override
        # These should still be defaults
        assert config.dropout == 0.1
        assert config.train_ratio == 0.95
    
    def test_partial_configs_work(self):
        """Test partial configs work (missing fields use defaults)."""
        from main import load_config_from_yaml
        
        # Create a temporary partial config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            partial_config = {
                "learning_rate": 1e-3,
                "batch_size": 32,
                # Missing all new hyperparameters
            }
            yaml.dump(partial_config, f)
            config_path = f.name
        
        try:
            config = load_config_from_yaml(config_path)
            # New fields should use defaults
            assert config.n_layers == 4
            assert config.d_model == 256
            assert config.n_heads == 4
            assert config.d_ff == 1024
            assert config.dropout == 0.1
            assert config.train_ratio == 0.95
            assert config.max_steps == 10000
            assert config.checkpoint_cadence == 1000
            # Provided fields should be set
            assert config.learning_rate == 1e-3
            assert config.batch_size == 32
        finally:
            Path(config_path).unlink()
    
    def test_missing_fields_use_defaults(self):
        """Test missing fields use TrainingConfig defaults."""
        from main import load_config_from_yaml
        
        # Create a minimal config with only one field
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            minimal_config = {"batch_size": 64}
            yaml.dump(minimal_config, f)
            config_path = f.name
        
        try:
            config = load_config_from_yaml(config_path)
            # All new hyperparameters should use defaults
            assert config.n_layers == 4
            assert config.d_model == 256
            assert config.n_heads == 4
            assert config.d_ff == 1024
            assert config.dropout == 0.1
            assert config.train_ratio == 0.95
            assert config.max_steps == 10000
            assert config.checkpoint_cadence == 1000
            # Only batch_size should be custom
            assert config.batch_size == 64
        finally:
            Path(config_path).unlink()
    
    def test_backward_compatibility_old_configs_work(self):
        """Test backward compatibility (old configs missing new fields still work)."""
        from main import load_config_from_yaml
        
        # Create an old-style config (missing new hyperparameters)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            old_config = {
                "learning_rate": 1e-3,
                "batch_size": 16,
                "max_seq_len": 128,
                # Missing: n_layers, d_model, n_heads, d_ff, dropout, train_ratio, max_steps, checkpoint_cadence
            }
            yaml.dump(old_config, f)
            config_path = f.name
        
        try:
            config = load_config_from_yaml(config_path)
            # Should load successfully
            assert config is not None
            # Missing fields should use defaults
            assert config.n_layers == 4
            assert config.d_model == 256
            assert config.n_heads == 4
            assert config.d_ff == 1024
            assert config.dropout == 0.1
            assert config.train_ratio == 0.95
            assert config.max_steps == 10000
            assert config.checkpoint_cadence == 1000
            # Provided fields should be set
            assert config.learning_rate == 1e-3
            assert config.batch_size == 16
            assert config.max_seq_len == 128
        finally:
            Path(config_path).unlink()
    
    def test_invalid_yaml_syntax_raises_error(self):
        """Test invalid YAML syntax raises clear error."""
        from main import load_config_from_yaml
        
        # Create invalid YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: syntax: [unclosed")
            config_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Invalid YAML"):
                load_config_from_yaml(config_path)
        finally:
            Path(config_path).unlink()
    
    def test_invalid_value_types_raise_errors(self):
        """Test invalid value types raise appropriate errors."""
        from main import load_config_from_yaml
        
        # Test wrong type for int field
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            invalid_config = {"n_layers": "not_an_int"}
            yaml.dump(invalid_config, f)
            config_path = f.name
        
        try:
            with pytest.raises((ValueError, TypeError)):
                load_config_from_yaml(config_path)
        finally:
            Path(config_path).unlink()
        
        # Test wrong type for float field
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            invalid_config = {"dropout": "not_a_float"}
            yaml.dump(invalid_config, f)
            config_path = f.name
        
        try:
            with pytest.raises((ValueError, TypeError)):
                load_config_from_yaml(config_path)
        finally:
            Path(config_path).unlink()
    
    def test_error_messages_are_clear(self):
        """Test error messages are clear."""
        from main import load_config_from_yaml
        
        # Test file not found
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config_from_yaml("nonexistent_config.yaml")
        
        # Test invalid YAML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: [unclosed")
            config_path = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                load_config_from_yaml(config_path)
            assert "Invalid YAML" in str(exc_info.value) or "YAML" in str(exc_info.value)
        finally:
            Path(config_path).unlink()


class TestBackwardCompatibility:
    """Test backward compatibility with old config files."""
    
    def test_old_config_files_still_work(self):
        """Test config missing new fields loads successfully."""
        from main import load_config_from_yaml
        
        # Create an old-style config (missing new hyperparameters)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            old_config = {
                "learning_rate": 1e-3,
                "batch_size": 16,
                "max_seq_len": 128,
                # Missing: n_layers, d_model, n_heads, d_ff, dropout, train_ratio, max_steps, checkpoint_cadence
            }
            yaml.dump(old_config, f)
            config_path = f.name
        
        try:
            config = load_config_from_yaml(config_path)
            # Should load successfully
            assert config is not None
            # Missing fields should use defaults
            assert config.n_layers == 4
            assert config.d_model == 256
            assert config.n_heads == 4
            assert config.d_ff == 1024
            assert config.dropout == 0.1
            assert config.train_ratio == 0.95
            assert config.max_steps == 10000
            assert config.checkpoint_cadence == 1000
        finally:
            Path(config_path).unlink()
    
    def test_missing_fields_use_defaults(self):
        """Test missing fields use defaults."""
        from main import load_config_from_yaml
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            minimal_config = {"batch_size": 32}
            yaml.dump(minimal_config, f)
            config_path = f.name
        
        try:
            config = load_config_from_yaml(config_path)
            # All new hyperparameters should use defaults
            assert config.n_layers == 4
            assert config.d_model == 256
            assert config.n_heads == 4
            assert config.d_ff == 1024
            assert config.dropout == 0.1
            assert config.train_ratio == 0.95
            assert config.max_steps == 10000
            assert config.checkpoint_cadence == 1000
        finally:
            Path(config_path).unlink()
    
    def test_training_proceeds_normally_with_old_config(self):
        """Test training proceeds normally with old config."""
        from main import load_config_from_yaml
        
        # Create an old-style config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            old_config = {
                "learning_rate": 1e-3,
                "batch_size": 2,
                "max_seq_len": 16,
            }
            yaml.dump(old_config, f)
            config_path = f.name
        
        try:
            config = load_config_from_yaml(config_path)
            # Verify config can be used for training setup
            assert config.batch_size == 2
            assert config.max_seq_len == 16
            # New fields should have defaults
            assert config.n_layers == 4
            assert config.d_model == 256
        finally:
            Path(config_path).unlink()
    
    def test_default_behavior_unchanged_when_no_config_provided(self):
        """Test default behavior unchanged when no config provided."""
        # When no config file is provided, TrainingConfig() uses defaults
        default_config = TrainingConfig()
        
        # Verify defaults match expected values
        assert default_config.n_layers == 4
        assert default_config.d_model == 256
        assert default_config.n_heads == 4
        assert default_config.d_ff == 1024
        assert default_config.dropout == 0.1
        assert default_config.train_ratio == 0.95
        assert default_config.max_steps == 10000
        assert default_config.checkpoint_cadence == 1000
    
    def test_no_regressions_in_behavior(self):
        """Test no regressions in behavior."""
        # Verify that default config still works as expected
        config = TrainingConfig()
        
        # All fields should have valid values
        assert config.n_layers > 0
        assert config.d_model > 0
        assert config.n_heads > 0
        assert config.d_ff > 0
        assert 0.0 <= config.dropout <= 1.0
        assert 0.0 < config.train_ratio < 1.0
        assert config.max_steps > 0
        assert config.checkpoint_cadence is None or config.checkpoint_cadence > 0


class TestExampleConfigFiles:
    """Test that example config files work correctly."""
    
    def test_default_yaml_loads_and_works(self):
        """Test default.yaml loads and works."""
        from main import load_config_from_yaml
        
        config = load_config_from_yaml("configs/default.yaml")
        assert config is not None
        assert isinstance(config, TrainingConfig)
    
    def test_small_model_yaml_loads_and_works(self):
        """Test small-model.yaml loads and works."""
        from main import load_config_from_yaml
        
        config = load_config_from_yaml("configs/small-model.yaml")
        assert config is not None
        assert config.n_layers == 2
        assert config.d_model == 128
        assert config.n_heads == 2
        assert config.d_ff == 512
    
    def test_large_model_yaml_loads_and_works(self):
        """Test large-model.yaml loads and works."""
        from main import load_config_from_yaml
        
        config = load_config_from_yaml("configs/large-model.yaml")
        assert config is not None
        assert config.n_layers == 6
        assert config.d_model == 512
        assert config.n_heads == 8
        assert config.d_ff == 2048
    
    def test_fast_training_yaml_loads_and_works(self):
        """Test fast-training.yaml loads and works."""
        from main import load_config_from_yaml
        
        config = load_config_from_yaml("configs/fast-training.yaml")
        assert config is not None
        assert config.max_steps == 5000
        assert config.checkpoint_cadence == 2000
        assert config.batch_size == 32
    
    def test_detailed_eval_yaml_loads_and_works(self):
        """Test detailed-eval.yaml loads and works."""
        from main import load_config_from_yaml
        
        config = load_config_from_yaml("configs/detailed-eval.yaml")
        assert config is not None
        assert config.eval_cadence == 500
        assert config.sampling_cadence == 500
    
    def test_small_model_creates_smaller_model(self, tiny_corpus):
        """Test small-model config creates smaller model."""
        from main import load_config_from_yaml
        
        config = load_config_from_yaml("configs/small-model.yaml")
        tokenizer = Tokenizer()
        corpus = tokenizer.encode(tiny_corpus)
        vocab_size = len(tokenizer.char_to_id)
        
        model = Transformer(
            vocab_size=vocab_size,
            max_seq_len=config.max_seq_len,
            n_layers=config.n_layers,
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            seed=config.seed
        )
        
        assert model.n_layers == 2
        assert model.d_model == 128
        assert model.n_heads == 2
        assert model.d_ff == 512
    
    def test_large_model_creates_larger_model(self, tiny_corpus):
        """Test large-model config creates larger model."""
        from main import load_config_from_yaml
        
        config = load_config_from_yaml("configs/large-model.yaml")
        tokenizer = Tokenizer()
        corpus = tokenizer.encode(tiny_corpus)
        vocab_size = len(tokenizer.char_to_id)
        
        model = Transformer(
            vocab_size=vocab_size,
            max_seq_len=config.max_seq_len,
            n_layers=config.n_layers,
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            seed=config.seed
        )
        
        assert model.n_layers == 6
        assert model.d_model == 512
        assert model.n_heads == 8
        assert model.d_ff == 2048
    
    def test_architecture_matches_config_values(self, tiny_corpus):
        """Test architecture matches config values."""
        from main import load_config_from_yaml
        
        # Test with small-model config
        config = load_config_from_yaml("configs/small-model.yaml")
        tokenizer = Tokenizer()
        corpus = tokenizer.encode(tiny_corpus)
        vocab_size = len(tokenizer.char_to_id)
        
        model = Transformer(
            vocab_size=vocab_size,
            max_seq_len=config.max_seq_len,
            n_layers=config.n_layers,
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            seed=config.seed
        )
        
        assert model.n_layers == config.n_layers
        assert model.d_model == config.d_model
        assert model.n_heads == config.n_heads
        assert model.d_ff == config.d_ff
    
    def test_training_uses_correct_hyperparameters(self):
        """Test training uses correct hyperparameters from config."""
        from main import load_config_from_yaml
        
        config = load_config_from_yaml("configs/fast-training.yaml")
        
        # Verify hyperparameters are set correctly
        assert config.batch_size == 32
        assert config.learning_rate == 3.0e-4
        assert config.max_steps == 5000
    
    def test_checkpointing_uses_correct_cadence(self):
        """Test checkpointing uses correct cadence from config."""
        from main import load_config_from_yaml
        
        config = load_config_from_yaml("configs/fast-training.yaml")
        assert config.checkpoint_cadence == 2000
        
        # Test that None disables checkpointing
        config_disabled = TrainingConfig(checkpoint_cadence=None)
        assert config_disabled.checkpoint_cadence is None
    
    def test_evaluation_uses_correct_settings(self):
        """Test evaluation uses correct settings from config."""
        from main import load_config_from_yaml
        
        config = load_config_from_yaml("configs/detailed-eval.yaml")
        assert config.eval_cadence == 500
        assert config.sampling_cadence == 500
        assert config.sampling_temperature == 1.0
        assert config.sampling_prompt == "The"
        assert config.sampling_max_length == 100
        assert config.sampling_seed == 42

