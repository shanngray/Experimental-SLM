"""Tests for hooks infrastructure."""

import pytest
import torch

from src.config import TrainingConfig
from src.hooks.forward_hooks import ActivationStatsHook, ForwardHook
from src.hooks.registry import HookRegistry
from src.hooks.update_hooks import IdentityUpdateHook, UpdateHook
from src.model.transformer import Transformer
from src.training.trainer import Trainer, create_optimizer


# Hook Registry Tests

def test_hook_registry_initialization():
    """Test hook registry can be initialized."""
    registry = HookRegistry()
    assert registry.forward_hooks == []
    assert registry.update_hooks == []


def test_hook_registry_load_from_config():
    """Test hook registry loads hooks from config."""
    config = {
        "hooks": {
            "forward": [
                {"name": "activation_stats", "enabled": True, "log_interval": 1}
            ],
            "update": [
                {"name": "identity", "enabled": True}
            ]
        }
    }
    registry = HookRegistry(config)
    assert len(registry.forward_hooks) == 1
    assert len(registry.update_hooks) == 1
    assert registry.forward_hooks[0][0].name == "activation_stats"
    assert registry.update_hooks[0][0].name == "identity"


def test_hook_registry_load_from_config_disabled():
    """Test hook registry loads disabled hooks from config."""
    config = {
        "hooks": {
            "forward": [
                {"name": "activation_stats", "enabled": False}
            ],
            "update": [
                {"name": "identity", "enabled": False}
            ]
        }
    }
    registry = HookRegistry(config)
    assert len(registry.forward_hooks) == 1
    assert len(registry.update_hooks) == 1
    assert registry.forward_hooks[0][1] is False  # disabled
    assert registry.update_hooks[0][1] is False  # disabled


def test_hook_registry_register_forward_hook():
    """Test hook registry can register forward hooks."""
    registry = HookRegistry()
    hook = ActivationStatsHook()
    registry.register_forward_hook(hook, enabled=True)
    assert len(registry.forward_hooks) == 1
    assert registry.forward_hooks[0][0] == hook
    assert registry.forward_hooks[0][1] is True


def test_hook_registry_register_update_hook():
    """Test hook registry can register update hooks."""
    registry = HookRegistry()
    hook = IdentityUpdateHook()
    registry.register_update_hook(hook, enabled=True)
    assert len(registry.update_hooks) == 1
    assert registry.update_hooks[0][0] == hook
    assert registry.update_hooks[0][1] is True


def test_hook_registry_toggle_forward_hook():
    """Test hook registry can enable/disable forward hooks."""
    registry = HookRegistry()
    hook = ActivationStatsHook()
    registry.register_forward_hook(hook, enabled=True)
    
    registry.disable_forward_hook(0)
    assert registry.forward_hooks[0][1] is False
    
    registry.enable_forward_hook(0)
    assert registry.forward_hooks[0][1] is True


def test_hook_registry_toggle_update_hook():
    """Test hook registry can enable/disable update hooks."""
    registry = HookRegistry()
    hook = IdentityUpdateHook()
    registry.register_update_hook(hook, enabled=True)
    
    registry.disable_update_hook(0)
    assert registry.update_hooks[0][1] is False
    
    registry.enable_update_hook(0)
    assert registry.update_hooks[0][1] is True


def test_hook_registry_toggle_invalid_index():
    """Test hook registry raises error on invalid toggle index."""
    registry = HookRegistry()
    
    with pytest.raises(IndexError):
        registry.enable_forward_hook(0)
    
    with pytest.raises(IndexError):
        registry.disable_forward_hook(0)
    
    with pytest.raises(IndexError):
        registry.enable_update_hook(0)
    
    with pytest.raises(IndexError):
        registry.disable_update_hook(0)


def test_hook_registry_call_forward_hooks():
    """Test hook registry calls enabled forward hooks."""
    registry = HookRegistry()
    
    call_count = {"count": 0}
    
    class TestHook(ForwardHook):
        @property
        def name(self):
            return "test"
        
        def __call__(self, activations):
            call_count["count"] += 1
    
    hook1 = TestHook()
    hook2 = TestHook()
    registry.register_forward_hook(hook1, enabled=True)
    registry.register_forward_hook(hook2, enabled=False)
    
    registry.call_forward_hooks(torch.randn(2, 10))
    assert call_count["count"] == 1  # Only enabled hook called


def test_hook_registry_call_update_hooks():
    """Test hook registry calls enabled update hooks in order."""
    registry = HookRegistry()
    
    call_order = []
    
    class TestHook(UpdateHook):
        def __init__(self, name):
            self._name = name
        
        @property
        def name(self):
            return self._name
        
        def __call__(self, gradients):
            call_order.append(self._name)
            return gradients
    
    hook1 = TestHook("hook1")
    hook2 = TestHook("hook2")
    hook3 = TestHook("hook3")
    
    registry.register_update_hook(hook1, enabled=True)
    registry.register_update_hook(hook2, enabled=False)
    registry.register_update_hook(hook3, enabled=True)
    
    gradients = {"param1": torch.randn(2, 3)}
    result = registry.call_update_hooks(gradients)
    
    assert call_order == ["hook1", "hook3"]  # Only enabled hooks, in order
    assert result == gradients


def test_hook_registry_get_active_hooks():
    """Test hook registry returns list of active hooks."""
    registry = HookRegistry()
    
    hook1 = ActivationStatsHook()
    hook2 = ActivationStatsHook()
    hook3 = IdentityUpdateHook()
    
    registry.register_forward_hook(hook1, enabled=True)
    registry.register_forward_hook(hook2, enabled=False)
    registry.register_update_hook(hook3, enabled=True)
    
    active = registry.get_active_hooks()
    assert active["forward"] == ["activation_stats"]
    assert active["update"] == ["identity"]


# Forward Hooks Tests

def test_activation_stats_hook_name():
    """Test activation stats hook has correct name."""
    hook = ActivationStatsHook()
    assert hook.name == "activation_stats"


def test_activation_stats_hook_logs_stats(capsys):
    """Test activation stats hook logs mean and std."""
    hook = ActivationStatsHook(log_interval=1)
    activations = torch.randn(2, 10, 100)
    
    hook(activations)
    
    captured = capsys.readouterr()
    assert "ActivationStatsHook" in captured.out
    assert "mean=" in captured.out
    assert "std=" in captured.out


def test_activation_stats_hook_log_interval(capsys):
    """Test activation stats hook respects log_interval."""
    hook = ActivationStatsHook(log_interval=2)
    activations = torch.randn(2, 10, 100)
    
    hook(activations)  # Step 1 - should not log
    captured1 = capsys.readouterr()
    assert "ActivationStatsHook" not in captured1.out
    
    hook(activations)  # Step 2 - should log
    captured2 = capsys.readouterr()
    assert "ActivationStatsHook" in captured2.out


def test_activation_stats_hook_dict_activations(capsys):
    """Test activation stats hook handles dict of activations."""
    hook = ActivationStatsHook(log_interval=1)
    activations = {
        "layer1": torch.randn(2, 10),
        "layer2": torch.randn(2, 10)
    }
    
    hook(activations)
    
    captured = capsys.readouterr()
    assert "activations[layer1]" in captured.out
    assert "activations[layer2]" in captured.out


def test_forward_hook_doesnt_modify_outputs():
    """Test forward hook doesn't modify model outputs.
    
    This test verifies that forward hooks are read-only observers that don't
    modify the logits they receive. We verify this by comparing training behavior
    with hooks enabled vs disabled - the loss should be identical, proving hooks
    don't affect the forward pass outputs.
    """
    vocab_size = 100
    batch_size = 2
    seq_len = 10
    
    # Create two identical models with dropout disabled for deterministic comparison
    # Dropout introduces non-determinism, so we disable it to test that hooks
    # don't modify outputs (rather than testing dropout behavior)
    torch.manual_seed(42)
    model1 = Transformer(vocab_size=vocab_size, dropout=0.0)
    torch.manual_seed(42)
    model2 = Transformer(vocab_size=vocab_size, dropout=0.0)
    model2.load_state_dict(model1.state_dict())
    
    config1 = TrainingConfig()
    config2 = TrainingConfig()
    
    # Model1: hooks disabled
    config1.hooks = {
        "forward": [{"name": "activation_stats", "enabled": False}],
        "update": [{"name": "identity", "enabled": True}]
    }
    
    # Model2: hooks enabled
    config2.hooks = {
        "forward": [{"name": "activation_stats", "enabled": True}],
        "update": [{"name": "identity", "enabled": True}]
    }
    
    optimizer1 = create_optimizer(model1, config1)
    optimizer2 = create_optimizer(model2, config2)
    
    trainer1 = Trainer(model1, optimizer1, config1)
    trainer2 = Trainer(model2, optimizer2, config2)
    
    # Use same inputs with fixed seed
    torch.manual_seed(123)
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Run training step on both models
    loss1 = trainer1.training_step(inputs)
    loss2 = trainer2.training_step(inputs)
    
    # Losses should be identical - this proves hooks don't modify forward pass outputs
    # If hooks modified logits, the loss would be different
    assert abs(loss1 - loss2) < 1e-6, f"Losses differ: {loss1} vs {loss2}"


def test_forward_hook_called_during_training_step(capsys):
    """Test forward hook is called during training step."""
    vocab_size = 100
    batch_size = 2
    seq_len = 10
    
    model = Transformer(vocab_size=vocab_size)
    config = TrainingConfig()
    config.hooks = {
        "forward": [{"name": "activation_stats", "enabled": True, "log_interval": 1}],
        "update": [{"name": "identity", "enabled": True}]
    }
    optimizer = create_optimizer(model, config)
    trainer = Trainer(model, optimizer, config)
    
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    trainer.training_step(inputs)
    
    captured = capsys.readouterr()
    assert "ActivationStatsHook" in captured.out


# Update Hooks Tests

def test_identity_update_hook_name():
    """Test identity update hook has correct name."""
    hook = IdentityUpdateHook()
    assert hook.name == "identity"


def test_identity_update_hook_doesnt_change_gradients():
    """Test identity update hook doesn't change gradients."""
    hook = IdentityUpdateHook()
    gradients = {
        "param1": torch.randn(2, 3),
        "param2": torch.randn(4, 5)
    }
    
    result = hook(gradients)
    
    assert result == gradients  # Same object
    assert torch.allclose(result["param1"], gradients["param1"])
    assert torch.allclose(result["param2"], gradients["param2"])


def test_update_hook_can_transform_gradients():
    """Test update hook can transform gradients."""
    class ScaleHook(UpdateHook):
        def __init__(self, scale):
            self.scale = scale
        
        @property
        def name(self):
            return "scale"
        
        def __call__(self, gradients):
            return {k: v * self.scale for k, v in gradients.items()}
    
    hook = ScaleHook(scale=2.0)
    gradients = {
        "param1": torch.tensor([1.0, 2.0]),
        "param2": torch.tensor([3.0, 4.0])
    }
    
    result = hook(gradients)
    
    assert torch.allclose(result["param1"], torch.tensor([2.0, 4.0]))
    assert torch.allclose(result["param2"], torch.tensor([6.0, 8.0]))


def test_update_hook_called_during_training_step():
    """Test update hook is called during optimizer step."""
    vocab_size = 100
    batch_size = 2
    seq_len = 10
    
    call_count = {"count": 0}
    
    class TestUpdateHook(UpdateHook):
        @property
        def name(self):
            return "test"
        
        def __call__(self, gradients):
            call_count["count"] += 1
            return gradients
    
    model = Transformer(vocab_size=vocab_size)
    config = TrainingConfig()
    optimizer = create_optimizer(model, config)
    
    # Register test hook manually
    trainer = Trainer(model, optimizer, config)
    test_hook = TestUpdateHook()
    trainer.hook_registry.register_update_hook(test_hook, enabled=True)
    
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    trainer.training_step(inputs)
    
    assert call_count["count"] == 1


# Run Logging Tests

def test_run_id_generated():
    """Test run_id is generated for each training run."""
    model = Transformer(vocab_size=100)
    config = TrainingConfig()
    optimizer = create_optimizer(model, config)
    
    trainer1 = Trainer(model, optimizer, config)
    trainer2 = Trainer(model, optimizer, config)
    
    assert trainer1.run_id != trainer2.run_id
    assert len(trainer1.run_id) > 0
    assert len(trainer2.run_id) > 0


def test_run_metadata_logged_once(capsys):
    """Test run metadata is logged once at start of training."""
    model = Transformer(vocab_size=100)
    config = TrainingConfig()
    optimizer = create_optimizer(model, config)
    trainer = Trainer(model, optimizer, config)
    
    inputs = torch.randint(0, 100, (2, 10))
    
    # First step should log metadata
    trainer.training_step(inputs)
    captured1 = capsys.readouterr()
    assert "run_id:" in captured1.out
    assert "git_commit:" in captured1.out
    assert "config_hash:" in captured1.out
    assert "hook_list:" in captured1.out
    
    # Second step should not log metadata again
    trainer.training_step(inputs)
    captured2 = capsys.readouterr()
    assert "run_id:" not in captured2.out


def test_config_hash_computed():
    """Test config_hash is computed and logged."""
    model = Transformer(vocab_size=100)
    config1 = TrainingConfig(learning_rate=1e-3)
    config2 = TrainingConfig(learning_rate=2e-3)
    optimizer1 = create_optimizer(model, config1)
    optimizer2 = create_optimizer(model, config2)
    
    trainer1 = Trainer(model, optimizer1, config1)
    trainer2 = Trainer(model, optimizer2, config2)
    
    hash1 = trainer1._compute_config_hash()
    hash2 = trainer2._compute_config_hash()
    
    assert hash1 != hash2  # Different configs should have different hashes
    assert len(hash1) > 0
    assert len(hash2) > 0


def test_hook_list_logged(capsys):
    """Test hook_list is logged correctly."""
    model = Transformer(vocab_size=100)
    config = TrainingConfig()
    config.hooks = {
        "forward": [
            {"name": "activation_stats", "enabled": True},
            {"name": "activation_stats", "enabled": False}
        ],
        "update": [
            {"name": "identity", "enabled": True}
        ]
    }
    optimizer = create_optimizer(model, config)
    trainer = Trainer(model, optimizer, config)
    
    inputs = torch.randint(0, 100, (2, 10))
    trainer.training_step(inputs)
    
    captured = capsys.readouterr()
    assert "hook_list:" in captured.out
    assert "activation_stats" in captured.out
    assert "identity" in captured.out


# Hook Safety Tests

def test_hooks_dont_break_training_step():
    """Test training step completes with hooks enabled."""
    vocab_size = 100
    batch_size = 2
    seq_len = 10
    
    model = Transformer(vocab_size=vocab_size)
    config = TrainingConfig()
    config.hooks = {
        "forward": [{"name": "activation_stats", "enabled": True}],
        "update": [{"name": "identity", "enabled": True}]
    }
    optimizer = create_optimizer(model, config)
    trainer = Trainer(model, optimizer, config)
    
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    loss = trainer.training_step(inputs)
    
    assert isinstance(loss, float)
    assert trainer.step == 1


def test_hooks_dont_break_training_step_disabled():
    """Test training step completes with hooks disabled."""
    vocab_size = 100
    batch_size = 2
    seq_len = 10
    
    model = Transformer(vocab_size=vocab_size)
    config = TrainingConfig()
    config.hooks = {
        "forward": [{"name": "activation_stats", "enabled": False}],
        "update": [{"name": "identity", "enabled": False}]
    }
    optimizer = create_optimizer(model, config)
    trainer = Trainer(model, optimizer, config)
    
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    loss = trainer.training_step(inputs)
    
    assert isinstance(loss, float)
    assert trainer.step == 1


def test_multiple_hooks_active_simultaneously():
    """Test multiple hooks can be active simultaneously."""
    vocab_size = 100
    batch_size = 2
    seq_len = 10
    
    forward_call_count = {"count": 0}
    update_call_count = {"count": 0}
    
    class TestForwardHook(ForwardHook):
        @property
        def name(self):
            return "test_forward"
        
        def __call__(self, activations):
            forward_call_count["count"] += 1
    
    class TestUpdateHook(UpdateHook):
        @property
        def name(self):
            return "test_update"
        
        def __call__(self, gradients):
            update_call_count["count"] += 1
            return gradients
    
    model = Transformer(vocab_size=vocab_size)
    config = TrainingConfig()
    optimizer = create_optimizer(model, config)
    trainer = Trainer(model, optimizer, config)
    
    # Register multiple hooks
    trainer.hook_registry.register_forward_hook(TestForwardHook(), enabled=True)
    trainer.hook_registry.register_forward_hook(ActivationStatsHook(), enabled=True)
    trainer.hook_registry.register_update_hook(TestUpdateHook(), enabled=True)
    trainer.hook_registry.register_update_hook(IdentityUpdateHook(), enabled=True)
    
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    loss = trainer.training_step(inputs)
    
    assert forward_call_count["count"] == 1
    assert update_call_count["count"] == 1
    assert isinstance(loss, float)


def test_unknown_hook_name_raises_error():
    """Test unknown hook name raises ValueError."""
    config = {
        "hooks": {
            "forward": [{"name": "unknown_hook", "enabled": True}]
        }
    }
    
    with pytest.raises(ValueError, match="Unknown forward hook name"):
        HookRegistry(config)
    
    config = {
        "hooks": {
            "update": [{"name": "unknown_hook", "enabled": True}]
        }
    }
    
    with pytest.raises(ValueError, match="Unknown update hook name"):
        HookRegistry(config)

