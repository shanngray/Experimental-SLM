"""Tests for model import CLI functionality.

This module provides tests for the import-model subcommand in main.py,
including HuggingFace model download, conversion, and registry integration.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from src.model.registry import ModelRegistry


class TestImportModelCLI:
    """Test import-model CLI command."""
    
    @pytest.fixture
    def temp_models_dir(self):
        """Create a temporary models directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir(parents=True)
            yield models_dir
    
    @pytest.fixture
    def temp_registry(self, temp_models_dir):
        """Create a temporary registry for testing."""
        registry_path = temp_models_dir.parent / "models" / "registry.json"
        registry = ModelRegistry(registry_path=registry_path)
        yield registry
    
    @patch('main.snapshot_download')
    @patch('main.AutoConfig.from_pretrained')
    @patch('src.model.adapters.qwen.QwenAdapter')
    @patch('main.ModelRegistry')
    def test_import_model_success(self, mock_registry_class, mock_adapter_class, 
                                   mock_config_class, mock_snapshot_download, temp_models_dir):
        """Test successful model import."""
        from main import handle_import_model
        from argparse import Namespace
        
        # Setup mocks
        mock_registry = Mock(spec=ModelRegistry)
        mock_registry.models = {}  # Empty registry
        mock_registry_class.return_value = mock_registry
        
        # Mock config
        mock_config = Mock()
        mock_config.num_parameters = 1000000
        mock_config.model_type = "qwen"  # Required for architecture check
        mock_config_class.return_value = mock_config
        
        # Mock adapter
        mock_adapter = Mock()
        mock_adapter.get_num_parameters.return_value = 1000000
        mock_adapter_class.return_value = mock_adapter
        
        # Create args
        args = Namespace(model_id="Qwen/Qwen-0.5B", name=None)
        
        # Calculate expected model name (sanitize_model_name removes dots)
        # "Qwen/Qwen-0.5B" -> "qwen-qwen-05b"
        expected_model_name = "qwen-qwen-05b"
        expected_model_dir = temp_models_dir / expected_model_name
        expected_model_dir.mkdir(parents=True)
        
        # Mock snapshot_download to return the directory path
        mock_snapshot_download.return_value = str(expected_model_dir)
        
        # Create mock README with license
        readme_path = expected_model_dir / "README.md"
        readme_path.write_text("license: apache-2.0\n")
        
        # Mock input for license acknowledgment
        with patch('builtins.input', return_value='y'):
            # Mock shutil.disk_usage to return enough space
            with patch('main.shutil.disk_usage') as mock_disk_usage:
                mock_stat = Mock()
                mock_stat.free = 10 * (1024 ** 3)  # 10 GB
                mock_disk_usage.return_value = mock_stat
                
                # Mock __file__ in main module so Path(__file__).parent returns temp_models_dir.parent
                # This way project_root / "models" will be temp_models_dir
                import main
                original_file = main.__file__
                # Create a fake __file__ path that when parented gives us temp_models_dir.parent
                fake_file = str(temp_models_dir.parent / "main.py")
                with patch.object(main, '__file__', fake_file):
                    result = handle_import_model(args)
        
        # Verify result
        assert result == 0
        
        # Verify registry was called
        mock_registry.add_model.assert_called_once()
        call_args = mock_registry.add_model.call_args
        assert call_args.kwargs['model_name'] == "qwen-qwen-05b"  # sanitize_model_name removes dots
        assert call_args.kwargs['model_id'] == "Qwen/Qwen-0.5B"
        assert call_args.kwargs['architecture_type'] == "qwen"
        assert call_args.kwargs['source'] == "huggingface"
    
    @patch('main.check_architecture_support')
    def test_import_model_unsupported_architecture(self, mock_check_arch, temp_models_dir):
        """Test import fails for unsupported architecture."""
        from main import handle_import_model
        from argparse import Namespace
        
        # Mock unsupported architecture
        mock_check_arch.return_value = (False, None)
        
        args = Namespace(model_id="Unsupported/Model", name=None)
        
        with patch('main.Path') as mock_path:
            mock_path.return_value.parent = temp_models_dir.parent
            result = handle_import_model(args)
        
        assert result == 1
    
    @patch('main.snapshot_download')
    @patch('main.AutoConfig.from_pretrained')
    @patch('main.ModelRegistry')
    def test_import_model_duplicate_name(self, mock_registry_class, mock_config_class,
                                         mock_snapshot_download, temp_models_dir):
        """Test import fails when model name already exists."""
        from main import handle_import_model
        from argparse import Namespace
        
        # Setup registry with existing model
        # Note: sanitize_model_name("Qwen/Qwen-0.5B") = "qwen-qwen-05b" (dot removed)
        mock_registry = Mock(spec=ModelRegistry)
        mock_registry.models = {"qwen-qwen-05b": {}}  # Model already exists
        mock_registry_class.return_value = mock_registry
        
        # Mock config
        mock_config = Mock()
        mock_config.num_parameters = 1000000
        mock_config.model_type = "qwen"  # Required for architecture check
        mock_config_class.return_value = mock_config
        
        args = Namespace(model_id="Qwen/Qwen-0.5B", name=None)
        
        # Mock __file__ in main module
        import main
        fake_file = str(temp_models_dir.parent / "main.py")
        with patch.object(main, '__file__', fake_file):
            result = handle_import_model(args)
        
        assert result == 1
    
    @patch('main.snapshot_download')
    @patch('main.AutoConfig.from_pretrained')
    @patch('src.model.adapters.qwen.QwenAdapter')
    @patch('main.ModelRegistry')
    def test_import_model_with_custom_name(self, mock_registry_class, mock_adapter_class,
                                           mock_config_class, mock_snapshot_download, temp_models_dir):
        """Test import with custom model name."""
        from main import handle_import_model
        from argparse import Namespace
        
        # Setup mocks
        mock_registry = Mock(spec=ModelRegistry)
        mock_registry.models = {}
        mock_registry_class.return_value = mock_registry
        
        mock_config = Mock()
        mock_config.num_parameters = 1000000
        mock_config.model_type = "qwen"  # Required for architecture check
        mock_config_class.return_value = mock_config
        
        mock_adapter = Mock()
        mock_adapter.get_num_parameters.return_value = 1000000
        mock_adapter_class.return_value = mock_adapter
        
        model_dir = temp_models_dir / "my-custom-name"
        model_dir.mkdir(parents=True)
        mock_snapshot_download.return_value = str(model_dir)
        
        readme_path = model_dir / "README.md"
        readme_path.write_text("license: apache-2.0\n")
        
        args = Namespace(model_id="Qwen/Qwen-0.5B", name="my-custom-name")
        
        with patch('builtins.input', return_value='y'):
            with patch('main.shutil.disk_usage') as mock_disk_usage:
                mock_stat = Mock()
                mock_stat.free = 10 * (1024 ** 3)
                mock_disk_usage.return_value = mock_stat
                
                with patch('main.Path') as mock_path:
                    mock_path.return_value.parent = temp_models_dir.parent
                    result = handle_import_model(args)
        
        assert result == 0
        call_args = mock_registry.add_model.call_args
        assert call_args.kwargs['model_name'] == "my-custom-name"
    
    @patch('main.snapshot_download')
    @patch('main.AutoConfig.from_pretrained')
    @patch('main.ModelRegistry')
    def test_import_model_license_rejection(self, mock_registry_class, mock_config_class,
                                            mock_snapshot_download, temp_models_dir):
        """Test import cancelled when license is not acknowledged."""
        from main import handle_import_model
        from argparse import Namespace
        
        # Setup mocks
        mock_registry = Mock(spec=ModelRegistry)
        mock_registry.models = {}
        mock_registry_class.return_value = mock_registry
        
        mock_config = Mock()
        mock_config.num_parameters = 1000000
        mock_config.model_type = "qwen"  # Required for architecture check
        mock_config_class.return_value = mock_config
        
        # Calculate expected model name (sanitize_model_name removes dots)
        # "Qwen/Qwen-0.5B" -> "qwen-qwen-05b"
        expected_model_name = "qwen-qwen-05b"
        model_dir = temp_models_dir / expected_model_name
        model_dir.mkdir(parents=True)
        mock_snapshot_download.return_value = str(model_dir)
        
        # Create mock README with license and some other files to pass empty check
        readme_path = model_dir / "README.md"
        readme_path.write_text("license: apache-2.0\n")
        # Add a config file to ensure directory is not empty
        config_path = model_dir / "config.json"
        config_path.write_text('{"model_type": "qwen"}')
        
        args = Namespace(model_id="Qwen/Qwen-0.5B", name=None)
        
        with patch('builtins.input', return_value='n'):  # Reject license
            with patch('main.shutil.disk_usage') as mock_disk_usage:
                mock_stat = Mock()
                mock_stat.free = 10 * (1024 ** 3)
                mock_disk_usage.return_value = mock_stat
                
                # Mock __file__ in main module so Path(__file__).parent returns temp_models_dir.parent
                # This way project_root / "models" will be temp_models_dir
                import main
                original_file = main.__file__
                # Create a fake __file__ path that when parented gives us temp_models_dir.parent
                fake_file = str(temp_models_dir.parent / "main.py")
                with patch.object(main, '__file__', fake_file):
                    with patch('main.shutil.rmtree') as mock_rmtree:
                        result = handle_import_model(args)
        
        assert result == 1
        # Verify cleanup was called
        mock_rmtree.assert_called_once()
    
    @patch('main.snapshot_download')
    @patch('main.AutoConfig.from_pretrained')
    @patch('src.model.adapters.qwen.QwenAdapter')
    @patch('main.ModelRegistry')
    def test_import_model_download_failure(self, mock_registry_class, mock_adapter_class,
                                           mock_config_class, mock_snapshot_download, temp_models_dir):
        """Test import fails when download fails."""
        from main import handle_import_model
        from argparse import Namespace
        
        # Setup mocks
        mock_registry = Mock(spec=ModelRegistry)
        mock_registry.models = {}
        mock_registry_class.return_value = mock_registry
        
        mock_config = Mock()
        mock_config.num_parameters = 1000000
        mock_config.model_type = "qwen"  # Required for architecture check
        mock_config_class.return_value = mock_config
        
        # Mock download failure
        mock_snapshot_download.side_effect = Exception("Download failed")
        
        args = Namespace(model_id="Qwen/Qwen-0.5B", name=None)
        
        # Create a partial download directory to simulate cleanup scenario
        expected_model_name = "qwen-qwen-05b"
        partial_dir = temp_models_dir / expected_model_name
        partial_dir.mkdir(parents=True)
        
        with patch('main.shutil.disk_usage') as mock_disk_usage:
            mock_stat = Mock()
            mock_stat.free = 10 * (1024 ** 3)
            mock_disk_usage.return_value = mock_stat
            
            # Mock __file__ in main module
            import main
            fake_file = str(temp_models_dir.parent / "main.py")
            with patch.object(main, '__file__', fake_file):
                with patch('main.shutil.rmtree') as mock_rmtree:
                    result = handle_import_model(args)
        
        assert result == 1
        # Verify cleanup was called (directory exists, so cleanup should happen)
        mock_rmtree.assert_called_once()
    
    @patch('main.snapshot_download')
    @patch('main.AutoConfig.from_pretrained')
    @patch('src.model.adapters.qwen.QwenAdapter')
    @patch('main.ModelRegistry')
    def test_import_model_validation_failure(self, mock_registry_class, mock_adapter_class,
                                             mock_config_class, mock_snapshot_download, temp_models_dir):
        """Test import fails when model validation fails."""
        from main import handle_import_model
        from argparse import Namespace
        
        # Setup mocks
        mock_registry = Mock(spec=ModelRegistry)
        mock_registry.models = {}
        mock_registry_class.return_value = mock_registry
        
        mock_config = Mock()
        mock_config.num_parameters = 1000000
        mock_config.model_type = "qwen"  # Required for architecture check
        mock_config_class.return_value = mock_config
        
        # Mock adapter validation failure
        mock_adapter_class.side_effect = Exception("Validation failed")
        
        # Calculate expected model name (sanitize_model_name removes dots)
        # "Qwen/Qwen-0.5B" -> "qwen-qwen-05b"
        expected_model_name = "qwen-qwen-05b"
        model_dir = temp_models_dir / expected_model_name
        model_dir.mkdir(parents=True)
        mock_snapshot_download.return_value = str(model_dir)
        
        # Create mock README with license and some other files to pass empty check
        readme_path = model_dir / "README.md"
        readme_path.write_text("license: apache-2.0\n")
        # Add a config file to ensure directory is not empty
        config_path = model_dir / "config.json"
        config_path.write_text('{"model_type": "qwen"}')
        
        args = Namespace(model_id="Qwen/Qwen-0.5B", name=None)
        
        with patch('builtins.input', return_value='y'):
            with patch('main.shutil.disk_usage') as mock_disk_usage:
                mock_stat = Mock()
                mock_stat.free = 10 * (1024 ** 3)
                mock_disk_usage.return_value = mock_stat
                
                # Mock __file__ in main module so Path(__file__).parent returns temp_models_dir.parent
                # This way project_root / "models" will be temp_models_dir
                import main
                original_file = main.__file__
                # Create a fake __file__ path that when parented gives us temp_models_dir.parent
                fake_file = str(temp_models_dir.parent / "main.py")
                with patch.object(main, '__file__', fake_file):
                    with patch('main.shutil.rmtree') as mock_rmtree:
                        result = handle_import_model(args)
        
        assert result == 1
        # Verify cleanup was called
        mock_rmtree.assert_called_once()

