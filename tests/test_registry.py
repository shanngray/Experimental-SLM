"""Tests for model registry system."""

import json
import tempfile
from pathlib import Path
from datetime import datetime

import pytest

from src.model.registry import ModelRegistry


@pytest.fixture
def temp_registry():
    """Create a temporary registry for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = Path(tmpdir) / "registry.json"
        registry = ModelRegistry(registry_path=registry_path)
        yield registry


class TestRegistryInitialization:
    """Test registry initialization."""
    
    def test_registry_creates_file_if_not_exists(self, temp_registry):
        """Test that registry creates file if it doesn't exist."""
        assert temp_registry.registry_path.exists()
        
        # Check file contents
        with open(temp_registry.registry_path, 'r') as f:
            data = json.load(f)
        
        assert data['schema_version'] == "1.0"
        assert data['models'] == {}
    
    def test_registry_loads_existing_file(self, temp_registry):
        """Test that registry loads existing file."""
        # Add a model
        temp_registry.add_model(
            model_name="test-model",
            model_id="test/id",
            architecture_type="custom-transformer",
            local_path="models/test-model"
        )
        
        # Create new registry instance pointing to same file
        registry2 = ModelRegistry(registry_path=temp_registry.registry_path)
        
        # Should have loaded the model
        assert registry2.get_model("test-model") is not None
        assert len(registry2.list_models()) == 1
    
    def test_registry_validates_schema_version(self, temp_registry):
        """Test that registry validates schema version."""
        # Manually write invalid schema version
        with open(temp_registry.registry_path, 'w') as f:
            json.dump({
                'schema_version': "2.0",
                'models': {}
            }, f)
        
        # Should raise error (wrapped in RuntimeError)
        with pytest.raises(RuntimeError, match="schema version mismatch"):
            ModelRegistry(registry_path=temp_registry.registry_path)


class TestAddModel:
    """Test adding models to registry."""
    
    def test_add_model_basic(self, temp_registry):
        """Test adding a basic model."""
        temp_registry.add_model(
            model_name="test-model",
            model_id="test/id",
            architecture_type="custom-transformer",
            local_path="models/test-model"
        )
        
        entry = temp_registry.get_model("test-model")
        assert entry is not None
        assert entry['model_name'] == "test-model"
        assert entry['model_id'] == "test/id"
        assert entry['architecture_type'] == "custom-transformer"
        assert entry['local_path'] == "models/test-model"
        assert entry['source'] == "custom"
        assert 'created_at' in entry
    
    def test_add_model_with_metadata(self, temp_registry):
        """Test adding model with custom metadata."""
        metadata = {"license": "MIT", "author": "test"}
        temp_registry.add_model(
            model_name="test-model",
            model_id="test/id",
            architecture_type="custom-transformer",
            local_path="models/test-model",
            metadata=metadata
        )
        
        entry = temp_registry.get_model("test-model")
        assert entry['metadata'] == metadata
    
    def test_add_model_with_fine_tuned_from(self, temp_registry):
        """Test adding fine-tuned model."""
        # Add base model first
        temp_registry.add_model(
            model_name="base-model",
            model_id="base/id",
            architecture_type="custom-transformer",
            local_path="models/base-model"
        )
        
        # Add fine-tuned model
        temp_registry.add_model(
            model_name="finetuned-model",
            model_id="finetuned/id",
            architecture_type="custom-transformer",
            local_path="models/finetuned-model",
            fine_tuned_from="base-model"
        )
        
        entry = temp_registry.get_model("finetuned-model")
        assert entry['fine_tuned_from'] == "base-model"
    
    def test_add_model_duplicate_name(self, temp_registry):
        """Test that adding duplicate model_name raises error."""
        temp_registry.add_model(
            model_name="test-model",
            model_id="test/id",
            architecture_type="custom-transformer",
            local_path="models/test-model"
        )
        
        with pytest.raises(ValueError, match="already exists"):
            temp_registry.add_model(
                model_name="test-model",
                model_id="test/id2",
                architecture_type="custom-transformer",
                local_path="models/test-model2"
            )
    
    def test_add_model_invalid_parent(self, temp_registry):
        """Test that fine_tuned_from must exist."""
        with pytest.raises(ValueError, match="not found in registry"):
            temp_registry.add_model(
                model_name="finetuned-model",
                model_id="finetuned/id",
                architecture_type="custom-transformer",
                local_path="models/finetuned-model",
                fine_tuned_from="nonexistent-parent"
            )


class TestGetModel:
    """Test getting models from registry."""
    
    def test_get_model_exists(self, temp_registry):
        """Test getting existing model."""
        temp_registry.add_model(
            model_name="test-model",
            model_id="test/id",
            architecture_type="custom-transformer",
            local_path="models/test-model"
        )
        
        entry = temp_registry.get_model("test-model")
        assert entry is not None
        assert entry['model_name'] == "test-model"
    
    def test_get_model_not_exists(self, temp_registry):
        """Test getting non-existent model."""
        entry = temp_registry.get_model("nonexistent")
        assert entry is None


class TestListModels:
    """Test listing models."""
    
    def test_list_models_empty(self, temp_registry):
        """Test listing empty registry."""
        models = temp_registry.list_models()
        assert len(models) == 0
    
    def test_list_models_multiple(self, temp_registry):
        """Test listing multiple models."""
        # Add multiple models
        temp_registry.add_model(
            model_name="model1",
            model_id="id1",
            architecture_type="custom-transformer",
            local_path="models/model1"
        )
        temp_registry.add_model(
            model_name="model2",
            model_id="id2",
            architecture_type="qwen",
            local_path="models/model2"
        )
        
        models = temp_registry.list_models()
        assert len(models) == 2
        
        # Check ordering (most recent first)
        assert models[0]['model_name'] == "model2"
        assert models[1]['model_name'] == "model1"
    
    def test_list_models_filter_architecture(self, temp_registry):
        """Test filtering by architecture type."""
        temp_registry.add_model(
            model_name="model1",
            model_id="id1",
            architecture_type="custom-transformer",
            local_path="models/model1"
        )
        temp_registry.add_model(
            model_name="model2",
            model_id="id2",
            architecture_type="qwen",
            local_path="models/model2"
        )
        
        qwen_models = temp_registry.list_models(architecture_type="qwen")
        assert len(qwen_models) == 1
        assert qwen_models[0]['model_name'] == "model2"
    
    def test_list_models_filter_source(self, temp_registry):
        """Test filtering by source."""
        temp_registry.add_model(
            model_name="model1",
            model_id="id1",
            architecture_type="custom-transformer",
            local_path="models/model1",
            source="huggingface"
        )
        temp_registry.add_model(
            model_name="model2",
            model_id="id2",
            architecture_type="custom-transformer",
            local_path="models/model2",
            source="custom"
        )
        
        huggingface_models = temp_registry.list_models(source="huggingface")
        assert len(huggingface_models) == 1
        assert huggingface_models[0]['model_name'] == "model1"


class TestDeleteModel:
    """Test deleting models."""
    
    def test_delete_model(self, temp_registry):
        """Test deleting a model."""
        temp_registry.add_model(
            model_name="test-model",
            model_id="test/id",
            architecture_type="custom-transformer",
            local_path="models/test-model"
        )
        
        temp_registry.delete_model("test-model")
        
        assert temp_registry.get_model("test-model") is None
        assert len(temp_registry.list_models()) == 0
    
    def test_delete_model_not_exists(self, temp_registry):
        """Test deleting non-existent model raises error."""
        with pytest.raises(ValueError, match="not found"):
            temp_registry.delete_model("nonexistent")
    
    def test_delete_model_with_files(self, temp_registry):
        """Test deleting model with files."""
        import tempfile
        import shutil
        
        # Create temporary model directory
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "test-model"
            model_dir.mkdir()
            (model_dir / "weights.pt").touch()
            
            temp_registry.add_model(
                model_name="test-model",
                model_id="test/id",
                architecture_type="custom-transformer",
                local_path=str(model_dir)
            )
            
            # Delete with files
            temp_registry.delete_model("test-model", delete_files=True)
            
            # Directory should be gone
            assert not model_dir.exists()


class TestFineTuningLineage:
    """Test fine-tuning lineage tracking."""
    
    def test_get_fine_tuning_children(self, temp_registry):
        """Test getting fine-tuning children."""
        # Add base model
        temp_registry.add_model(
            model_name="base",
            model_id="base/id",
            architecture_type="custom-transformer",
            local_path="models/base"
        )
        
        # Add fine-tuned models
        temp_registry.add_model(
            model_name="finetuned1",
            model_id="ft1/id",
            architecture_type="custom-transformer",
            local_path="models/ft1",
            fine_tuned_from="base"
        )
        temp_registry.add_model(
            model_name="finetuned2",
            model_id="ft2/id",
            architecture_type="custom-transformer",
            local_path="models/ft2",
            fine_tuned_from="base"
        )
        
        children = temp_registry.get_fine_tuning_children("base")
        assert len(children) == 2
        assert {c['model_name'] for c in children} == {"finetuned1", "finetuned2"}
    
    def test_get_fine_tuning_lineage(self, temp_registry):
        """Test getting full lineage chain."""
        # Create chain: base -> ft1 -> ft2
        temp_registry.add_model(
            model_name="base",
            model_id="base/id",
            architecture_type="custom-transformer",
            local_path="models/base"
        )
        temp_registry.add_model(
            model_name="ft1",
            model_id="ft1/id",
            architecture_type="custom-transformer",
            local_path="models/ft1",
            fine_tuned_from="base"
        )
        temp_registry.add_model(
            model_name="ft2",
            model_id="ft2/id",
            architecture_type="custom-transformer",
            local_path="models/ft2",
            fine_tuned_from="ft1"
        )
        
        # Get lineage for ft2
        lineage = temp_registry.get_fine_tuning_lineage("ft2")
        assert len(lineage) == 3
        assert lineage[0]['model_name'] == "base"
        assert lineage[1]['model_name'] == "ft1"
        assert lineage[2]['model_name'] == "ft2"


class TestQueryMethods:
    """Test query methods."""
    
    def test_query_by_architecture(self, temp_registry):
        """Test querying by architecture."""
        temp_registry.add_model(
            model_name="model1",
            model_id="id1",
            architecture_type="custom-transformer",
            local_path="models/model1"
        )
        temp_registry.add_model(
            model_name="model2",
            model_id="id2",
            architecture_type="qwen",
            local_path="models/model2"
        )
        
        qwen_models = temp_registry.query_by_architecture("qwen")
        assert len(qwen_models) == 1
        assert qwen_models[0]['architecture_type'] == "qwen"
    
    def test_query_by_source(self, temp_registry):
        """Test querying by source."""
        temp_registry.add_model(
            model_name="model1",
            model_id="id1",
            architecture_type="custom-transformer",
            local_path="models/model1",
            source="huggingface"
        )
        temp_registry.add_model(
            model_name="model2",
            model_id="id2",
            architecture_type="custom-transformer",
            local_path="models/model2",
            source="custom"
        )
        
        hf_models = temp_registry.query_by_source("huggingface")
        assert len(hf_models) == 1
        assert hf_models[0]['source'] == "huggingface"


class TestGetModelInfo:
    """Test getting detailed model info."""
    
    def test_get_model_info(self, temp_registry):
        """Test getting model info with file size."""
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "test-model"
            model_dir.mkdir()
            
            # Create a file with known size
            test_file = model_dir / "weights.pt"
            test_file.write_bytes(b"x" * 1024)  # 1KB
            
            temp_registry.add_model(
                model_name="test-model",
                model_id="test/id",
                architecture_type="custom-transformer",
                local_path=str(model_dir)
            )
            
            info = temp_registry.get_model_info("test-model")
            assert info is not None
            assert 'total_size_bytes' in info
            assert 'total_size_mb' in info
            assert info['total_size_bytes'] >= 1024

