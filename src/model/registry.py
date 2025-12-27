"""Model registry for tracking available models and their metadata.

This module provides a registry system to manage models, their metadata,
and enable model discovery and selection. The registry persists to a JSON file
and tracks both user-friendly model names and original identifiers.
"""

import json
import os
import warnings
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Optional, Any
import tempfile


class ModelRegistry:
    """Registry for managing model metadata and discovery.
    
    The registry tracks models with dual identifiers:
    - model_name: User-friendly identifier for configs (e.g., "qwen-0.5b-base")
    - model_id: Original identifier from source (e.g., "Qwen/Qwen-0.5B")
    
    Registry is persisted to JSON file at `models/registry.json`.
    
    Attributes:
        registry_path: Path to registry JSON file.
        models: Dictionary mapping model_name to model entry.
        schema_version: Schema version for future compatibility.
    """
    
    SCHEMA_VERSION = "1.0"
    
    def __init__(self, registry_path: Optional[Path] = None):
        """Initialize model registry.
        
        Args:
            registry_path: Path to registry JSON file. Defaults to models/registry.json
                relative to project root.
        """
        if registry_path is None:
            # Default to models/registry.json relative to project root
            project_root = Path(__file__).parent.parent.parent
            registry_path = project_root / "models" / "registry.json"
        
        self.registry_path = Path(registry_path)
        self.models: Dict[str, Dict[str, Any]] = {}
        self.schema_version = self.SCHEMA_VERSION
        
        # Ensure models directory exists
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing registry if it exists
        if self.registry_path.exists():
            self._load()
        else:
            # Initialize empty registry
            self._save()
    
    def _load(self) -> None:
        """Load registry from JSON file."""
        try:
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate schema version
            if data.get('schema_version') != self.SCHEMA_VERSION:
                raise ValueError(
                    f"Registry schema version mismatch. Expected {self.SCHEMA_VERSION}, "
                    f"got {data.get('schema_version')}"
                )
            
            self.models = data.get('models', {})
            self.schema_version = data.get('schema_version', self.SCHEMA_VERSION)
            
            # Normalize entries: ensure all models have fine_tuned_from key for consistency
            for entry in self.models.values():
                if 'fine_tuned_from' not in entry:
                    entry['fine_tuned_from'] = None
            
            # Validate registry integrity
            self._validate_integrity()
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in registry file: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to load registry: {e}") from e
    
    def _save(self) -> None:
        """Save registry to JSON file atomically."""
        data = {
            'schema_version': self.schema_version,
            'models': self.models
        }
        
        # Atomic write: write to temp file, then rename
        temp_path = self.registry_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            temp_path.replace(self.registry_path)
        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"Failed to save registry: {e}") from e
    
    def _validate_integrity(self) -> None:
        """Validate registry integrity and fix issues where possible."""
        # Check for duplicate model_names (shouldn't happen, but check anyway)
        model_names = list(self.models.keys())
        if len(model_names) != len(set(model_names)):
            raise ValueError("Duplicate model_names found in registry")
        
        # Check for missing local_path directories
        for model_name, entry in self.models.items():
            local_path = entry.get('local_path')
            if local_path:
                path = Path(local_path)
                if not path.is_absolute():
                    # Relative path - resolve relative to project root
                    project_root = Path(__file__).parent.parent.parent
                    path = project_root / path
                
                if not path.exists():
                    # Warn but don't fail - model might be on different machine
                    warnings.warn(
                        f"Model '{model_name}' has missing local_path: {local_path}",
                        UserWarning
                    )
    
    def add_model(
        self,
        model_name: str,
        model_id: str,
        architecture_type: str,
        local_path: str,
        source: str = "custom",
        metadata: Optional[Dict[str, Any]] = None,
        fine_tuned_from: Optional[str] = None
    ) -> None:
        """Add a model to the registry.
        
        Args:
            model_name: Unique user-friendly identifier for the model.
            model_id: Original identifier (e.g., HuggingFace repo ID).
            architecture_type: Model architecture family (e.g., "qwen", "custom-transformer").
            local_path: Relative path to model directory.
            source: Source of model ("huggingface", "custom", "finetuned").
            metadata: Additional model-specific metadata.
            fine_tuned_from: model_name of parent model if this is a fine-tuned model.
        
        Raises:
            ValueError: If model_name already exists or validation fails.
        """
        if model_name in self.models:
            raise ValueError(f"Model '{model_name}' already exists in registry")
        
        # Validate fine_tuned_from exists if specified
        if fine_tuned_from is not None and fine_tuned_from not in self.models:
            raise ValueError(
                f"Parent model '{fine_tuned_from}' not found in registry"
            )
        
        entry = {
            'model_name': model_name,
            'model_id': model_id,
            'architecture_type': architecture_type,
            'local_path': local_path,
            'source': source,
            'created_at': datetime.now(UTC).isoformat(),
            'metadata': metadata or {},
            'fine_tuned_from': fine_tuned_from,
        }
        
        self.models[model_name] = entry
        self._save()
    
    def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get model entry by model_name.
        
        Args:
            model_name: User-friendly model identifier.
        
        Returns:
            Model entry dictionary if found, None otherwise.
        """
        entry = self.models.get(model_name)
        if entry is None:
            return None
        
        # Validate local_path exists
        local_path = entry.get('local_path')
        if local_path:
            path = Path(local_path)
            if not path.is_absolute():
                project_root = Path(__file__).parent.parent.parent
                path = project_root / path
            
            if not path.exists():
                warnings.warn(
                    f"Model '{model_name}' local_path does not exist: {local_path}",
                    UserWarning
                )
        
        return entry.copy()
    
    def list_models(
        self,
        architecture_type: Optional[str] = None,
        source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all models in registry, optionally filtered.
        
        Args:
            architecture_type: Filter by architecture type (e.g., "qwen").
            source: Filter by source (e.g., "huggingface").
        
        Returns:
            List of model entries with summary information, ordered by created_at
            (most recent first).
        """
        results = []
        
        for model_name, entry in self.models.items():
            # Apply filters
            if architecture_type is not None:
                if entry.get('architecture_type') != architecture_type:
                    continue
            
            if source is not None:
                if entry.get('source') != source:
                    continue
            
            # Create summary entry
            summary = {
                'model_name': entry['model_name'],
                'model_id': entry['model_id'],
                'architecture_type': entry['architecture_type'],
                'source': entry['source'],
                'created_at': entry['created_at'],
            }
            
            if 'fine_tuned_from' in entry:
                summary['fine_tuned_from'] = entry['fine_tuned_from']
            
            results.append(summary)
        
        # Sort by created_at (most recent first)
        results.sort(key=lambda x: x['created_at'], reverse=True)
        
        return results
    
    def delete_model(self, model_name: str, delete_files: bool = False) -> None:
        """Delete a model from the registry.
        
        Args:
            model_name: User-friendly model identifier.
            delete_files: If True, also delete model files from disk.
        
        Raises:
            ValueError: If model_name not found.
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        entry = self.models[model_name]
        
        # Check if this model is a parent of fine-tuned models
        children = [
            name for name, e in self.models.items()
            if e.get('fine_tuned_from') == model_name
        ]
        if children:
            warnings.warn(
                f"Model '{model_name}' is parent of fine-tuned models: {children}",
                UserWarning
            )
        
        # Delete files if requested
        if delete_files:
            local_path = entry.get('local_path')
            if local_path:
                path = Path(local_path)
                if not path.is_absolute():
                    project_root = Path(__file__).parent.parent.parent
                    path = project_root / path
                
                if path.exists():
                    import shutil
                    shutil.rmtree(path)
        
        # Remove from registry
        del self.models[model_name]
        self._save()
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a model.
        
        Args:
            model_name: User-friendly model identifier.
        
        Returns:
            Complete model metadata including registry fields and file size info,
            or None if not found.
        """
        entry = self.get_model(model_name)
        if entry is None:
            return None
        
        # Add file size information
        local_path = entry.get('local_path')
        if local_path:
            path = Path(local_path)
            if not path.is_absolute():
                project_root = Path(__file__).parent.parent.parent
                path = project_root / path
            
            if path.exists():
                total_size = sum(
                    f.stat().st_size for f in path.rglob('*') if f.is_file()
                )
                entry['total_size_bytes'] = total_size
                entry['total_size_mb'] = total_size / (1024 * 1024)
        
        return entry
    
    def query_by_architecture(self, architecture_type: str) -> List[Dict[str, Any]]:
        """Query models by architecture type.
        
        Args:
            architecture_type: Architecture type to filter by.
        
        Returns:
            List of models matching architecture type, ordered by created_at.
        """
        return self.list_models(architecture_type=architecture_type)
    
    def query_by_source(self, source: str) -> List[Dict[str, Any]]:
        """Query models by source.
        
        Args:
            source: Source to filter by.
        
        Returns:
            List of models matching source, ordered by created_at.
        """
        return self.list_models(source=source)
    
    def get_fine_tuning_children(self, model_name: str) -> List[Dict[str, Any]]:
        """Get all models fine-tuned from a specific base model.
        
        Args:
            model_name: Base model name.
        
        Returns:
            List of fine-tuned models with fine-tuning timestamps.
        """
        children = []
        for name, entry in self.models.items():
            if entry.get('fine_tuned_from') == model_name:
                children.append({
                    'model_name': entry['model_name'],
                    'model_id': entry['model_id'],
                    'created_at': entry['created_at'],
                })
        
        # Sort by created_at
        children.sort(key=lambda x: x['created_at'], reverse=True)
        return children
    
    def get_fine_tuning_lineage(self, model_name: str) -> List[Dict[str, Any]]:
        """Get full fine-tuning lineage chain for a model.
        
        Args:
            model_name: Model name to get lineage for.
        
        Returns:
            List of models in lineage chain from base to current, with timestamps.
            Returns empty list if model not found or has no lineage.
        """
        if model_name not in self.models:
            return []
        
        chain = []
        current_name = model_name
        
        # Traverse up the chain
        while current_name in self.models:
            entry = self.models[current_name]
            chain.append({
                'model_name': entry['model_name'],
                'model_id': entry['model_id'],
                'created_at': entry['created_at'],
            })
            
            # Check if this model was fine-tuned from another
            parent = entry.get('fine_tuned_from')
            if parent is None:
                break
            
            current_name = parent
        
        # Reverse to get base -> current order
        chain.reverse()
        return chain

