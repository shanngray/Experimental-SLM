# Multi-Model Support Tutorial

This tutorial walks through importing a pretrained model from HuggingFace, fine-tuning it, and managing multiple models.

## Prerequisites

- Project dependencies installed (`uv sync`)
- HuggingFace account (for gated models, optional for public models)
- Basic understanding of YAML configuration files

## Step 1: Import a Model

Import a Qwen model from HuggingFace:

```bash
uv run python main.py import-model Qwen/Qwen-0.5B
```

This will:
1. Download the model from HuggingFace Hub
2. Convert it to local format
3. Save to `models/qwen-0.5b-base/`
4. Register in `models/registry.json`
5. Display the `model_name` for use in configs

**Output:**
```
Model imported successfully!
Model name: qwen-0.5b-base
Use this name in your config file: model_name: "qwen-0.5b-base"
```

### Import with Custom Name

You can specify a custom name:

```bash
uv run python main.py import-model Qwen/Qwen-0.5B --name my-qwen-model
```

## Step 2: List Available Models

View all imported models:

```bash
uv run python main.py list-models
```

**Output:**
```
Available Models:
  qwen-0.5b-base
    Model ID: Qwen/Qwen-0.5B
    Architecture: qwen
    Source: huggingface
    Created: 2024-01-15T10:30:00Z
```

## Step 3: View Model Details

Get detailed information about a model:

```bash
uv run python main.py model-info qwen-0.5b-base
```

This shows:
- Model ID and architecture
- Source and creation date
- Fine-tuning lineage (if applicable)
- Model size and metadata

## Step 4: Create Fine-Tuning Config

Create a config file for fine-tuning:

```yaml
# configs/qwen-finetuned.yaml
model_name: "qwen-0.5b-base"  # Use imported model
learning_rate: 5e-5  # Lower LR for fine-tuning
batch_size: 16
max_steps: 5000
eval_cadence: 500
sampling_cadence: 500
```

Key differences from training from scratch:
- `model_name` specifies the imported model
- Lower `learning_rate` (typically 1e-5 to 1e-4 for fine-tuning)
- Architecture parameters come from the model, not config

## Step 5: Fine-Tune the Model

Start fine-tuning:

```bash
uv run python main.py --config configs/qwen-finetuned.yaml
```

The system will:
1. Load the Qwen model from registry
2. Use Qwen's native tokenizer
3. Train on your dataset
4. Save checkpoints with fine-tuning metadata

## Step 6: Resume Fine-Tuning

If training is interrupted, resume from a checkpoint:

```bash
uv run python main.py --resume checkpoints/checkpoint_step_1000 --config configs/qwen-finetuned.yaml
```

The checkpoint metadata ensures:
- Correct model architecture is loaded
- Fine-tuning lineage is preserved
- Training continues from the correct step

## Step 7: Check Fine-Tuning Lineage

Checkpoints track fine-tuning lineage. View checkpoint metadata:

```bash
cat checkpoints/checkpoint_step_1000/metadata.json | grep -A 5 model
```

**Output:**
```json
{
  "model_name": "qwen-0.5b-base",
  "model_id": "Qwen/Qwen-0.5B",
  "model_source": "huggingface",
  "fine_tuned_from": null
}
```

If you fine-tune from a checkpoint, subsequent checkpoints will have:
```json
{
  "fine_tuned_from": "qwen-0.5b-base"
}
```

## Step 8: Compare Models

You can train multiple variants and compare them:

1. **Base model**: Import and use directly
2. **Fine-tuned variant 1**: Fine-tune with learning rate 5e-5
3. **Fine-tuned variant 2**: Fine-tune with learning rate 1e-4

Each variant maintains its lineage in checkpoints, making it easy to track which configuration produced which results.

## Step 9: Validate Models

Before using a model, validate its integrity:

```bash
uv run python main.py validate-model qwen-0.5b-base
```

This checks:
- Model files exist and are readable
- Registry entry is valid
- Model can be loaded successfully

## Step 10: Delete Models

Remove a model from the registry:

```bash
# Remove from registry only
uv run python main.py delete-model qwen-0.5b-base

# Remove from registry and delete files
uv run python main.py delete-model qwen-0.5b-base --delete-files
```

## Tips and Best Practices

### Learning Rates

- **From scratch**: 1e-4 to 3e-4
- **Fine-tuning**: 1e-5 to 1e-4 (typically 10x lower)

### Checkpoint Management

- Checkpoints include model metadata automatically
- Resume always uses the correct model architecture
- Fine-tuning lineage is preserved across checkpoints

### Model Selection

- Use `model_name` in config to select models
- Set `model_name: null` for custom Transformer (backward compatible)
- Registry validates model existence before training

### Tokenizer Considerations

- Imported models use their native tokenizers
- Custom Transformer uses character-level tokenizer
- Tokenizer is automatically selected based on `model_name`

## Troubleshooting

### Model Not Found

**Error**: `Model 'qwen-0.5b-base' not found in registry`

**Solution**: Import the model first:
```bash
uv run python main.py import-model Qwen/Qwen-0.5B
```

### Checkpoint Model Mismatch

**Warning**: `Checkpoint was saved with model_name='qwen-0.5b-base', but config specifies model_name='custom'`

**Solution**: The system automatically uses the checkpoint's model. Update your config to match, or use the checkpoint's model.

### Import Fails

**Error**: Authentication required for gated model

**Solution**: Set HuggingFace token:
```bash
export HUGGINGFACE_HUB_TOKEN=your_token_here
```

Or login interactively:
```bash
huggingface-cli login
```

## Next Steps

- Experiment with different Qwen model sizes (0.5B, 1.8B)
- Fine-tune on domain-specific data
- Compare fine-tuned vs. base model performance
- Explore adding support for other architectures

