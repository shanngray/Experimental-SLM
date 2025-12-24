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
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml

from src.config import TrainingConfig
from src.dataloader import DataLoader
from src.dataset import WindowDataset, split_corpus
from src.model.transformer import Transformer
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


def parse_args():
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train a transformer language model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from scratch
  python main.py

  # Resume from checkpoint
  python main.py --resume checkpoints/checkpoint_step_1000

  # Use custom config and data
  python main.py --config configs/experiment.yaml --data data/custom.txt

  # Override max steps
  python main.py --max-steps 5000
        """
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        metavar='PATH',
        help='Resume training from checkpoint directory'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        metavar='PATH',
        help='Load configuration from YAML file'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        metavar='PATH',
        help='Path to training data file (default: data/uni-alg-int.txt)'
    )
    
    parser.add_argument(
        '--max-steps',
        type=int,
        metavar='N',
        help='Maximum training steps (overrides config)'
    )
    
    return parser.parse_args()


def main():
    """Main training entry point."""
    # Parse command-line arguments
    args = parse_args()
    
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
    
    # Initialize tokenizer and tokenize corpus
    print("Initializing tokenizer...")
    tokenizer = Tokenizer()
    corpus = tokenizer.encode(text_data)
    vocab_size = len(tokenizer.char_to_id)
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
            # Create model and optimizer first (will be loaded from checkpoint)
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
            optimizer = create_optimizer(model, config)
            
            # Load checkpoint and create trainer
            trainer = Trainer.from_checkpoint(args.resume, model, optimizer, tokenizer)
            # Update val_dataloader and tokenizer since from_checkpoint doesn't restore these
            trainer.val_dataloader = val_dataloader
            trainer.tokenizer = tokenizer
            print(f"Resumed from step {trainer.step}")
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Error loading checkpoint: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Create model and optimizer from scratch
        print("Creating model...")
        print(f"  Architecture: n_layers={config.n_layers}, d_model={config.d_model}, "
              f"n_heads={config.n_heads}, d_ff={config.d_ff}, dropout={config.dropout}")
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
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model created: {num_params:,} parameters")
        
        print("Creating optimizer...")
        optimizer = create_optimizer(model, config)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            config=config,
            val_dataloader=val_dataloader,
            tokenizer=tokenizer
        )
        print("Trainer initialized")
    
    # Training loop
    print(f"\nStarting training for {max_steps} steps...")
    if config.checkpoint_cadence is not None:
        print(f"Checkpoints will be saved every {config.checkpoint_cadence} steps")
    else:
        print("Periodic checkpointing is disabled (checkpoint_cadence is None)")
    print("Press Ctrl+C to interrupt and save checkpoint\n")
    
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
                    checkpoint_path = trainer.save_checkpoint(tokenizer)
                    print(f"Checkpoint saved to {checkpoint_path}")
            
            # If we've completed an epoch, continue from start
            if trainer.step < max_steps:
                continue
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint before exit...")
        try:
            checkpoint_path = trainer.save_checkpoint(tokenizer)
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
    checkpoint_path = trainer.save_checkpoint(tokenizer, checkpoint_name="final")
    print(f"Final checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
