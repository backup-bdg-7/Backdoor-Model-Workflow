# Render.com Free Tier Compatibility Notes

This document outlines important considerations for running the AI Model Training Platform on Render.com's free tier.

## Free Tier Limitations

The Render.com free tier has the following limitations:

- **Memory**: ~512MB RAM per service (varies by service type)
- **CPU**: Shared CPU, limited compute resources
- **Disk**: No persistent disk storage, only ephemeral `/tmp` directory
- **Build Minutes**: Limited monthly build minutes
- **Bandwidth**: Limited monthly bandwidth
- **Concurrent Processes**: Limited number of concurrent processes
- **Sleep Mode**: Free tier services sleep after periods of inactivity

## Optimization Strategies

To work within these constraints, the platform implements these strategies:

### Memory Management

1. **Distributed Architecture**: Split functionality across multiple services to distribute memory usage
2. **Gradient Checkpointing**: Reduce memory usage during training backpropagation
3. **Streaming Dataset Processing**: Process datasets in chunks rather than loading entirely
4. **Dynamic Batch Sizing**: Adjust batch sizes based on available memory
5. **Memory Monitoring**: Actively track memory usage and release resources when needed

### Storage Management

1. **Ephemeral Storage**: Use `/tmp` directory for all storage needs
2. **Minimal Dataset Samples**: Limit the amount of training data to fit in memory
3. **Checkpoint Management**: Only keep the most recent checkpoints
4. **Export Cleanup**: Automatically clean up exports after download

### Performance Optimization

1. **Small Model Configurations**: Default to smaller model architectures
2. **Mixed Precision Training**: Use FP16/BF16 for efficient computation
3. **Progressive Training**: Train in stages with increasing complexity
4. **Reduced Worker Count**: Use single-worker configuration for services

## User Guidelines

When using the platform on the free tier:

1. Select "small" model size in the training interface
2. Limit dataset samples to under 10,000
3. Keep training epochs low (1-3)
4. Be patient with training progress - resources are limited
5. Download exports promptly as they may be cleaned up
6. Expect services to have higher latency than paid tiers

## Detection and Adaptation

The application automatically detects when running on the free tier and adjusts its behavior:

1. Reduces parallelism in data processing
2. Uses more conservative memory settings
3. Implements more aggressive cleanup policies
4. Provides clearer feedback about resource limitations

## Upgrading

To remove these limitations, upgrading to a paid tier on Render.com will enable:

1. More memory for larger models
2. Persistent disk storage
3. Better performance
4. No sleep mode
5. More concurrent processes
