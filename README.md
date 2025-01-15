# EminescuAI - LLaMA Fine-tuning Project

A project for fine-tuning LLaMA models to generate text in the style of the influent Romanian poet Mihai Eminescu. One complete trained model on Llama can be found here: https://huggingface.co/adrianpintilie/EminescuAI

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU
- Hugging Face account with access to LLaMA models
- Weights & Biases account

## Required Dependencies

```bash
pip install transformers datasets accelerate bitsandbytes
pip install loralib
pip install git+https://github.com/huggingface/peft.git
pip install wandb
```

## Project Structure

- `main.py` - Main training and testing pipeline
- `count.py` - Utility for counting tokens and words in input files
- `evaluare.py` - Model evaluation and comparison script
- `setari.py` - Configuration file for API tokens
- `data/poezii.txt` and `data/publicistica.txt` - Training content

## Setup

1. First, configure your API tokens in `setari.py`:
   - `HF_TOKEN` - Your Hugging Face API token
   - `WANDB_API_KEY` - Your Weights & Biases API key
   - Execute the script

2. Prepare your training data:
   - The training data is not included in this repository due to copyright considerations
   - You will need to source your own texts, particularly works by Mihai Eminescu
   - Create a `data/` directory and place your text files there:
     - `data/poezii.txt` - Poetry dataset
     - `data/publicistica.txt` - Prose dataset
   - Ensure your texts are:
     - In plain text format (.txt)
     - UTF-8 encoded
     - Clean and properly formatted
     - Comply with copyright laws in your jurisdiction

3. Model Configuration:
   - Default base model: `meta-llama/Llama-3.2-3B`
   - Alternative model: `meta-llama/Llama-3.1-8B` (commented out)
   - Output directories:
     - `./llama-finetuned` - Training checkpoints
     - `./llama-finetuned-final` - Final model

## Usage

### Training the Model

Run the main training script (do not forget first to run setari.py):

```bash
python main.py
```

This will:
1. Prepare and tokenize the dataset
2. Initialize the model with LoRA configuration
3. Train the model using the specified parameters
4. Save the fine-tuned model

### Evaluating the Model

To compare the base model with your fine-tuned version:

```bash
python evaluare.py
```

This script will generate text using both models for comparison using test prompts like:
- "Scrie în română o poezie despre natură:"
- "Scrie în română un paragraf despre viață:"
- "Descrie în română o seară de toamnă:"

### Analyzing Text Data

To analyze your training data:

```bash
python count.py
```

This will provide statistics about:
- Total tokens
- Unique tokens
- Total words
- Unique words

## Training Configuration

The training uses the following key parameters:
- 3 epochs
- Batch size: 2
- Gradient accumulation steps: 4
- Learning rate: 2e-4
- LoRA Configuration:
  - r=16
  - alpha=32
  - dropout=0.05
  - Target modules: query, key, value, and output projections

## Memory Management

The project includes memory management features:
- 4-bit quantization
- CPU offloading
- Automatic GPU memory clearing
- Gradient accumulation

## Monitoring

Training progress can be monitored through:
- Weights & Biases dashboard
- Local logging in `./logs` directory

## Safety and Ethics

Please ensure you have appropriate permissions for:
- Access to LLaMA models
- Usage of training data
- API tokens and credentials

Do not share API tokens publicly or commit them to version control.

## Error Handling

The code includes comprehensive error handling for:
- File processing
- Model loading
- Training process
- Memory management

Errors are logged with descriptive messages to help with debugging.

## Contributing

When contributing to this project:
1. Use consistent code formatting
2. Add error handling for new features
3. Document any new parameters or functions
4. Test memory usage with different batch sizes
5. Update this README for significant changes

## License

Apache License 2.0

Copyright 2025

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.