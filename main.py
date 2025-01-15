# @title eminescuAI - meta-llama/Llama-3.2-3B

!pip install -q transformers datasets accelerate bitsandbytes
!pip install -q loralib
!pip install -q git+https://github.com/huggingface/peft.git
!pip install -q wandb

import os
import torch
import gc
import wandb
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset
from huggingface_hub import login
import warnings
warnings.filterwarnings("ignore")

# Configurare
HF_TOKEN = os.environ['HF_TOKEN']
WANDB_API_KEY = os.environ['WANDB_API_KEY'] 
wandb.login(key=WANDB_API_KEY)

MODEL_NAME = "meta-llama/Llama-3.2-3B"  # @param {type:"string"}
#MODEL_NAME = "meta-llama/Llama-3.1-8B"  # @param {type:"string"}

OUTPUT_DIR = "./llama-finetuned"  # @param {type:"string"}
FINAL_MODEL_DIR = "./llama-finetuned-final"  # @param {type:"string"}

def prepare_dataset(file_paths, chunk_size=1024, overlap=128):
    #  Pregatirea setului de date si impartirea lui in mini-seturi ce se suprapun

    chunks = []
    total_chunks = 0
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                
            words = text.split()
            n_chunks = len(words) // (chunk_size - overlap)
            print(f"Procesare {file_path}: {n_chunks} in seturi...")
            
            for i in range(0, len(words), chunk_size - overlap):
                chunk = ' '.join(words[i:i + chunk_size])
                chunks.append(chunk)
                total_chunks += 1
                if total_chunks % 100 == 0:
                    print(f"Au fost create {total_chunks} seturi in total...")
                
            print(f"Completate {file_path}: adaugate {len(chunks) - total_chunks} seturi")
        except Exception as e:
            print(f"Eroare procesare fisier {file_path}: {str(e)}")
    
    return Dataset.from_dict({"text": chunks})

def setup_model_and_tokenizer(base_model_name, hf_token, training=True):
    # Setarea modelului

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_cpu_offload=True,  # Trimitere in memoria CPU
    )
    
    print("Incarca tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        token=hf_token
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Incarcare model de baza...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token
    )
    
    if training:
        # Configurare LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        print("Implementare LoRA...")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=200):
    # Generare raspunsuri folosind modelul 
    
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
            top_p=0.92,
            top_k=50,
            repetition_penalty=1.1
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Curatare memorie GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return generated_text

def clear_gpu_memory():
    # Curatare memorie GPU

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()  

def train_model():
    # Antrenare model cu LoRA

    try:
        # Login Hugging Face
        login(os.environ['HF_TOKEN'])
        
        # Setare model si tokenizer
        model, tokenizer = setup_model_and_tokenizer(MODEL_NAME, os.environ['HF_TOKEN'])
        
        # Pregatiare seturi de date
        print("Pregatire seturi de date...")
        file_paths = ['data/poezii.txt', 'data/publicistica.txt']
        full_dataset = prepare_dataset(file_paths, chunk_size=512)
        dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
        print(f"Creare set de date de antrenament {len(dataset['train'])} si set de date de evaluare {len(dataset['test'])}")

        # Incarcare tokenizare 
        def tokenize_function(examples):
            outputs = tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors=None
            )
            outputs["labels"] = outputs["input_ids"].copy()
            return outputs

        # Rulare tokenizare dataset
        print("Tokenizare set de date...")
        tokenized_train = dataset['train'].map(
            tokenize_function,
            batched=True,
            remove_columns=dataset['train'].column_names
        )
        tokenized_eval = dataset['test'].map(
            tokenize_function,
            batched=True,
            remove_columns=dataset['test'].column_names
        )

        # Personalizare
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            run_name="eminescu_style_finetuning",
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            logging_steps=10,
            learning_rate=2e-4,
            save_total_limit=3,
            remove_unused_columns=False,
            fp16=True,
            optim="paged_adamw_32bit",
            warmup_ratio=0.1,
            load_best_model_at_end=True,
            report_to=["wandb"],
            logging_dir="./logs",
        )

        # Initializare personalizare
        print("Initializare personalizare...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
        )

        # Start personalizare
        print("Start personalizare...")
        trainer.train()

        # Salvare model personalizat
        print("Salvare model...")
        trainer.save_model(FINAL_MODEL_DIR)
        print("Personalizare finalizata cu succes!")

        return True

    except Exception as e:
        print(f"Eroare aparuta in personalizare: {str(e)}")
        return False

def test_model(model_path=FINAL_MODEL_DIR):
    # Testare model

    try:
        # Curatare memorie
        clear_gpu_memory()
        print("Memorie GPU curatata...")

        # Setare model si tokenizare
        print("Setare model si tokenizare...")
        base_model, tokenizer = setup_model_and_tokenizer(
            MODEL_NAME,
            os.environ['HF_TOKEN'],
            training=False
        )
        
        # Incarcare valori LoRA personalizate
        print("Incarcare valori LoRA personalizate...")
        model = PeftModel.from_pretrained(base_model, model_path)
        print("Model incarcat cu succes!")

        # Testare model
        test_prompts = [
            "Scrie o poezie despre natura:",
            "Scrie un paragraf despre viata:",
            "Descrie o seara de toamna:"
        ]
        
        print("\nTestare model...")
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            generated_text = generate_text(model, tokenizer, prompt)
            print("Text generat:", generated_text)
            print("-" * 50)

        return True

    except Exception as e:
        print(f"Eroare aparuta in testare : {str(e)}")
        return False

def main():
    print("Start aplicatie...")
    
    # Golire memorie
    clear_gpu_memory()
    print("Memorie golita...")
    
    # Personalizare model
    print("\n=== Start personalizare ===")
    training_success = train_model()
    
    if training_success:
        # Testare model
        print("\n=== Start testare ===")
        testing_success = test_model()
        
        if testing_success:
            print("\nTestare incheiata cu succes!")
        else:
            print("\nTestare esuata")
    else:
        print("\nTestare esuata!")

if __name__ == "__main__":
    main()