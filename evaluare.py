import os
import torch
import gc
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset
from huggingface_hub import login
import warnings
warnings.filterwarnings("ignore")

# Configurare
BASE_MODEL_NAME = "meta-llama/Llama-3.2-3B"
#BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B"  

FINETUNED_MODEL_PATH = "./llama-finetuned-final"
HF_TOKEN = os.environ['HF_TOKEN']

# Login la Hugging Face
login(token=HF_TOKEN)

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def compare_models(
    base_model_name,
    finetuned_model_path,
    prompt,
    hf_token,
    max_length=200,
    num_comparisons=1
):
    # Comparare rezultate modele
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
    )
    
    print("Incarcare tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        token=hf_token,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Incarcare model de baza
    print("Incarcare model de baza...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True
    )
    base_model.eval() 
    
    # Incarcare model personalizat
    print("Incarcare model personalizat...")
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True
    )
    finetuned_model = PeftModel.from_pretrained(
        finetuned_model, 
        finetuned_model_path,
        trust_remote_code=True
    )
    finetuned_model.eval()  
    
    results = {
        "prompt": prompt,
        "base_model_generations": [],
        "finetuned_model_generations": []
    }
    
    # Parametri model inferenta
    gen_params = {
        "max_length": max_length,
        "temperature": 0.7,  
        "do_sample": True,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.2,
        "num_return_sequences": 1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    print("\nGenerare comparatii...")
    for i in range(num_comparisons):
        print(f"\nComparatia {i+1}/{num_comparisons}")
        
        # Model de baza
        print("Generare cu model de baza...")
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)
            base_outputs = base_model.generate(
                **inputs,
                **gen_params
            )
            base_text = tokenizer.decode(base_outputs[0], skip_special_tokens=True)
            results["base_model_generations"].append(base_text)
        
        # Model personalizat
        print("Generare cu model personalizat...")
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(finetuned_model.device)
            finetuned_outputs = finetuned_model.generate(
                **inputs,
                **gen_params
            )
            finetuned_text = tokenizer.decode(finetuned_outputs[0], skip_special_tokens=True)
            results["finetuned_model_generations"].append(finetuned_text)
        
        # Curatare memorie
        clear_gpu_memory()
    
    return results

if __name__ == "__main__":
    test_prompts = [
        "Scrie în română o poezie despre natură:",
        "Scrie în română un paragraf despre viață:",
        "Descrie în română o seară de toamnă:"
    ]
    
    print("Start aplicatie...")
    clear_gpu_memory()
    print("Memorie golita...")

    # Run comparisons
    for prompt in test_prompts:
        print(f"\n{'='*50}\nComparare prompt: {prompt}\n{'='*50}")
        try:
            results = compare_models(
                BASE_MODEL_NAME,
                FINETUNED_MODEL_PATH,
                prompt,
                HF_TOKEN,
                max_length=200,
                num_comparisons=1
            )
            
            for i in range(len(results["base_model_generations"])):
                print("\nModel de baza:")
                print(results["base_model_generations"][i])
                print("\nModel personalizat:")
                print(results["finetuned_model_generations"][i])
                print("\n" + "-"*50)
        except Exception as e:
            print(f"Eroare procesare prompt '{prompt}': {str(e)}")
        finally:
            clear_gpu_memory() 