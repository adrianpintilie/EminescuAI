from transformers import AutoTokenizer

# Configurare
MODEL_NAME = "meta-llama/Llama-3.1-8B"

def count_tokens_and_words_in_files(file_paths):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        token=os.getenv('HF_TOKEN'),
        legacy=False
    )
    
    total_tokens = 0
    unique_tokens = set()
    tokens_per_file = {}
    unique_tokens_per_file = {}
    
    total_words = 0
    unique_words = set()
    words_per_file = {}
    unique_words_per_file = {}
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                
                # Contorizare token-uri
                tokens = tokenizer.encode(text)
                token_count = len(tokens)
                file_unique_tokens = set(tokens)
                tokens_per_file[file_path] = token_count
                unique_tokens_per_file[file_path] = len(file_unique_tokens)
                unique_tokens.update(tokens)
                total_tokens += token_count
                
                # Contorizare cuvinte 
                words = text.split()
                words_per_file[file_path] = len(words)
                unique_words_per_file[file_path] = len(set(words))
                unique_words.update(words)
                total_words += len(words)
                
        except Exception as e:
            print(f"Eroare in procesarea fisierului {file_path}: {str(e)}")
    
    return {
        'tokens': {
            'per_file': tokens_per_file,
            'unique_per_file': unique_tokens_per_file,
            'total_unique': len(unique_tokens),
            'total': total_tokens
        },
        'words': {
            'per_file': words_per_file,
            'unique_per_file': unique_words_per_file,
            'total_unique': len(unique_words),
            'total': total_words
        }
    }

# Fisiere
file_paths = ['data/poezii.txt', 'data/publicistica.txt']
results = count_tokens_and_words_in_files(file_paths)

# Afisare rezultate
for file_path in file_paths:
    print(f"\nFisier: {file_path}")
    print(f"Token-uri:")
    print(f"  - Total: {results['tokens']['per_file'][file_path]}")
    print(f"  - Unice: {results['tokens']['unique_per_file'][file_path]}")
    print(f"Cuvinte:")
    print(f"  - Total: {results['words']['per_file'][file_path]}")
    print(f"  - Unice: {results['words']['unique_per_file'][file_path]}")

print("\nStatistici totale")
print(f"Token-uri:")
print(f"  - Total in fisiere: {results['tokens']['total']}")
print(f"  - Unice in fisiere: {results['tokens']['total_unique']}")
print(f"Cuvinte:")
print(f"  - Total in fisiere: {results['words']['total']}")
print(f"  - Unice in fisiere: {results['words']['total_unique']}")