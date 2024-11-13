import torch
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import os
from datetime import datetime
import gc
import glob

class MemoryTracker:
    @staticmethod
    def get_gpu_memory():
        if torch.cuda.is_available():
            return f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        return "GPU not available"

    @staticmethod
    def clear_gpu_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

def setup_gpu():
    """Set up GPU if available."""
    print("\n=== GPU Setup ===")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"✓ Initial GPU memory usage: {MemoryTracker.get_gpu_memory()}")
    else:
        device = torch.device("cpu")
        print("✗ GPU not available, using CPU")
    return device

def get_latest_progress_file():
    """Get the latest progress file and its count."""
    progress_files = glob.glob('progress/responses_*_*.json')
    if not progress_files:
        return None, 0
    
    # Extract count and find latest file
    latest_file = max(progress_files, key=os.path.getctime)
    count = int(latest_file.split('_')[1])
    return latest_file, count

def load_previous_responses():
    """Load the latest progress file."""
    latest_file, count = get_latest_progress_file()
    if latest_file:
        print(f"Found checkpoint: {latest_file}")
        print(f"Completed questions: {count}")
        df = pd.read_json(latest_file)
        return df, count
    return None, 0

def load_reference_dataset():
    """Load dataset from HuggingFace."""
    print("\n=== Loading Reference Dataset ===")
    try:
        # Try to load from local file first
        if os.path.exists('data/reference_dataset.json'):
            print("Loading from local cache...")
            df = pd.read_json('data/reference_dataset.json')
            print(f"✓ Loaded {len(df)} questions from cache")
            return df
    except Exception as e:
        print(f"Cache load failed: {e}")

    print("1. Downloading from HuggingFace...")
    dataset = load_dataset("instructlab/InstructLabCommunity")
    print("✓ Download complete")
    
    print("2. Processing dataset...")
    df = pd.DataFrame(dataset['train'])
    df['question'] = df['messages'].apply(lambda x: x[0]['content'])
    df['reference_answer'] = df['messages'].apply(lambda x: x[1]['content'])
    df['category'] = df['metadata'].apply(lambda x: x.get('path') if isinstance(x, dict) else x)
    
    # Clean up memory
    del dataset
    MemoryTracker.clear_gpu_memory()
    
    print("3. Saving to cache...")
    os.makedirs('data', exist_ok=True)
    df[['question', 'reference_answer', 'category']].to_json(
        'data/reference_dataset.json', 
        orient='records'
    )
    
    return df

def setup_model(device):
    """Download and setup the model."""
    print("\n=== Model Setup ===")
    print("1. Setting up tokenizer...")
    model_name = "instructlab/granite-7b-lab"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        truncation_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer ready")
    
    print("2. Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model.config.pad_token_id = model.config.eos_token_id
    print(f"✓ Model loaded (Memory usage: {MemoryTracker.get_gpu_memory()})")
    
    return model, tokenizer

def format_prompt(question):
    """Format the prompt to get more direct answers."""
    return f"""Below is a question. Provide a clear and concise answer.

Question: {question}

Answer:"""

def get_model_response(model, tokenizer, question, device):
    """Get response from model for a single question."""
    try:
        prompt = format_prompt(question)
        inputs = tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=2048
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                min_new_tokens=30,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Answer:")[-1].strip()
        
    except Exception as e:
        print(f"\n✗ Error processing question: {question[:100]}...")
        print(f"✗ Error: {str(e)}")
        return f"Error: {str(e)}"

def generate_responses(df, model, tokenizer, device, start_idx=0):
    """Generate responses for remaining questions."""
    print(f"\n=== Resuming Response Generation from Question {start_idx} ===")
    start_time = datetime.now()
    
    # Load previous responses if any
    if start_idx > 0:
        prev_df, _ = load_previous_responses()
        model_responses = prev_df['model_response'].tolist()
        print(f"Loaded {len(model_responses)} previous responses")
    else:
        model_responses = []
    
    try:
        for idx in tqdm(range(start_idx, len(df)), initial=start_idx, total=len(df)):
            question = df.iloc[idx]['question']
            response = get_model_response(model, tokenizer, question, device)
            model_responses.append(response)
            
            # Progress tracking and saving
            if (idx + 1) % 10 == 0:
                elapsed = datetime.now() - start_time
                avg_time = elapsed / (idx + 1 - start_idx)
                remaining = avg_time * (len(df) - idx - 1)
                
                print(f"\nProgress Update ({idx + 1}/{len(df)}):")
                print(f"Average time per question: {avg_time.total_seconds():.2f}s")
                print(f"Estimated time remaining: {str(remaining).split('.')[0]}")
                print(f"GPU Memory: {MemoryTracker.get_gpu_memory()}")
                
                # Save progress
                save_progress(df.iloc[:idx+1], model_responses, idx+1)
                MemoryTracker.clear_gpu_memory()
            
            # Sample outputs
            if (idx + 1) % 50 == 0:
                print(f"\nSample Q&A #{idx + 1}:")
                print(f"Q: {question[:150]}...")
                print(f"A: {response[:150]}...")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving progress...")
        save_progress(df.iloc[:len(model_responses)], model_responses, len(model_responses))
        raise
    
    return model_responses

def save_progress(df_subset, responses, current_idx):
    """Save progress to file."""
    if len(df_subset) != len(responses):
        print("⚠️ Warning: Length mismatch in progress saving")
        min_len = min(len(df_subset), len(responses))
        df_subset = df_subset.iloc[:min_len]
        responses = responses[:min_len]
    
    progress_df = df_subset.copy()
    progress_df['model_response'] = responses
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"progress/responses_{current_idx}_{timestamp}.json"
    
    progress_df.to_json(filename, orient='records', indent=2)
    print(f"✓ Progress saved: {filename}")

def save_final_results(df, responses):
    """Save final results."""
    print("\n=== Saving Final Results ===")
    os.makedirs('results', exist_ok=True)
    
    final_df = df.iloc[:len(responses)].copy()
    final_df['model_response'] = responses
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"results/final_responses_{timestamp}.json"
    
    final_df.to_json(filename, orient='records', indent=2)
    print(f"✓ Final results saved: {filename}")
    
    # Save summary statistics
    summary = {
        'total_questions': len(final_df),
        'completion_time': str(datetime.now()),
        'categories': final_df['category'].value_counts().to_dict(),
        'avg_response_length': sum(len(str(r)) for r in responses) / len(responses)
    }
    
    with open(f"results/summary_{timestamp}.json", 'w') as f:
        json.dump(summary, f, indent=2)

def main():
    print(f"\n=== Resuming Response Generation ===")
    print(f"Start Time: {datetime.now()}")
    
    try:
        # Find latest progress
        latest_file, last_count = get_latest_progress_file()
        if latest_file:
            print(f"Resuming from question {last_count}")
            print(f"Using checkpoint: {latest_file}")
        
        # Setup
        device = setup_gpu()
        df = load_reference_dataset()
        model, tokenizer = setup_model(device)
        
        # Generate remaining responses
        responses = generate_responses(df, model, tokenizer, device, start_idx=last_count)
        
        # Save final results
        save_final_results(df, responses)
        
        print(f"\n=== Process Complete ===")
        print(f"End Time: {datetime.now()}")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print("\n=== Error Occurred ===")
        print(f"Error: {str(e)}")
        import traceback
        print(traceback.format_exc())
    finally:
        MemoryTracker.clear_gpu_memory()

if __name__ == "__main__":
    main()
