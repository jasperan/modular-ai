import os
import torch
from typing import Optional
from tqdm import tqdm
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

warnings.filterwarnings('ignore')

# Constants
DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
CHUNK_SIZE = 1000  # Size of text chunks for processing

SYS_PROMPT = """
You are a world class text pre-processor, here is the raw data, please parse and return it in a way that is crispy and usable to send to a podcast writer.

The raw data may be messed up with new lines, special characters and you will see fluff that we can remove completely. Remove any details that you think might be useless in a podcast author's transcript.

Remember, the podcast could be on any topic whatsoever so the issues listed above are not exhaustive.

Please be smart with what you remove and be creative ok?

Remember DO NOT START SUMMARIZING THIS, YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED

Be very smart and aggressive with removing details, you will get a running portion of the text and keep returning the processed text.

PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALISATION ETC LIKES

ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions ok?
Here is the text:
"""

def validate_file(file_path: str) -> bool:
    """Validate if file exists and can be read"""
    if not os.path.exists(file_path):
        print(f"Error: File not found at path: {file_path}")
        return False
    return True

def read_text_file(file_path: str, max_chars: int = 100000) -> Optional[str]:
    """Read and return text from file with character limit"""
    if not validate_file(file_path):
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read(max_chars)
            print(f"\nExtraction complete! Total characters: {len(text)}")
            return text
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None

def create_word_bounded_chunks(text: str, target_chunk_size: int) -> list:
    """Split text into chunks at word boundaries"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # +1 for the space
        if current_length + word_length > target_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def setup_model():
    """Initialize and return the model and tokenizer"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    accelerator = Accelerator()
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, use_safetensors=True)
    model, tokenizer = accelerator.prepare(model, tokenizer)
    
    return model, tokenizer, device

def process_chunk(text_chunk: str, chunk_num: int, model, tokenizer, device) -> str:
    """Process a chunk of text using the model"""
    conversation = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": text_chunk},
    ]
    
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=512
        )
    
    processed_text = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()
    
    print(f"\nChunk {chunk_num + 1}:")
    print(f"INPUT TEXT:\n{text_chunk[:200]}...")
    print(f"\nPROCESSED TEXT:\n{processed_text[:200]}...")
    print("="*80)
    
    return processed_text

def preprocess_text(input_file: str, output_file: str = None) -> str:
    """Main function to preprocess text"""
    if not output_file:
        output_file = f"clean_{os.path.basename(input_file)}"
    
    # Read input file
    text = read_text_file(input_file)
    if text is None:
        return None
    
    # Create chunks
    chunks = create_word_bounded_chunks(text, CHUNK_SIZE)
    num_chunks = len(chunks)
    print(f"\nTotal chunks to process: {num_chunks}")
    
    # Setup model
    model, tokenizer, device = setup_model()
    
    # Process chunks
    processed_text = ""
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for chunk_num, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            processed_chunk = process_chunk(chunk, chunk_num, model, tokenizer, device)
            processed_text += processed_chunk + "\n"
            
            # Write chunk immediately to file
            out_file.write(processed_chunk + "\n")
            out_file.flush()
    
    print(f"\nProcessing complete!")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Total chunks processed: {num_chunks}")
    
    return processed_text

if __name__ == "__main__":
    # Example usage
    input_file = "path/to/your/input.txt"  # Replace with your input file path
    processed_text = preprocess_text(input_file)
    
    if processed_text:
        print("\nPreview of final processed text:")
        print("\nBEGINNING:")
        print(processed_text[:1000])
        print("\n...\n\nEND:")
        print(processed_text[-1000:])
