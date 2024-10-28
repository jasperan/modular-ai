import os
import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from tqdm import tqdm
import warnings
import sys

warnings.filterwarnings('ignore')

class TranscriptGenerator:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup_model()
        
    def setup_model(self):
        """Initialize the model and tokenizer"""
        self.accelerator = Accelerator()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            device_map=self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_safetensors=True)
        self.model, self.tokenizer = self.accelerator.prepare(self.model, self.tokenizer)

    def read_clean_text(self, file_path: str) -> Optional[str]:
        """Read cleaned text file"""
        if not os.path.exists(file_path):
            print(f"Error: File not found at path: {file_path}")
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            return None

    def process_chunk(self, text_chunk: str, chunk_num: int) -> str:
        """Convert text chunk into podcast transcript format"""
        sys_prompt = """
        You are a professional podcast script writer. Convert this text into a natural, 
        engaging podcast script format. Include speaker transitions, natural pauses, 
        and conversational elements. Make it sound authentic and engaging.
        
        Use this format:
        HOST: [Speaker text...]
        GUEST: [Speaker text...]
        
        Make it sound like a natural conversation between experts.
        """
        
        conversation = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": text_chunk},
        ]
        
        prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                temperature=0.8,  # Slightly higher for more creative responses
                top_p=0.9,
                max_new_tokens=1024  # Longer output for transcript format
            )
        
        processed_text = self.tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()
        return processed_text

    def generate_transcript(self, input_file: str, output_file: str = None) -> str:
        """Main function to generate podcast transcript"""
        if not output_file:
            output_file = f"transcript_{os.path.basename(input_file)}"
        
        # Read input text
        text = self.read_clean_text(input_file)
        if text is None:
            return None
        
        # Create chunks (larger chunks for context)
        chunk_size = 2000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        # Process chunks
        transcript = ""
        with open(output_file, 'w', encoding='utf-8') as out_file:
            for chunk_num, chunk in enumerate(tqdm(chunks, desc="Generating transcript")):
                processed_chunk = self.process_chunk(chunk, chunk_num)
                transcript += processed_chunk + "\n\n"
                out_file.write(processed_chunk + "\n\n")
                out_file.flush()
        
        print(f"\nTranscript generation complete!")
        print(f"Output saved to: {output_file}")
        return transcript

def main():
    if len(sys.argv) < 2:
        print("Usage: python transcript_generator.py <input_file> [output_file]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    generator = TranscriptGenerator()
    transcript = generator.generate_transcript(input_file, output_file)
    
    if transcript:
        print("\nPreview of transcript:")
        print("\nBEGINNING:")
        print(transcript[:500])
        print("\n...\n\nEND:")
        print(transcript[-500:])

if __name__ == "__main__":
    main()
