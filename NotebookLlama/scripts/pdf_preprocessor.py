import os
import PyPDF2
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from tqdm import tqdm
import warnings
import sys

warnings.filterwarnings('ignore')

class PDFPreprocessor:
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

    def validate_pdf(self, file_path: str) -> bool:
        """Validate if file exists and is a PDF"""
        if not os.path.exists(file_path):
            print(f"Error: File not found at path: {file_path}")
            return False
        if not file_path.lower().endswith('.pdf'):
            print("Error: File is not a PDF")
            return False
        return True

    def get_pdf_metadata(self, file_path: str) -> Optional[dict]:
        """Extract metadata from PDF"""
        if not self.validate_pdf(file_path):
            return None
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata = {
                    'num_pages': len(pdf_reader.pages),
                    'metadata': pdf_reader.metadata
                }
                return metadata
        except Exception as e:
            print(f"Error extracting metadata: {str(e)}")
            return None

    def extract_text_from_pdf(self, file_path: str, max_chars: int = 100000) -> Optional[str]:
        """Extract text from PDF with character limit"""
        if not self.validate_pdf(file_path):
            return None
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                print(f"Processing PDF with {num_pages} pages...")
                
                extracted_text = []
                total_chars = 0
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    if total_chars + len(text) > max_chars:
                        remaining_chars = max_chars - total_chars
                        extracted_text.append(text[:remaining_chars])
                        print(f"Reached {max_chars} character limit at page {page_num + 1}")
                        break
                    
                    extracted_text.append(text)
                    total_chars += len(text)
                    print(f"Processed page {page_num + 1}/{num_pages}")
                
                final_text = '\n'.join(extracted_text)
                print(f"\nExtraction complete! Total characters: {len(final_text)}")
                return final_text
                
        except Exception as e:
            print(f"Error extracting text: {str(e)}")
            return None

    def process_chunk(self, text_chunk: str, chunk_num: int) -> str:
        """Process a chunk of text using the model"""
        sys_prompt = """
        You are a world class text pre-processor. Clean up this text for a podcast writer.
        Remove unnecessary formatting, special characters, and fluff.
        DO NOT summarize - only clean and rewrite when needed.
        Start directly with the processed text, no acknowledgements.
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
                temperature=0.7,
                top_p=0.9,
                max_new_tokens=512
            )
        
        processed_text = self.tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()
        return processed_text

    def process_pdf(self, pdf_path: str, output_file: str = None) -> str:
        """Main function to process PDF and save cleaned text"""
        if not output_file:
            output_file = f"clean_{os.path.basename(pdf_path)}.txt"
        
        # Extract metadata
        metadata = self.get_pdf_metadata(pdf_path)
        if metadata:
            print("\nPDF Metadata:")
            print(f"Number of pages: {metadata['num_pages']}")
            print("Document info:")
            for key, value in metadata['metadata'].items():
                print(f"{key}: {value}")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        if text is None:
            return None
        
        # Create chunks
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        
        # Process chunks
        processed_text = ""
        with open(output_file, 'w', encoding='utf-8') as out_file:
            for chunk_num, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
                processed_chunk = self.process_chunk(chunk, chunk_num)
                processed_text += processed_chunk + "\n"
                out_file.write(processed_chunk + "\n")
                out_file.flush()
        
        print(f"\nProcessing complete!")
        print(f"Output saved to: {output_file}")
        return processed_text

def main():
    # Example usage
    preprocessor = PDFPreprocessor()
    pdf_path = "./resources/example.pdf"  # Replace with your PDF path
    processed_text = preprocessor.process_pdf(pdf_path)
    
    if processed_text:
        print("\nPreview of processed text:")
        print("\nBEGINNING:")
        print(processed_text[:500])
        print("\n...\n\nEND:")
        print(processed_text[-500:])

if __name__ == "__main__":
    main()
