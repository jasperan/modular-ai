import os
from typing import Optional, Dict
import trafilatura
import json
from urllib.parse import urlparse
from datetime import datetime
from tqdm import tqdm
import warnings
import sys
import argparse
import textwrap

warnings.filterwarnings('ignore')

class WebExtractor:
    def __init__(self):
        """Initialize web content extractor"""
        self.setup_output_directory()
        
    def setup_output_directory(self):
        """Create output directory for extracted content"""
        self.output_dir = "extracted_content"
        os.makedirs(self.output_dir, exist_ok=True)

    def sanitize_filename(self, url: str) -> str:
        """Create a safe filename from URL"""
        parsed = urlparse(url)
        filename = parsed.netloc + parsed.path
        # Remove or replace invalid filename characters
        filename = "".join(c if c.isalnum() or c in "._- " else "_" for c in filename)
        return filename.strip("._- ") or "extracted_content"

    def extract_content(self, url: str, output_file: Optional[str] = None) -> Optional[Dict]:
        """Extract content from a webpage"""
        try:
            # Download and extract content
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                print(f"Error: Could not download content from {url}")
                return None

            # Extract main content
            content = trafilatura.extract(
                downloaded,
                include_links=True,
                include_images=True,
                include_tables=True,
                output_format='json'
            )
            
            if not content:
                print(f"Error: Could not extract content from {url}")
                return None

            # Parse JSON content
            content_dict = json.loads(content)
            
            # Add metadata
            content_dict['metadata'] = {
                'url': url,
                'extraction_date': datetime.now().isoformat(),
                'word_count': len(content_dict.get('text', '').split())
            }

            # Save to file if output_file specified
            if not output_file:
                output_file = os.path.join(self.output_dir, f"{self.sanitize_filename(url)}.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(content_dict, f, indent=2, ensure_ascii=False)
            
            print(f"\nContent extracted successfully!")
            print(f"Output saved to: {output_file}")
            print(f"Word count: {content_dict['metadata']['word_count']}")
            
            return content_dict

        except Exception as e:
            print(f"Error extracting content: {str(e)}")
            return None

    def extract_from_urls(self, urls: list, output_dir: Optional[str] = None) -> Dict[str, Dict]:
        """Extract content from multiple URLs"""
        if output_dir:
            self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)
        
        results = {}
        for url in tqdm(urls, desc="Processing URLs"):
            output_file = os.path.join(self.output_dir, f"{self.sanitize_filename(url)}.json")
            content = self.extract_content(url, output_file)
            if content:
                results[url] = content
        
        # Save summary
        summary_file = os.path.join(self.output_dir, "extraction_summary.json")
        summary = {
            'extraction_date': datetime.now().isoformat(),
            'total_urls': len(urls),
            'successful_extractions': len(results),
            'urls_processed': list(results.keys())
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        return results

    def get_text_content(self, content_dict: Dict) -> str:
        """Extract plain text from content dictionary"""
        text_content = []
        
        # Add title
        if content_dict.get('title'):
            text_content.append(f"Title: {content_dict['title']}\n")
        
        # Add author
        if content_dict.get('author'):
            text_content.append(f"Author: {content_dict['author']}\n")
        
        # Add main content
        if content_dict.get('text'):
            text_content.append(content_dict['text'])
        
        # Add comments if available
        if content_dict.get('comments'):
            text_content.append("\nComments:")
            text_content.append(content_dict['comments'])
        
        return "\n".join(text_content)

def main():
    """Main function with argparse for command line arguments"""
    parser = argparse.ArgumentParser(
        description='Extract content from web pages using trafilatura',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''
            Examples:
              %(prog)s https://example.com/article output.txt
              %(prog)s --file urls.txt output_directory/
            ''')
    )
    
    # Add arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        'url',
        nargs='?',
        help='URL to extract content from'
    )
    group.add_argument(
        '--file',
        help='File containing list of URLs (one per line)'
    )
    
    parser.add_argument(
        'output',
        help='Output file for single URL or directory for multiple URLs'
    )
    
    parser.add_argument(
        '--include-comments',
        action='store_true',
        help='Include comments in extracted content'
    )
    
    parser.add_argument(
        '--format',
        choices=['json', 'text'],
        default='json',
        help='Output format (default: json)'
    )
    
    args = parser.parse_args()
    
    extractor = WebExtractor()
    
    if args.file:
        # Process multiple URLs from file
        try:
            with open(args.file, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
        except Exception as e:
            parser.error(f"Error reading URL file: {str(e)}")
        
        if not urls:
            parser.error("No valid URLs found in file")
        
        results = extractor.extract_from_urls(urls, args.output)
        print(f"\nProcessed {len(results)}/{len(urls)} URLs successfully")
        print(f"Results saved in: {extractor.output_dir}")
        
    else:
        # Process single URL
        content = extractor.extract_content(args.url, args.output)
        if content:
            text_content = extractor.get_text_content(content)
            print("\nPreview of extracted content:")
            print("-" * 50)
            print(text_content[:500] + "...")
            print("-" * 50)
            
            if args.format == 'text':
                # Save as text file
                text_output = args.output if args.output.endswith('.txt') else f"{args.output}.txt"
                with open(text_output, 'w', encoding='utf-8') as f:
                    f.write(text_content)
                print(f"\nText content saved to: {text_output}")

if __name__ == "__main__":
    main()
