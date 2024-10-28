import os
from typing import Optional, Dict
import xml.etree.ElementTree as ET
from datetime import datetime
import hashlib
import mimetypes
import argparse
import json
import shutil
from tqdm import tqdm
import warnings
import sys

warnings.filterwarnings('ignore')

class PodcastPublisher:
    def __init__(self, config_file: Optional[str] = None):
        self.config = self.load_config(config_file)
        self.setup_directories()
        
    def load_config(self, config_file: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            "podcast_title": "AI Generated Podcast",
            "podcast_description": "Automatically generated podcast from text content",
            "author": "AI Publisher",
            "email": "ai@example.com",
            "language": "en-us",
            "category": "Technology",
            "explicit": "no",
            "output_dir": "podcast_output",
            "rss_filename": "feed.xml",
            "base_url": "http://example.com/podcasts/"  # Base URL for RSS feed
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                default_config.update(loaded_config)
            except Exception as e:
                print(f"Error loading config file: {str(e)}")
        
        return default_config

    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.config['output_dir'], exist_ok=True)
        os.makedirs(os.path.join(self.config['output_dir'], 'audio'), exist_ok=True)
        os.makedirs(os.path.join(self.config['output_dir'], 'metadata'), exist_ok=True)

    def prepare_audio(self, audio_file: str) -> Optional[str]:
        """Prepare audio file for publishing"""
        if not os.path.exists(audio_file):
            print(f"Error: Audio file not found: {audio_file}")
            return None
            
        # Copy audio to output directory with sanitized filename
        filename = os.path.basename(audio_file)
        clean_filename = ''.join(c for c in filename if c.isalnum() or c in '._- ').strip()
        output_path = os.path.join(self.config['output_dir'], 'audio', clean_filename)
        
        try:
            shutil.copy2(audio_file, output_path)
            return clean_filename
        except Exception as e:
            print(f"Error copying audio file: {str(e)}")
            return None

    def generate_episode_metadata(self, 
                                audio_file: str, 
                                title: str, 
                                description: str, 
                                transcript_file: Optional[str] = None) -> Dict:
        """Generate metadata for podcast episode"""
        file_size = os.path.getsize(audio_file)
        
        # Calculate MD5 hash of file
        md5_hash = hashlib.md5()
        with open(audio_file, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                md5_hash.update(chunk)
        
        metadata = {
            "title": title,
            "description": description,
            "pubDate": datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT"),
            "duration": "00:30:00",  # TODO: Calculate actual duration
            "file": {
                "url": os.path.join(self.config['base_url'], 'audio', os.path.basename(audio_file)),
                "size": file_size,
                "type": mimetypes.guess_type(audio_file)[0] or "audio/mpeg",
                "md5": md5_hash.hexdigest()
            }
        }
        
        # Add transcript if available
        if transcript_file and os.path.exists(transcript_file):
            transcript_filename = os.path.basename(transcript_file)
            transcript_dest = os.path.join(self.config['output_dir'], 'metadata', transcript_filename)
            shutil.copy2(transcript_file, transcript_dest)
            metadata["transcript_file"] = transcript_filename
        
        return metadata

    def update_rss_feed(self, episode_metadata: Dict):
        """Update RSS feed with new episode"""
        rss_file = os.path.join(self.config['output_dir'], self.config['rss_filename'])
        
        # Create or load RSS feed
        if os.path.exists(rss_file):
            tree = ET.parse(rss_file)
            root = tree.getroot()
            channel = root.find('channel')
        else:
            root = ET.Element('rss', version="2.0")
            channel = ET.SubElement(root, 'channel')
            
            # Add podcast metadata
            ET.SubElement(channel, 'title').text = self.config['podcast_title']
            ET.SubElement(channel, 'description').text = self.config['podcast_description']
            ET.SubElement(channel, 'language').text = self.config['language']
            ET.SubElement(channel, 'author').text = self.config['author']
            ET.SubElement(channel, 'explicit').text = self.config['explicit']
            ET.SubElement(channel, 'category').text = self.config['category']
        
        # Add new episode
        item = ET.SubElement(channel, 'item')
        ET.SubElement(item, 'title').text = episode_metadata['title']
        ET.SubElement(item, 'description').text = episode_metadata['description']
        ET.SubElement(item, 'pubDate').text = episode_metadata['pubDate']
        ET.SubElement(item, 'duration').text = episode_metadata['duration']
        
        enclosure = ET.SubElement(item, 'enclosure')
        enclosure.set('url', episode_metadata['file']['url'])
        enclosure.set('length', str(episode_metadata['file']['size']))
        enclosure.set('type', episode_metadata['file']['type'])
        
        # Save updated RSS feed
        tree = ET.ElementTree(root)
        tree.write(rss_file, encoding='utf-8', xml_declaration=True)

    def publish_episode(self, 
                       audio_file: str, 
                       title: str, 
                       description: str, 
                       transcript_file: Optional[str] = None) -> bool:
        """Publish a new podcast episode"""
        try:
            # Prepare audio file
            prepared_audio = self.prepare_audio(audio_file)
            if not prepared_audio:
                return False
            
            # Generate metadata
            metadata = self.generate_episode_metadata(
                audio_file=audio_file,
                title=title,
                description=description,
                transcript_file=transcript_file
            )
            
            # Save episode metadata
            metadata_file = os.path.join(
                self.config['output_dir'], 
                'metadata', 
                f"episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            # Update RSS feed
            self.update_rss_feed(metadata)
            
            print(f"\nEpisode published successfully!")
            print(f"Audio: {prepared_audio}")
            print(f"Metadata: {metadata_file}")
            print(f"RSS feed updated: {self.config['rss_filename']}")
            return True
            
        except Exception as e:
            print(f"Error publishing episode: {str(e)}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Publish a podcast episode')
    parser.add_argument('audio_file', help='Path to the audio file')
    parser.add_argument('title', help='Episode title')
    parser.add_argument('description', help='Episode description')
    parser.add_argument('--transcript', help='Path to transcript file (optional)')
    parser.add_argument('--config', help='Path to config file (optional)')
    
    args = parser.parse_args()
    
    publisher = PodcastPublisher(config_file=args.config)
    success = publisher.publish_episode(
        audio_file=args.audio_file,
        title=args.title,
        description=args.description,
        transcript_file=args.transcript
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
