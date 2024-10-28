import os
from typing import Optional, List, Dict
import torch
from TTS.api import TTS
import soundfile as sf
import re
from tqdm import tqdm
import warnings
import sys

warnings.filterwarnings('ignore')

class AudioGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup_tts()
        
    def setup_tts(self):
        """Initialize TTS models for different speakers"""
        print("Loading TTS models...")
        
        # Initialize main TTS model
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        
        # Define voice presets
        self.voices = {
            "HOST": "p326",  # Default host voice
            "GUEST": "p225"  # Default guest voice
        }
        
    def parse_transcript(self, transcript: str) -> List[Dict[str, str]]:
        """Parse transcript into segments with speaker labels"""
        segments = []
        current_speaker = None
        current_text = []
        
        for line in transcript.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check for speaker labels (HOST: or GUEST:)
            speaker_match = re.match(r'^(HOST|GUEST):\s*(.*)$', line)
            
            if speaker_match:
                # If we have accumulated text for previous speaker, save it
                if current_speaker and current_text:
                    segments.append({
                        "speaker": current_speaker,
                        "text": ' '.join(current_text)
                    })
                    current_text = []
                
                # Start new speaker segment
                current_speaker = speaker_match.group(1)
                text = speaker_match.group(2)
                if text:
                    current_text.append(text)
            else:
                # Continue with current speaker
                if current_speaker:
                    current_text.append(line)
        
        # Add final segment
        if current_speaker and current_text:
            segments.append({
                "speaker": current_speaker,
                "text": ' '.join(current_text)
            })
        
        return segments

    def generate_audio_segment(self, text: str, speaker: str, output_path: str) -> bool:
        """Generate audio for a single segment"""
        try:
            # Get voice preset for speaker
            voice = self.voices.get(speaker, self.voices["HOST"])
            
            # Generate audio
            wav = self.tts.tts(
                text=text,
                speaker=voice,
                language="en"
            )
            
            # Save audio file
            sf.write(output_path, wav, self.tts.synthesizer.output_sample_rate)
            return True
            
        except Exception as e:
            print(f"Error generating audio for segment: {str(e)}")
            return False

    def generate_podcast(self, transcript_file: str, output_dir: str = "audio_output") -> Optional[str]:
        """Generate full podcast audio from transcript"""
        # Validate input
        if not os.path.exists(transcript_file):
            print(f"Error: Transcript file not found: {transcript_file}")
            return None
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Read transcript
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript = f.read()
        except Exception as e:
            print(f"Error reading transcript: {str(e)}")
            return None
        
        # Parse transcript into segments
        segments = self.parse_transcript(transcript)
        if not segments:
            print("Error: No valid segments found in transcript")
            return None
        
        # Generate audio for each segment
        audio_files = []
        for i, segment in enumerate(tqdm(segments, desc="Generating audio segments")):
            output_path = os.path.join(output_dir, f"segment_{i:03d}.wav")
            success = self.generate_audio_segment(
                text=segment["text"],
                speaker=segment["speaker"],
                output_path=output_path
            )
            if success:
                audio_files.append(output_path)
        
        if not audio_files:
            print("Error: No audio segments were generated")
            return None
        
        # Combine all segments into final podcast
        final_output = os.path.join(output_dir, "final_podcast.wav")
        try:
            # Read all audio segments
            segments_data = []
            sample_rate = None
            for file in audio_files:
                data, sr = sf.read(file)
                if sample_rate is None:
                    sample_rate = sr
                segments_data.append(data)
            
            # Concatenate all segments
            combined_audio = torch.cat([torch.tensor(seg) for seg in segments_data])
            
            # Save final podcast
            sf.write(final_output, combined_audio.numpy(), sample_rate)
            
            print(f"\nPodcast generation complete!")
            print(f"Final output saved to: {final_output}")
            
            # Clean up individual segments
            for file in audio_files:
                os.remove(file)
            
            return final_output
            
        except Exception as e:
            print(f"Error combining audio segments: {str(e)}")
            return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python audio_generator.py <transcript_file> [output_directory]")
        sys.exit(1)
        
    transcript_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "audio_output"
    
    generator = AudioGenerator()
    output_file = generator.generate_podcast(transcript_file, output_dir)
    
    if output_file:
        print(f"\nPodcast audio generated successfully at: {output_file}")
    else:
        print("\nFailed to generate podcast audio")
        sys.exit(1)

if __name__ == "__main__":
    main()
