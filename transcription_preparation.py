#!/usr/bin/env python3
"""
Batch Audio Processor for BioTTS Transcription
Processes multiple audio files: trim to 11 minutes, normalize volume, and transcribe

Usage:
    python batch_process_audio.py --input-dir raw_data --output-dir whisper_transcription
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple
import shutil

import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import torch
import torchaudio
import time
from typing import Dict

class BioTTSTranscriptionCreator:
    """
    High-quality transcription creator for BioTTS training
    
    Biological metaphor: DNA sequencer with precise nucleotide identification
    """
    
    def __init__(self, 
                 model_size: str = "large-v3",
                 device: str = "auto",
                 compute_type: str = "float16"):
        """
        Initialize transcription creator
        
        Args:
            model_size: WhisperX model size ("large-v3", "large-v2", "medium", "small")
            device: Device to use ("auto", "cuda", "cpu")
            compute_type: Computation precision ("float16", "float32", "int8")
        """
        import torch
        
        self.model_size = model_size
        self.compute_type = compute_type
        
        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Auto-adjust compute type for CPU
        if self.device == "cpu" and compute_type == "float16":
            self.compute_type = "float32"
            logger.info("🔄 Auto-switched to float32 for CPU compatibility")
        else:
            self.compute_type = compute_type
            
        # Check GPU memory for optimal settings
        if self.device == "cuda":
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory < 8:
                logger.warning(f"GPU memory: {gpu_memory:.1f}GB - consider using smaller model")
                if self.model_size == "large-v3":
                    self.model_size = "large-v2"
                    logger.info("Automatically switched to large-v2 for memory efficiency")
        
        self.whisperx_model = None
        self.alignment_model = None
        self.alignment_metadata = None
        
        logger.info(f"🧬 BioTTS Transcription Creator initialized")
        logger.info(f"   🔬 Model: {self.model_size}")
        logger.info(f"   💾 Device: {self.device}")
        logger.info(f"   ⚡ Compute: {self.compute_type}")
    
    def install_dependencies(self):
        """Install required dependencies"""
        logger.info("📦 Checking dependencies...")
        
        required_packages = [
            "whisperx",
            "torch", 
            "torchaudio",
            "faster-whisper",
            "transformers",
            "accelerate"
        ]
        
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                logger.info(f"   ✅ {package}")
            except ImportError:
                logger.info(f"   📥 Installing {package}...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package
                ])
        
        logger.info("✅ All dependencies ready")
    
    def preprocess_audio(self, 
                        audio_path: str, 
                        target_sr: int = 16000,
                        normalize: bool = True,
                        remove_silence: bool = True) -> Tuple[torch.Tensor, int]:
        """
        Preprocess audio for optimal transcription quality
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate for WhisperX
            normalize: Normalize audio amplitude
            remove_silence: Remove leading/trailing silence
            
        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        import torch
        import torchaudio
        
        logger.info(f"🎵 Preprocessing audio: {audio_path}")
        
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
            logger.info("   🔄 Converted stereo to mono")
        
        # Resample if needed
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            audio = resampler(audio)
            sr = target_sr
            logger.info(f"   🔄 Resampled to {target_sr}Hz")
        
        # Normalize amplitude
        if normalize:
            audio = audio / torch.max(torch.abs(audio))
            logger.info("   🔄 Normalized amplitude")
        
        # Remove silence (simple threshold-based)
        if remove_silence:
            threshold = 0.01  # Adjust based on your audio
            non_silent = torch.abs(audio) > threshold
            if torch.any(non_silent):
                # Find first and last non-silent samples
                non_silent_indices = torch.where(non_silent[0])[0]
                start_idx = non_silent_indices[0].item()
                end_idx = non_silent_indices[-1].item()
                
                # Add small padding
                padding = int(0.1 * sr)  # 0.1 second padding
                start_idx = max(0, start_idx - padding)
                end_idx = min(audio.shape[1], end_idx + padding)
                
                audio = audio[:, start_idx:end_idx]
                logger.info(f"   ✂️  Trimmed silence: {start_idx/sr:.2f}s to {end_idx/sr:.2f}s")
        
        logger.info(f"   ✅ Audio preprocessed: {audio.shape[1]/sr:.2f}s duration")
        return audio.squeeze(0), sr  # Remove channel dimension
    
    def load_whisperx_model(self, language: str = "pl"):
        """Load WhisperX model and alignment model"""
        logger.info(f"🔬 Loading WhisperX model: {self.model_size}")
        
        try:
            import whisperx
        except ImportError:
            logger.error("WhisperX not installed. Install with: pip install whisperx")
            raise
        
        # Load main transcription model
        self.whisperx_model = whisperx.load_model(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            language=language
        )
        logger.info("   ✅ Transcription model loaded")
        
        # Load alignment model
        try:
            self.alignment_model, self.alignment_metadata = whisperx.load_align_model(
                language_code=language,
                device=self.device
            )
            logger.info("   ✅ Alignment model loaded")
        except Exception as e:
            logger.warning(f"Could not load alignment model: {e}")
            logger.warning("Proceeding without forced alignment")
            self.alignment_model = None
    
    def transcribe_with_whisperx(self, 
                                audio: torch.Tensor,
                                language: str = "pl",
                                batch_size: int = 16,
                                chunk_length: int = 30) -> Dict:
        """
        Transcribe audio using WhisperX
        
        Args:
            audio: Audio tensor
            language: Language code
            batch_size: Batch size for processing
            chunk_length: Chunk length in seconds
            
        Returns:
            Transcription result dictionary
        """
        logger.info("🧬 Running WhisperX transcription...")
        
        # Convert tensor to numpy for WhisperX
        if isinstance(audio, torch.Tensor):
            audio_np = audio.numpy()
        else:
            audio_np = np.array(audio)
        
        # Transcribe - try different API versions
        try:
            # Try newer API first
            result = self.whisperx_model.transcribe(
                audio_np,
                batch_size=batch_size,
                language=language
            )
        except TypeError as e:
            if "chunk_length" in str(e):
                logger.info("   🔄 Trying older API without chunk_length...")
                # Try older API
                result = self.whisperx_model.transcribe(
                    audio_np,
                    batch_size=batch_size
                )
            else:
                # Try minimal API
                logger.info("   🔄 Trying minimal API...")
                result = self.whisperx_model.transcribe(audio_np)
        
        logger.info(f"   ✅ Transcribed {len(result.get('segments', []))} segments")
        
        # Add word-level timestamps if alignment model available
        if self.alignment_model is not None:
            logger.info("🔗 Running forced alignment...")
            import whisperx
            result = whisperx.align(
                result["segments"],
                self.alignment_model,
                self.alignment_metadata,
                audio_np,
                device=self.device,
                return_char_alignments=True  # Character-level for TTS
            )
            logger.info("   ✅ Forced alignment completed")
        
        return result
    
    def create_character_alignment(self, result: Dict) -> List[Dict]:
        """
        Create character-level alignment for TTS training
        
        Args:
            result: WhisperX result with alignment
            
        Returns:
            List of character-level alignment dictionaries
        """
        logger.info("📝 Creating character-level alignment...")
        
        char_alignments = []
        
        for segment in result.get("segments", []):
            segment_start = segment.get("start", 0.0)
            segment_end = segment.get("end", 0.0)
            segment_text = segment.get("text", "").strip()
            
            # If we have word-level alignment, use it
            if "words" in segment:
                for word_info in segment["words"]:
                    word_start = word_info.get("start", segment_start)
                    word_end = word_info.get("end", segment_end)
                    word_text = word_info.get("word", "")
                    
                    # Character-level timing within word
                    if "chars" in word_info:
                        # Use character-level alignment if available
                        for char_info in word_info["chars"]:
                            char_alignments.append({
                                "char": char_info.get("char", ""),
                                "start": char_info.get("start", word_start),
                                "end": char_info.get("end", word_end),
                                "confidence": char_info.get("score", 1.0),
                                "word": word_text,
                                "segment_id": len(char_alignments)
                            })
                    else:
                        # Interpolate character timing within word
                        if word_text:
                            char_duration = (word_end - word_start) / len(word_text)
                            for i, char in enumerate(word_text):
                                char_start = word_start + i * char_duration
                                char_end = word_start + (i + 1) * char_duration
                                
                                char_alignments.append({
                                    "char": char,
                                    "start": char_start,
                                    "end": char_end,
                                    "confidence": word_info.get("score", 1.0),
                                    "word": word_text,
                                    "segment_id": len(char_alignments)
                                })
            else:
                # No word-level alignment, interpolate across segment
                if segment_text:
                    char_duration = (segment_end - segment_start) / len(segment_text)
                    for i, char in enumerate(segment_text):
                        char_start = segment_start + i * char_duration
                        char_end = segment_start + (i + 1) * char_duration
                        
                        char_alignments.append({
                            "char": char,
                            "start": char_start,
                            "end": char_end,
                            "confidence": segment.get("score", 1.0),
                            "word": "",
                            "segment_id": len(char_alignments)
                        })
        
        logger.info(f"   ✅ Created {len(char_alignments)} character alignments")
        return char_alignments
    
    def validate_transcription(self, 
                             result: Dict, 
                             char_alignments: List[Dict],
                             min_confidence: float = 0.7) -> Dict:
        """
        Validate transcription quality and provide statistics
        
        Args:
            result: WhisperX result
            char_alignments: Character-level alignments  
            min_confidence: Minimum confidence threshold
            
        Returns:
            Validation report
        """
        logger.info("🔍 Validating transcription quality...")
        
        report = {
            "total_segments": len(result.get("segments", [])),
            "total_characters": len(char_alignments),
            "total_duration": 0.0,
            "average_confidence": 0.0,
            "low_confidence_chars": 0,
            "quality_issues": []
        }
        
        if result.get("segments"):
            # Calculate total duration
            first_start = result["segments"][0].get("start", 0.0)
            last_end = result["segments"][-1].get("end", 0.0)
            report["total_duration"] = last_end - first_start
            
            # Calculate confidence statistics
            confidences = []
            for segment in result["segments"]:
                if "words" in segment:
                    for word in segment["words"]:
                        if "score" in word:
                            confidences.append(word["score"])
                elif "score" in segment:
                    confidences.append(segment["score"])
            
            if confidences:
                report["average_confidence"] = np.mean(confidences)
                report["low_confidence_chars"] = sum(1 for c in confidences if c < min_confidence)
        
        # Check for potential issues
        if report["average_confidence"] < 0.8:
            report["quality_issues"].append("Low average confidence")
        
        if report["low_confidence_chars"] > len(char_alignments) * 0.1:
            report["quality_issues"].append("High number of low-confidence characters")
        
        if report["total_duration"] == 0:
            report["quality_issues"].append("No timing information")
        
        # Log results
        logger.info(f"   📊 Validation Report:")
        logger.info(f"      Segments: {report['total_segments']}")
        logger.info(f"      Characters: {report['total_characters']}")
        logger.info(f"      Duration: {report['total_duration']:.2f}s")
        logger.info(f"      Avg Confidence: {report['average_confidence']:.3f}")
        logger.info(f"      Low Confidence: {report['low_confidence_chars']}")
        
        if report["quality_issues"]:
            logger.warning(f"   ⚠️  Quality Issues: {', '.join(report['quality_issues'])}")
        else:
            logger.info("   ✅ Quality validation passed")
        
        return report
    
    def create_biotts_format(self, 
                           result: Dict,
                           char_alignments: List[Dict],
                           validation_report: Dict,
                           audio_path: str) -> Dict:
        """
        Create BioTTS compatible transcription format
        
        Args:
            result: WhisperX transcription result
            char_alignments: Character-level alignments
            validation_report: Quality validation report
            audio_path: Original audio file path
            
        Returns:
            BioTTS formatted transcription
        """
        logger.info("🧬 Creating BioTTS compatible format...")
        
        # Extract full text
        full_text = " ".join([seg.get("text", "").strip() for seg in result.get("segments", [])])
        
        # Create BioTTS format
        biotts_format = {
            "metadata": {
                "audio_file": str(Path(audio_path).name),
                "audio_path": str(audio_path),
                "transcription_model": self.model_size,
                "language": result.get("language", "unknown"),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": validation_report.get("total_duration", 0.0),
                "total_characters": len(char_alignments),
                "average_confidence": validation_report.get("average_confidence", 0.0),
                "quality_issues": validation_report.get("quality_issues", [])
            },
            
            "text": {
                "full_text": full_text,
                "normalized_text": full_text,  # Could add normalization here
                "character_count": len(full_text),
                "word_count": len(full_text.split())
            },
            
            "segments": [
                {
                    "id": i,
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", 0.0),
                    "text": seg.get("text", "").strip(),
                    "confidence": seg.get("score", 1.0),
                    "words": seg.get("words", [])
                }
                for i, seg in enumerate(result.get("segments", []))
            ],
            
            "character_alignments": char_alignments,
            
            "training_fragments": self._create_training_fragments(char_alignments),
            
            "biotts_compatibility": {
                "format_version": "1.0",
                "character_level": True,
                "forced_alignment": self.alignment_model is not None,
                "suitable_for_training": len(validation_report.get("quality_issues", [])) == 0
            }
        }
        
        logger.info("   ✅ BioTTS format created")
        return biotts_format
    
    def _create_training_fragments(self, char_alignments: List[Dict]) -> List[Dict]:
        """Create training fragments of different lengths for curriculum learning"""
        fragments = []
        
        # Character-level fragments (for debugging)
        for i, char_align in enumerate(char_alignments[:100]):  # Limit for space
            fragments.append({
                "type": "character",
                "id": f"char_{i}",
                "start": char_align["start"],
                "end": char_align["end"],
                "text": char_align["char"],
                "duration": char_align["end"] - char_align["start"]
            })
        
        # Word-level fragments (for early training)
        current_word = ""
        word_start = 0.0
        word_chars = []
        
        for char_align in char_alignments:
            if char_align["char"] == " " and current_word:
                # End of word
                fragments.append({
                    "type": "word",
                    "id": f"word_{len([f for f in fragments if f['type'] == 'word'])}",
                    "start": word_start,
                    "end": char_align["start"],
                    "text": current_word,
                    "duration": char_align["start"] - word_start,
                    "character_count": len(current_word)
                })
                current_word = ""
                word_chars = []
            elif char_align["char"] != " ":
                if not current_word:
                    word_start = char_align["start"]
                current_word += char_align["char"]
                word_chars.append(char_align)
        
        # Sentence-level fragments (for intermediate training)
        sentence_chars = []
        sentence_start = None
        
        for char_align in char_alignments:
            if sentence_start is None:
                sentence_start = char_align["start"]
            
            sentence_chars.append(char_align)
            
            # End sentence on punctuation
            if char_align["char"] in ".!?":
                sentence_text = "".join([c["char"] for c in sentence_chars])
                fragments.append({
                    "type": "sentence",
                    "id": f"sentence_{len([f for f in fragments if f['type'] == 'sentence'])}",
                    "start": sentence_start,
                    "end": char_align["end"],
                    "text": sentence_text,
                    "duration": char_align["end"] - sentence_start,
                    "character_count": len(sentence_text)
                })
                sentence_chars = []
                sentence_start = None
        
        logger.info(f"   📊 Created training fragments:")
        logger.info(f"      Characters: {len([f for f in fragments if f['type'] == 'character'])}")
        logger.info(f"      Words: {len([f for f in fragments if f['type'] == 'word'])}")
        logger.info(f"      Sentences: {len([f for f in fragments if f['type'] == 'sentence'])}")
        
        return fragments
    
    def process_audio_file(self, 
                          audio_path: str,
                          output_path: str,
                          language: str = "pl",
                          batch_size: int = 16,
                          preprocess: bool = True) -> Dict:
        """
        Complete pipeline to process audio file and create transcription
        
        Args:
            audio_path: Path to input audio file
            output_path: Path to output JSON file
            language: Language code
            batch_size: Processing batch size
            preprocess: Whether to preprocess audio
            
        Returns:
            BioTTS formatted transcription
        """
        logger.info(f"🧬 Starting BioTTS transcription pipeline")
        logger.info(f"   📁 Input: {audio_path}")
        logger.info(f"   📁 Output: {output_path}")
        logger.info(f"   🌍 Language: {language}")
        
        # Check input file
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Preprocess audio
        if preprocess:
            audio, sr = self.preprocess_audio(audio_path)
        else:
            import torch
            import torchaudio
            audio, sr = torchaudio.load(audio_path)
            audio = audio.mean(dim=0)  # Convert to mono
        
        # Load models
        self.load_whisperx_model(language)
        
        # Transcribe
        result = self.transcribe_with_whisperx(
            audio, 
            language=language, 
            batch_size=batch_size
        )
        
        # Create character alignments
        char_alignments = self.create_character_alignment(result)
        
        # Validate quality
        validation_report = self.validate_transcription(result, char_alignments)
        
        # Create BioTTS format
        biotts_transcription = self.create_biotts_format(
            result, 
            char_alignments, 
            validation_report, 
            audio_path
        )
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(biotts_transcription, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Transcription saved to: {output_path}")
        logger.info(f"🧬 BioTTS transcription pipeline completed successfully!")
        
        return biotts_transcription

class BatchAudioProcessor:
    """
    Batch processor for audio files before BioTTS transcription
    """
    
    def __init__(self, 
                 target_duration: float = 11 * 60,  # 11 minutes in seconds
                 target_sr: int = 22050,
                 normalize_loudness: bool = True,
                 target_lufs: float = -23.0):
        """
        Initialize batch processor
        
        Args:
            target_duration: Target duration in seconds (11 minutes = 660s)
            target_sr: Target sample rate
            normalize_loudness: Whether to normalize loudness
            target_lufs: Target LUFS for loudness normalization
        """
        self.target_duration = target_duration
        self.target_sr = target_sr
        self.normalize_loudness = normalize_loudness
        self.target_lufs = target_lufs
        
        logger.info(f"🎵 Batch Audio Processor initialized")
        logger.info(f"   ⏱️  Target duration: {target_duration/60:.1f} minutes")
        logger.info(f"   🔊 Target sample rate: {target_sr}Hz")
        logger.info(f"   📊 Normalize loudness: {normalize_loudness}")
        if normalize_loudness:
            logger.info(f"   🎯 Target LUFS: {target_lufs}")
    
    def install_dependencies(self):
        """Install required dependencies"""
        logger.info("📦 Checking audio processing dependencies...")
        
        required_packages = [
            "librosa",
            "soundfile", 
            "numpy",
            "tqdm",
            "pyloudnorm"  # For loudness normalization
        ]
        
        for package in required_packages:
            try:
                if package == "pyloudnorm":
                    import pyloudnorm
                else:
                    __import__(package)
                logger.info(f"   ✅ {package}")
            except ImportError:
                logger.info(f"   📥 Installing {package}...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package
                ])
        
        logger.info("✅ All audio dependencies ready")
    
    def find_audio_files(self, input_dir: str) -> List[Path]:
        """
        Find all audio files in input directory
        
        Args:
            input_dir: Input directory path
            
        Returns:
            List of audio file paths
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Common audio extensions
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.wma'}
        
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(input_path.glob(f"*{ext}"))
            audio_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        audio_files.sort()  # Sort for consistent processing order
        
        logger.info(f"📁 Found {len(audio_files)} audio files in {input_dir}")
        for i, file_path in enumerate(audio_files, 1):
            logger.info(f"   {i}. {file_path.name}")
        
        return audio_files
    
    def trim_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Trim audio to target duration
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Trimmed audio array
        """
        current_duration = len(audio) / sr
        target_samples = int(self.target_duration * sr)
        
        if len(audio) > target_samples:
            # Trim from the beginning (skip potential silence)
            # But keep some variety - trim from different starting points
            max_start_offset = min(len(audio) - target_samples, int(30 * sr))  # Max 30s offset
            start_offset = np.random.randint(0, max_start_offset + 1) if max_start_offset > 0 else 0
            
            audio = audio[start_offset:start_offset + target_samples]
            logger.info(f"   ✂️  Trimmed from {current_duration:.1f}s to {len(audio)/sr:.1f}s (offset: {start_offset/sr:.1f}s)")
        else:
            logger.info(f"   ℹ️  Audio duration ({current_duration:.1f}s) is shorter than target ({self.target_duration/60:.1f}min)")
        
        return audio
    
    def normalize_loudness_func(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Normalize audio loudness to target LUFS
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Loudness-normalized audio
        """
        try:
            import pyloudnorm as pyln
            
            # Measure loudness
            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(audio)
            
            if np.isfinite(loudness):
                # Normalize to target LUFS
                normalized_audio = pyln.normalize.loudness(audio, loudness, self.target_lufs)
                logger.info(f"   🔊 Normalized loudness: {loudness:.1f} LUFS → {self.target_lufs:.1f} LUFS")
                return normalized_audio
            else:
                logger.warning("   ⚠️  Could not measure loudness, using RMS normalization")
                return self.normalize_rms(audio)
                
        except ImportError:
            logger.warning("   ⚠️  pyloudnorm not available, using RMS normalization")
            return self.normalize_rms(audio)
        except Exception as e:
            logger.warning(f"   ⚠️  Loudness normalization failed: {e}, using RMS normalization")
            return self.normalize_rms(audio)
    
    def normalize_rms(self, audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
        """
        Fallback RMS normalization
        
        Args:
            audio: Audio array
            target_rms: Target RMS level
            
        Returns:
            RMS-normalized audio
        """
        current_rms = np.sqrt(np.mean(audio**2))
        if current_rms > 0:
            scaling_factor = target_rms / current_rms
            # Prevent clipping
            scaling_factor = min(scaling_factor, 0.95 / np.max(np.abs(audio)))
            normalized_audio = audio * scaling_factor
            logger.info(f"   🔊 RMS normalization: {current_rms:.4f} → {np.sqrt(np.mean(normalized_audio**2)):.4f}")
            return normalized_audio
        else:
            logger.warning("   ⚠️  Silent audio, skipping normalization")
            return audio
    
    def process_single_file(self, 
                          input_path: Path, 
                          output_dir: Path, 
                          sample_name: str,
                          create_preview: bool = True,
                          preview_dir: Path = None,
                          preview_duration: float = 30.0) -> Tuple[Path, bool, Path]:
        """
        Process a single audio file
        
        Args:
            input_path: Input audio file path
            output_dir: Output directory
            sample_name: Output sample name (e.g., "sample1")
            create_preview: Whether to create preview sample
            preview_dir: Directory for preview samples
            preview_duration: Duration of preview in seconds
            
        Returns:
            Tuple of (processed_file_path, success, preview_path)
        """
        logger.info(f"🎵 Processing: {input_path.name} → {sample_name}")
        
        try:
            # Load audio
            logger.info("   📂 Loading audio...")
            audio, sr = librosa.load(str(input_path), sr=self.target_sr, mono=True)
            logger.info(f"   ✅ Loaded: {len(audio)/sr:.1f}s at {sr}Hz")
            
            # Create preview BEFORE processing (to show original quality)
            preview_path = None
            if create_preview and preview_dir is not None:
                original_preview_path = self.create_preview_sample(
                    audio, sr, preview_dir, f"{sample_name}_original", 
                    preview_duration, start_offset=30.0
                )
            
            # Trim to target duration
            audio = self.trim_audio(audio, sr)
            
            # Normalize loudness
            if self.normalize_loudness:
                audio = self.normalize_loudness_func(audio, sr)
            
            # Ensure no clipping
            if np.max(np.abs(audio)) > 0.99:
                audio = audio * (0.95 / np.max(np.abs(audio)))
                logger.info("   🔧 Applied anti-clipping scaling")
            
            # Create preview AFTER processing (to show processed quality)
            if create_preview and preview_dir is not None:
                preview_path = self.create_preview_sample(
                    audio, sr, preview_dir, sample_name, 
                    preview_duration, start_offset=60.0
                )
            
            # Save processed audio
            output_path = output_dir / f"{sample_name}.wav"
            sf.write(str(output_path), audio, sr, subtype='PCM_16')
            logger.info(f"   💾 Saved processed audio: {output_path}")
            
            return output_path, True, preview_path
            
        except Exception as e:
            logger.error(f"   ❌ Failed to process {input_path.name}: {e}")
            return input_path, False, None
    
    def create_preview_sample(self, 
                            audio: np.ndarray, 
                            sr: int, 
                            output_dir: Path, 
                            sample_name: str,
                            preview_duration: float = 30.0,
                            start_offset: float = 60.0) -> Path:
        """
        Create a preview sample for quality checking
        
        Args:
            audio: Full audio array
            sr: Sample rate
            output_dir: Output directory for preview
            sample_name: Sample name
            preview_duration: Duration of preview in seconds
            start_offset: Start offset from beginning in seconds
            
        Returns:
            Path to preview file
        """
        preview_samples = int(preview_duration * sr)
        start_samples = int(start_offset * sr)
        
        # Extract preview segment
        if len(audio) > start_samples + preview_samples:
            preview_audio = audio[start_samples:start_samples + preview_samples]
        elif len(audio) > preview_samples:
            # If not enough for offset, take from beginning
            preview_audio = audio[:preview_samples]
        else:
            # If audio is shorter than preview duration, use all
            preview_audio = audio
        
        # Save preview
        preview_path = output_dir / f"{sample_name}_preview.wav"
        sf.write(str(preview_path), preview_audio, sr, subtype='PCM_16')
        
        logger.info(f"   🎧 Created preview: {preview_path} ({len(preview_audio)/sr:.1f}s)")
        return preview_path

    def process_batch(self, 
                    input_dir: str, 
                    output_dir: str,
                    transcription_model: str = "large-v3",
                    language: str = "pl",
                    create_previews: bool = True,
                    preview_duration: float = 30.0,
                    device: str = "auto") -> dict:  # <- Dodaj device parametr
        """
        Process all audio files in batch
        
        Args:
            input_dir: Input directory with audio files
            output_dir: Output directory for processed files and transcriptions
            transcription_model: WhisperX model size
            language: Language code for transcription
            create_previews: Whether to create preview samples
            preview_duration: Duration of preview samples in seconds
            
        Returns:
            Processing report dictionary
        """
        logger.info(f"🚀 Starting batch processing")
        logger.info(f"   📁 Input: {input_dir}")
        logger.info(f"   📁 Output: {output_dir}")
        
        # Create output directories
        output_path = Path(output_dir)
        audio_output_dir = output_path / "processed_audio"
        transcription_output_dir = output_path / "transcriptions"
        preview_output_dir = output_path / "preview_samples" if create_previews else None
        
        audio_output_dir.mkdir(parents=True, exist_ok=True)
        transcription_output_dir.mkdir(parents=True, exist_ok=True)
        if create_previews:
            preview_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find audio files
        audio_files = self.find_audio_files(input_dir)
        
        if not audio_files:
            raise ValueError(f"No audio files found in {input_dir}")
        
        # Limit to 6 files as requested
        if len(audio_files) > 6:
            logger.info(f"   ℹ️  Limiting to first 6 files (found {len(audio_files)})")
            audio_files = audio_files[:6]
        
        # Initialize transcription creator
        # Initialize transcription creator
        # Initialize transcription creator  
        transcription_creator = BioTTSTranscriptionCreator(
            model_size=transcription_model,
            device=device,  # <- Zmień z args.device na device
            compute_type="float16"
        )
        
        # Process each file
        report = {
            "processed_files": [],
            "successful_audio": 0,
            "successful_transcriptions": 0,
            "failed_files": [],
            "total_files": len(audio_files)
        }
        
        for i, audio_file in enumerate(tqdm(audio_files, desc="Processing files")):
            sample_name = f"sample{i+1}"
            
            # Process audio
            processed_path, audio_success, preview_path = self.process_single_file(
                audio_file, 
                audio_output_dir, 
                sample_name,
                create_preview=create_previews,
                preview_dir=preview_output_dir,
                preview_duration=preview_duration
            )
            
            file_report = {
                "original_file": str(audio_file),
                "sample_name": sample_name,
                "audio_processing": audio_success,
                "transcription": False,
                "processed_audio_path": str(processed_path) if audio_success else None,
                "preview_path": str(preview_path) if preview_path else None,
                "transcription_path": None
            }
            
            if audio_success:
                report["successful_audio"] += 1
                
                # Create transcription
                try:
                    logger.info(f"🧬 Creating transcription for {sample_name}...")
                    
                    transcription_path = transcription_output_dir / f"{sample_name}_transcription.json"
                    
                    transcription_result = transcription_creator.process_audio_file(
                        audio_path=str(processed_path),
                        output_path=str(transcription_path),
                        language=language,
                        batch_size=16,
                        preprocess=True
                    )
                    
                    file_report["transcription"] = True
                    file_report["transcription_path"] = str(transcription_path)
                    report["successful_transcriptions"] += 1
                    
                    logger.info(f"   ✅ Transcription completed: {transcription_path}")
                    
                except Exception as e:
                    logger.error(f"   ❌ Transcription failed for {sample_name}: {e}")
                    file_report["transcription_error"] = str(e)
            else:
                report["failed_files"].append(str(audio_file))
            
            report["processed_files"].append(file_report)
        
        # Save processing report
        report_path = output_path / "processing_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Print summary
        logger.info(f"\n🎯 BATCH PROCESSING COMPLETED")
        logger.info(f"   📊 Total files: {report['total_files']}")
        logger.info(f"   ✅ Audio processed: {report['successful_audio']}")
        logger.info(f"   🧬 Transcriptions: {report['successful_transcriptions']}")
        logger.info(f"   ❌ Failed: {len(report['failed_files'])}")
        logger.info(f"   📋 Report saved: {report_path}")
        
        if report["failed_files"]:
            logger.warning(f"   ⚠️  Failed files: {', '.join(report['failed_files'])}")
        
        return report


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Batch process audio files for BioTTS transcription"
    )
    
    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        default="raw_data",
        help="Input directory with audio files (default: raw_data)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="whisper_transcription",
        help="Output directory for processed files (default: whisper_transcription)"
    )
    
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=11.0,
        help="Target duration in minutes (default: 11.0)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="large-v3",
        choices=["large-v3", "large-v2", "medium", "small", "base", "tiny"],
        help="WhisperX model size (default: large-v3)"
    )
    
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="pl",
        help="Language code for transcription (default: pl)"
    )
    
    parser.add_argument(
        "--sample-rate", "-sr",
        type=int,
        default=22050,
        help="Target sample rate (default: 22050)"
    )
    
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip loudness normalization"
    )
    
    parser.add_argument(
        "--target-lufs",
        type=float,
        default=-23.0,
        help="Target LUFS for loudness normalization (default: -23.0)"
    )
    
    parser.add_argument(
        "--no-previews",
        action="store_true",
        help="Skip creating preview samples for quality checking"
    )
    
    parser.add_argument(
        "--preview-duration",
        type=float,
        default=30.0,
        help="Duration of preview samples in seconds (default: 30.0)"
    )
    
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install required dependencies"
    )

    parser.add_argument(
        "--device", 
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for transcription (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Create processor
    processor = BatchAudioProcessor(
        target_duration=args.duration * 60,  # Convert minutes to seconds
        target_sr=args.sample_rate,
        normalize_loudness=not args.no_normalize,
        target_lufs=args.target_lufs
    )
    
    # Install dependencies if requested
    if args.install_deps:
        processor.install_dependencies()
    
    # Process batch
    try:
        report = processor.process_batch(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            transcription_model=args.model,
            language=args.language,
            create_previews=not args.no_previews,
            preview_duration=args.preview_duration,
            device=args.device  # <- Dodaj tę linię
        )
        
        print(f"\n🎯 SUCCESS!")
        print(f"Processed {report['successful_audio']}/{report['total_files']} audio files")
        print(f"Created {report['successful_transcriptions']} transcriptions")
        print(f"Results saved in: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Batch processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()