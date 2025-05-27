"""
Stateful-Aware Nucleotide Tokenizer
====================================
Optimized for continuous state processing in audiobook generation
Handles special tokens intelligently for stateful vs stateless modes
"""

import os
import json
import re
import unicodedata
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from enum import Enum

class ProcessingMode(Enum):
    STATELESS = "stateless"  # Traditional chunk-by-chunk with boundaries
    STATEFUL = "stateful"    # Continuous processing across chunks
    AUDIOBOOK = "audiobook"  # Long-form continuous narrative

class StatefulNucleotideTokenizer:
    """
    Enhanced tokenizer with stateful processing awareness
    """
    def __init__(self, vocab_path: Optional[str] = None):
        self.normalize_orthography = False
        self.case_sensitive = True
        self.include_digraphs = True
        self.training_mode = False
        
        # Processing mode - affects special token usage
        self.processing_mode = ProcessingMode.STATELESS
        
        if vocab_path is None:
            vocab_path = "nucleotide_vocab.json"
        self.vocab_path = Path(vocab_path)

        self.digraph_map = {
            "sz": "<sz>", "Sz": "<Sz>", "SZ": "<SZ>",
            "cz": "<cz>", "Cz": "<Cz>", "CZ": "<CZ>",
            "dz": "<dz>", "Dz": "<Dz>", "DZ": "<DZ>",
            "dź": "<dź>", "Dź": "<Dź>", "DŹ": "<DŹ>",
            "dż": "<dż>", "Dż": "<Dż>", "DŻ": "<DŻ>",
            "ch": "<ch>", "Ch": "<Ch>", "CH": "<CH>",
            "rz": "<rz>", "Rz": "<Rz>", "RZ": "<RZ>"
        }

        if self.vocab_path.exists():
            self._load_vocab()
        else:
            self._create_vocab()
            self._save_vocab()
    
    def set_processing_mode(self, mode: Union[ProcessingMode, str]):
        """Set processing mode for appropriate special token handling"""
        if isinstance(mode, str):
            mode = ProcessingMode(mode)
        
        self.processing_mode = mode
        print(f"🔄 Set processing mode: {mode.value}")
        
        if mode == ProcessingMode.STATEFUL:
            print("   📝 Stateful mode: Minimal special tokens for continuous flow")
        elif mode == ProcessingMode.AUDIOBOOK:
            print("   📚 Audiobook mode: Chapter boundaries and prosodic markers")
        else:
            print("   🔄 Stateless mode: Standard special token usage")

    def _create_vocab(self):
        """Create vocabulary with enhanced special tokens"""
        vocab = {}
        idx = 0
        
        # Space first (most common)
        vocab[" "] = idx
        idx += 1
        
        # Essential special tokens
        essential_tokens = [
            "<pad>",      # Padding
            "<unk>",      # Unknown
            "<s>",        # Sequence start (for stateless)
            "</s>",       # Sequence end (for stateless)
        ]
        for token in essential_tokens:
            vocab[token] = idx
            idx += 1
        
        # Prosodic and narrative special tokens (important for audiobooks)
        prosodic_tokens = [
            "<pause>",       # Short pause
            "<long_pause>",  # Long pause
            "<breath>",      # Breath mark
            "<emphasis>",    # Emphasis
            "<whisper>",     # Whisper
            "<loud>",        # Loud speech
            "<fast>",        # Fast speech
            "<slow>",        # Slow speech
            # Chapter/section markers for audiobooks
            "<chapter>",     # Chapter boundary
            "<section>",     # Section boundary
            "<paragraph>",   # Paragraph boundary
            "<sentence>",    # Sentence boundary (soft)
        ]
        for token in prosodic_tokens:
            vocab[token] = idx
            idx += 1
        
        # Standard punctuation
        punctuation = ".,!?:;-—\"'()[]…"
        for char in punctuation:
            vocab[char] = idx
            idx += 1
        
        # Digits
        for digit in "0123456789":
            vocab[digit] = idx
            idx += 1
        
        # Polish characters (lowercase)
        lowercase = "aąbcćdeęfghijklłmnńoópqrsśtuvwxyzźż"
        for char in lowercase:
            vocab[char] = idx
            idx += 1
        
        # Polish characters (uppercase)
        uppercase = "AĄBCĆDEĘFGHIJKLŁMNŃOÓPQRSŚTUVWXYZŹŻ"
        for char in uppercase:
            vocab[char] = idx
            idx += 1
        
        # Digraph tokens
        digraph_tokens = list(self.digraph_map.values())
        for token in digraph_tokens:
            vocab[token] = idx
            idx += 1
        
        # Additional symbols
        symbols = "→←@#$%&*+=<>^_|~"
        for char in symbols:
            vocab[char] = idx
            idx += 1
        
        self.token2id = vocab
        self.id2token = {v: k for k, v in vocab.items()}
        print(f"🧬 Created stateful-aware vocabulary: {len(vocab)} tokens")
    
    def _load_vocab(self):
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            self.token2id = json.load(f)
        self.id2token = {v: k for k, v in self.token2id.items()}

    def _save_vocab(self):
        self.vocab_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)

    def normalize_text(self, text: str) -> str:
        """Enhanced text normalization"""
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        text = (text.replace('„', '"').replace('"', '"')
                     .replace('»', '"').replace('«', '"')
                     .replace(''', "'").replace(''', "'")
                     .replace('–', '-').replace('―', '-').replace('—', '-'))
        return text

    def text_to_nucleotides(self, text: str) -> List[int]:
        """Convert text to nucleotide tokens"""
        text = self.normalize_text(text)
        nucleotides = []
        i = 0

        while i < len(text):
            found = False
            
            # Check for special prosodic tokens
            special_tokens = [
                "<pause>", "<long_pause>", "<breath>", "<emphasis>", 
                "<whisper>", "<loud>", "<fast>", "<slow>",
                "<chapter>", "<section>", "<paragraph>", "<sentence>"
            ]
            for special in special_tokens:
                if text[i:i+len(special)] == special:
                    nucleotides.append(self.token2id.get(special, self.token2id.get("<unk>", 2)))
                    i += len(special)
                    found = True
                    break
            if found:
                continue
            
            # Check for digraphs
            for digraph in sorted(self.digraph_map.keys(), key=len, reverse=True):
                if text[i:i+len(digraph)] == digraph:
                    mapped_token = self.digraph_map[digraph]
                    nucleotides.append(self.token2id.get(mapped_token, self.token2id.get("<unk>", 2)))
                    i += len(digraph)
                    found = True
                    break
            if found:
                continue
            
            # Single character
            char = text[i]
            nucleotides.append(self.token2id.get(char, self.token2id.get("<unk>", 2)))
            i += 1
        
        return nucleotides

    def encode(self, text: str, add_special_tokens: Optional[bool] = None, chunk_position: str = "middle") -> List[int]:
        """
        Encode text with mode-aware special token handling
        
        Args:
            text: Input text
            add_special_tokens: Override automatic special token handling
            chunk_position: "start", "middle", "end" for stateful processing
        """
        nucleotides = []
        
        # Determine whether to add special tokens based on mode
        if add_special_tokens is None:
            if self.processing_mode == ProcessingMode.STATELESS:
                add_special_tokens = True
            elif self.processing_mode == ProcessingMode.STATEFUL:
                # Only add special tokens at document boundaries, not chunk boundaries
                add_special_tokens = (chunk_position in ["start", "end"])
            elif self.processing_mode == ProcessingMode.AUDIOBOOK:
                # Add chapter/section boundaries, but not chunk boundaries
                add_special_tokens = False  # Handle manually
        
        # Add start token if needed
        if add_special_tokens and chunk_position in ["start", "middle"]:
            if chunk_position == "start" or self.processing_mode == ProcessingMode.STATELESS:
                nucleotides.append(self.token2id.get("<s>", 3))
        
        # Main content tokens
        nucleotides.extend(self.text_to_nucleotides(text))
        
        # Add end token if needed
        if add_special_tokens and chunk_position in ["end", "middle"]:
            if chunk_position == "end" or self.processing_mode == ProcessingMode.STATELESS:
                nucleotides.append(self.token2id.get("</s>", 4))
        
        return nucleotides

    def encode_for_stateful(self, text: str, is_first_chunk: bool = False, is_last_chunk: bool = False) -> List[int]:
        """
        Encode specifically for stateful processing
        Only adds special tokens at document boundaries, not chunk boundaries
        """
        nucleotides = []
        
        # Add start token only for the very first chunk of a document/audiobook
        if is_first_chunk:
            nucleotides.append(self.token2id.get("<s>", 3))
        
        # Main content - no chunk boundary tokens
        nucleotides.extend(self.text_to_nucleotides(text))
        
        # Add end token only for the very last chunk of a document/audiobook
        if is_last_chunk:
            nucleotides.append(self.token2id.get("</s>", 4))
        
        return nucleotides

    def encode_audiobook_chunk(self, text: str, chunk_info: Dict) -> List[int]:
        """
        Encode for audiobook with chapter/section awareness
        
        Args:
            text: Chunk text
            chunk_info: {
                'is_first_chunk': bool,
                'is_last_chunk': bool, 
                'is_chapter_start': bool,
                'is_chapter_end': bool,
                'is_section_start': bool,
                'is_section_end': bool
            }
        """
        nucleotides = []
        
        # Document start
        if chunk_info.get('is_first_chunk', False):
            nucleotides.append(self.token2id.get("<s>", 3))
        
        # Chapter start
        if chunk_info.get('is_chapter_start', False):
            nucleotides.append(self.token2id.get("<chapter>", self.token2id.get("<unk>", 2)))
        
        # Section start  
        if chunk_info.get('is_section_start', False):
            nucleotides.append(self.token2id.get("<section>", self.token2id.get("<unk>", 2)))
        
        # Main content
        nucleotides.extend(self.text_to_nucleotides(text))
        
        # Section end
        if chunk_info.get('is_section_end', False):
            nucleotides.append(self.token2id.get("</section>", self.token2id.get("<unk>", 2)))
        
        # Chapter end
        if chunk_info.get('is_chapter_end', False):
            nucleotides.append(self.token2id.get("</chapter>", self.token2id.get("<unk>", 2)))
        
        # Document end
        if chunk_info.get('is_last_chunk', False):
            nucleotides.append(self.token2id.get("</s>", 4))
        
        return nucleotides

    def nucleotides_to_text(self, nucleotides: List[int]) -> str:
        """Convert nucleotide tokens back to text"""
        tokens = [self.id2token.get(nuc_id, "<unk>") for nuc_id in nucleotides]
        text = ""
        for token in tokens:
            if token in ["<pad>", "<unk>", "<s>", "</s>"]:
                continue
            elif token in self.digraph_map.values():
                # Find original digraph
                text += [k for k, v in self.digraph_map.items() if v == token][0]
            elif token.startswith("<") and token.endswith(">"):
                # Special tokens - keep as is for now
                text += token
            else:
                text += token
        return text

    def decode(self, nucleotides: List[int], skip_special_tokens: bool = True) -> str:
        """Decode nucleotides back to text"""
        if skip_special_tokens:
            special_ids = {
                self.token2id.get("<s>", 3),
                self.token2id.get("</s>", 4),
                self.token2id.get("<pad>", 1)
            }
            nucleotides = [nuc for nuc in nucleotides if nuc not in special_ids]
        return self.nucleotides_to_text(nucleotides)

    def get_vocab_size(self) -> int:
        return len(self.token2id)

    def get_vocab_dict(self) -> Dict[str, int]:
        return self.token2id.copy()


def demo_stateful_tokenizer():
    """Demonstrate stateful tokenizer capabilities"""
    print("🧬 STATEFUL-AWARE NUCLEOTIDE TOKENIZER DEMO")
    print("=" * 60)
    
    tokenizer = StatefulNucleotideTokenizer()
    print(f"📊 Vocab size: {tokenizer.get_vocab_size()}")
    
    test_text = "Cześć! Jak się masz? To jest test stateful tokenizacji."
    
    print(f"\n📝 Test text: '{test_text}'")
    print(f"   Normalized: '{tokenizer.normalize_text(test_text)}'")
    
    # Test different modes
    modes = [ProcessingMode.STATELESS, ProcessingMode.STATEFUL, ProcessingMode.AUDIOBOOK]
    
    for mode in modes:
        print(f"\n🔄 Mode: {mode.value}")
        tokenizer.set_processing_mode(mode)
        
        if mode == ProcessingMode.STATELESS:
            # Traditional encoding
            encoded = tokenizer.encode(test_text)
            decoded = tokenizer.decode(encoded)
            print(f"   Encoded: {encoded}")
            print(f"   Length: {len(encoded)} tokens")
            print(f"   Decoded: '{decoded}'")
            
        elif mode == ProcessingMode.STATEFUL:
            # Stateful encoding - simulate 3 chunks
            chunks = [
                ("Cześć! Jak się", True, False),   # First chunk
                (" masz? To jest", False, False),  # Middle chunk  
                (" test stateful tokenizacji.", False, True)  # Last chunk
            ]
            
            all_encoded = []
            for chunk_text, is_first, is_last in chunks:
                encoded = tokenizer.encode_for_stateful(chunk_text, is_first, is_last)
                all_encoded.extend(encoded)
                print(f"   Chunk '{chunk_text}': {encoded} ({len(encoded)} tokens)")
            
            decoded = tokenizer.decode(all_encoded)
            print(f"   Combined: {all_encoded}")
            print(f"   Total length: {len(all_encoded)} tokens")
            print(f"   Decoded: '{decoded}'")
            
        elif mode == ProcessingMode.AUDIOBOOK:
            # Audiobook encoding with chapter info
            chunk_info = {
                'is_first_chunk': True,
                'is_chapter_start': True,
                'is_section_start': False,
                'is_section_end': False,
                'is_chapter_end': False,
                'is_last_chunk': False
            }
            
            encoded = tokenizer.encode_audiobook_chunk(test_text, chunk_info)
            decoded = tokenizer.decode(encoded, skip_special_tokens=False)
            print(f"   Encoded: {encoded}")
            print(f"   Length: {len(encoded)} tokens")
            print(f"   Decoded: '{decoded}'")
    
    print(f"\n✅ Stateful tokenizer demo completed!")
    print(f"💡 Key insight: Stateful mode avoids chunk boundary tokens")
    print(f"   → Better for continuous state propagation")
    print(f"   → Essential for audiobook prosodic continuity")


if __name__ == "__main__":
    demo_stateful_tokenizer()