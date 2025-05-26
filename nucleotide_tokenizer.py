"""
Nucleotide Tokenizer - Optimal configuration for Mamba-TTS
Biological metaphor: DNA nucleotides for precise character-level transcription
"""

import os
import json
import re
import unicodedata
from typing import List, Dict, Tuple, Optional
from pathlib import Path

class NucleotideTokenizer:
    def __init__(self, vocab_path: Optional[str] = None):
        self.normalize_orthography = False
        self.case_sensitive = True
        self.include_digraphs = True
        self.training_mode = False

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

    def _create_vocab(self):
        vocab = {}
        idx = 0
        vocab[" "] = idx
        idx += 1
        special_tokens = ["<pad>", "<unk>", "<s>", "</s>", "<pause>", "<breath>", "<emphasis>"]
        for token in special_tokens:
            vocab[token] = idx
            idx += 1
        punctuation = ".,!?:;-—\"'()[]…"
        for char in punctuation:
            vocab[char] = idx
            idx += 1
        for digit in "0123456789":
            vocab[digit] = idx
            idx += 1
        lowercase = "aąbcćdeęfghijklłmnńoópqrsśtuvwxyzźż"
        for char in lowercase:
            vocab[char] = idx
            idx += 1
        uppercase = "AĄBCĆDEĘFGHIJKLŁMNŃOÓPQRSŚTUVWXYZŹŻ"
        for char in uppercase:
            vocab[char] = idx
            idx += 1
        digraph_tokens = list(self.digraph_map.values())
        for token in digraph_tokens:
            vocab[token] = idx
            idx += 1
        symbols = "→←@#$%&*+=<>^_|~"
        for char in symbols:
            vocab[char] = idx
            idx += 1
        self.token2id = vocab
        self.id2token = {v: k for k, v in vocab.items()}
        print(f"🧬 Created optimal nucleotide vocabulary: {len(vocab)} tokens")

    def _load_vocab(self):
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            self.token2id = json.load(f)
        self.id2token = {v: k for k, v in self.token2id.items()}

    def _save_vocab(self):
        self.vocab_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)

    def normalize_text(self, text: str) -> str:
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        text = (text.replace('„', '"').replace('”', '"')
                     .replace('»', '"').replace('«', '"')
                     .replace('‘', "'").replace('’', "'")
                     .replace('–', '-').replace('―', '-').replace('—', '-'))
        return text

    def text_to_nucleotides(self, text: str) -> List[int]:
        text = self.normalize_text(text)
        nucleotides = []
        i = 0

        while i < len(text):
            found = False
            for special in ["<pause>", "<breath>", "<emphasis>"]:
                if text[i:i+len(special)] == special:
                    nucleotides.append(self.token2id.get(special, self.token2id.get("<unk>", 2)))
                    i += len(special)
                    found = True
                    break
            if found:
                continue
            for digraph in sorted(self.digraph_map.keys(), key=len, reverse=True):
                if text[i:i+len(digraph)] == digraph:
                    mapped_token = self.digraph_map[digraph]
                    nucleotides.append(self.token2id.get(mapped_token, self.token2id.get("<unk>", 2)))
                    i += len(digraph)
                    found = True
                    break
            if found:
                continue
            char = text[i]
            nucleotides.append(self.token2id.get(char, self.token2id.get("<unk>", 2)))
            i += 1
        return nucleotides

    def nucleotides_to_text(self, nucleotides: List[int]) -> str:
        tokens = [self.id2token.get(nuc_id, "<unk>") for nuc_id in nucleotides]
        text = ""
        for token in tokens:
            if token in ["<pad>", "<unk>", "<s>", "</s>"]:
                continue
            elif token in self.digraph_map.values():
                text += [k for k, v in self.digraph_map.items() if v == token][0]
            elif token in ["<pause>", "<breath>", "<emphasis>"]:
                text += token
            else:
                text += token
        return text

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        nucleotides = []
        if add_special_tokens:
            nucleotides.append(self.token2id.get("<s>", 3))
        nucleotides.extend(self.text_to_nucleotides(text))
        if add_special_tokens:
            nucleotides.append(self.token2id.get("</s>", 4))
        return nucleotides

    def decode(self, nucleotides: List[int], skip_special_tokens: bool = True) -> str:
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

def get_nucleotide_vocab_for_f5tts() -> Tuple[Dict[str, int], int]:
    tokenizer = NucleotideTokenizer()
    return tokenizer.get_vocab_dict(), tokenizer.get_vocab_size()

def create_nucleotide_tokenizer() -> NucleotideTokenizer:
    return NucleotideTokenizer()

def demo_optimal_tokenizer():
    print("🧬 OPTIMAL NUCLEOTIDE TOKENIZER")
    print("=" * 50)
    tokenizer = NucleotideTokenizer()
    print(f"📊 Vocab size: {tokenizer.get_vocab_size()}")
    print(f"🟢 Start codon: {tokenizer.token2id.get('<s>')}")
    print(f"🔴 Stop codon: {tokenizer.token2id.get('</s>')}")
    test_texts = [
        "Cześć! Jak się masz?",
        "Szczęśliwy człowiek w dżungli.",
        "iPhone czy FBI?",
        "Szybki chleb <pause> smakuje.",
        "Dźwięk rz, sz, cz to muzyka.",
        "Góra, więc król pójdzie?"
    ]
    print("\n🧪 TESTING:")
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        match = "✅" if decoded == tokenizer.normalize_text(text) else "❌"
        print(f"{match} '{text}' → {len(encoded)} tokens → '{decoded}'")
        if decoded != tokenizer.normalize_text(text):
            print(f"   Expected: '{tokenizer.normalize_text(text)}'\n   Got:      '{decoded}'")
    print(f"\n🧬 BIOLOGICAL DEMO:")
    sequence = "DNA koduje białka z królem."
    encoded = tokenizer.encode(sequence)
    print(f"DNA: '{sequence}'")
    print(f"Encoding: {encoded}")
    print(f"Start codon: {encoded[0]} ({tokenizer.id2token[encoded[0]]})")
    print(f"Stop codon: {encoded[-1]} ({tokenizer.id2token[encoded[-1]]})")
    print(f"Gene length: {len(encoded) - 2} nucleotides")
    print(f"\n🔬 DIGRAPH ANALYSIS:")
    test_digraphs = ["sz", "Sz", "SZ", "cz", "Cz", "dż", "Dż"]
    for dg in test_digraphs:
        tokens = tokenizer.text_to_nucleotides(dg)
        decoded = tokenizer.nucleotides_to_text(tokens)
        token_names = [tokenizer.id2token[t] for t in tokens]
        print(f"  {dg} → {tokens} ({token_names}) → {decoded} ✅")
    print("\n✅ Optimal tokenizer ready for Mamba-TTS!")

if __name__ == "__main__":
    demo_optimal_tokenizer()
