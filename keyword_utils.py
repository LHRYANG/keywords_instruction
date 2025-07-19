#!/usr/bin/env python3
"""
Utility functions for reading different types of keyword files in the keywords_instruction project.

This module provides functions to read and parse various keyword file formats:
- Inside keywords (frequency-based JSON files)
- Outside keywords (categorized JSON files)
- Text files (line-by-line format)
- Generated results (expanded keyword pairs)
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union, Optional


class KeywordLoader:
    """A utility class for loading different types of keyword files."""
    
    def __init__(self, base_path: str = None):
        """
        Initialize the KeywordLoader.
        
        Args:
            base_path: Base path to the keywords_instruction project. 
                      If None, uses current directory.
        """
        if base_path is None:
            base_path = os.getcwd()
        self.base_path = Path(base_path)
        self.inside_keywords_path = self.base_path / "inside_keywords"
        self.outside_keywords_path = self.base_path / "outside_keywords"
    
    def load_inside_keywords(self, filename: str, max_length: int = None, 
                           min_frequency: int = 1) -> Dict[str, int]:
        """
        Load inside keywords from frequency-based JSON files.
        
        Args:
            filename: Name of the JSON file (e.g., 'edit_type_add_noun_freq_by_length.json')
            max_length: Maximum keyword length to include (None for all)
            min_frequency: Minimum frequency threshold
            
        Returns:
            Dictionary mapping keywords to their frequencies
        """
        file_path = self.inside_keywords_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Inside keywords file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        result = {}
        for key, keywords in data.items():
            # Extract length from key like "keyword_length_1"
            if key.startswith("keyword_length_"):
                length = int(key.split("_")[-1])
                if max_length is None or length <= max_length:
                    for keyword, freq in keywords.items():
                        if freq >= min_frequency:
                            result[keyword] = freq
        
        return result
    
    def load_inside_keywords_by_length(self, filename: str, 
                                     target_lengths: List[int] = None) -> Dict[int, Dict[str, int]]:
        """
        Load inside keywords grouped by length.
        
        Args:
            filename: Name of the JSON file
            target_lengths: List of lengths to include (None for all)
            
        Returns:
            Dictionary mapping length -> {keyword: frequency}
        """
        file_path = self.inside_keywords_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Inside keywords file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        result = {}
        for key, keywords in data.items():
            if key.startswith("keyword_length_"):
                length = int(key.split("_")[-1])
                if target_lengths is None or length in target_lengths:
                    result[length] = keywords
        
        return result
    
    def load_outside_keywords(self, filename: str, 
                            categories: List[str] = None) -> Dict[str, List[str]]:
        """
        Load outside keywords from categorized JSON files.
        
        Args:
            filename: Name of the JSON file (e.g., 'keyword_color.json')
            categories: List of categories to include (None for all)
            
        Returns:
            Dictionary mapping categories to keyword lists
        """
        file_path = self.outside_keywords_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Outside keywords file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if categories is None:
            return data
        
        return {cat: data[cat] for cat in categories if cat in data}
    
    def load_text_file(self, filename: str, strip_empty: bool = True) -> List[str]:
        """
        Load keywords from text files (line-by-line format).
        
        Args:
            filename: Name of the text file (e.g., 'animal.txt')
            strip_empty: Whether to remove empty lines
            
        Returns:
            List of keywords
        """
        file_path = self.outside_keywords_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
        
        if strip_empty:
            lines = [line for line in lines if line]
        
        return lines
    
    def load_generated_results(self, filepath: str) -> List[Dict]:
        """
        Load generated keyword results (expanded keyword pairs).
        
        Args:
            filepath: Path to the generated results JSON file
            
        Returns:
            List of dictionaries with object and expanded_keywords
        """
        file_path = Path(filepath)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Generated results file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def sample_keywords_from_outside(self, max_per_category: int = 10, 
                                   exclude_files: List[str] = None) -> str:
        """
        Sample keywords from outside keyword files for use in prompts.
        
        Args:
            max_per_category: Maximum number of keywords per category
            exclude_files: List of files to exclude (e.g., ['keyword_color.json'])
            
        Returns:
            Formatted string with sampled keywords
        """
        if exclude_files is None:
            exclude_files = []
        
        exclude_stems = [Path(f).stem for f in exclude_files]
        
        all_jsons = list(self.outside_keywords_path.glob("*.json"))
        lines = []
        
        for json_file in all_jsons:
            if json_file.stem in exclude_stems:
                continue
                
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                sampled = {}
                for category, words in data.items():
                    if isinstance(words, list):
                        sampled_words = random.sample(words, min(len(words), max_per_category))
                        sampled[category] = sampled_words
                
                line = f"{json_file.stem}: {json.dumps(sampled, ensure_ascii=False)}"
                lines.append(line)
            
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
                continue
        
        return "\n".join(lines)
    
    def get_all_nouns(self, max_length: int = 3, min_frequency: int = 1) -> Set[str]:
        """
        Get all nouns from the add_noun frequency file.
        
        Args:
            max_length: Maximum keyword length
            min_frequency: Minimum frequency threshold
            
        Returns:
            Set of all noun keywords
        """
        try:
            keywords = self.load_inside_keywords(
                "edit_type_add_noun_freq_by_length.json", 
                max_length=max_length, 
                min_frequency=min_frequency
            )
            return set(keywords.keys())
        except FileNotFoundError:
            print("Warning: edit_type_add_noun_freq_by_length.json not found")
            return set()
    
    def get_color_keywords(self) -> List[str]:
        """
        Get all color keywords from the color JSON file.
        
        Returns:
            List of color keywords
        """
        try:
            color_data = self.load_outside_keywords("keyword_color.json")
            return color_data.get("colors", [])
        except FileNotFoundError:
            print("Warning: keyword_color.json not found")
            return []
    
    def list_inside_files(self) -> List[str]:
        """List all available inside keyword files."""
        if not self.inside_keywords_path.exists():
            return []
        return [f.name for f in self.inside_keywords_path.glob("*.json")]
    
    def list_outside_files(self) -> List[str]:
        """List all available outside keyword files."""
        if not self.outside_keywords_path.exists():
            return []
        return [f.name for f in self.outside_keywords_path.glob("*")]
    
    def get_file_info(self, filename: str, file_type: str = "auto") -> Dict:
        """
        Get information about a keyword file.
        
        Args:
            filename: Name of the file
            file_type: Type of file ('inside', 'outside', 'text', 'auto')
            
        Returns:
            Dictionary with file information
        """
        info = {"filename": filename, "exists": False}
        
        if file_type == "auto":
            if filename.endswith(".json"):
                if self.inside_keywords_path.joinpath(filename).exists():
                    file_type = "inside"
                elif self.outside_keywords_path.joinpath(filename).exists():
                    file_type = "outside"
            elif filename.endswith(".txt"):
                file_type = "text"
        
        if file_type == "inside":
            file_path = self.inside_keywords_path / filename
        elif file_type == "outside":
            file_path = self.outside_keywords_path / filename
        elif file_type == "text":
            file_path = self.outside_keywords_path / filename
        else:
            return info
        
        if file_path.exists():
            info["exists"] = True
            info["type"] = file_type
            info["path"] = str(file_path)
            info["size"] = file_path.stat().st_size
            
            try:
                if file_type in ["inside", "outside"]:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    info["structure"] = list(data.keys())
                elif file_type == "text":
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    info["line_count"] = len(lines)
            except Exception as e:
                info["error"] = str(e)
        
        return info


# Convenience functions for backward compatibility
def load_nouns_from_json(json_path: str, max_length: int = 1) -> List[str]:
    """Load nouns from inside keywords JSON file (backward compatibility)."""
    loader = KeywordLoader(os.path.dirname(json_path))
    filename = os.path.basename(json_path)
    keywords = loader.load_inside_keywords(filename, max_length=max_length)
    return list(keywords.keys())


def sample_keywords_from_json(folder: str, max_per_category: int = 10) -> str:
    """Sample keywords from outside folder (backward compatibility)."""
    loader = KeywordLoader(os.path.dirname(folder))
    return loader.sample_keywords_from_outside(max_per_category=max_per_category)


# Example usage and testing
if __name__ == "__main__":
    # Initialize loader
    loader = KeywordLoader()
    
    print("=== Available Files ===")
    print("Inside files:", loader.list_inside_files())
    print("Outside files:", loader.list_outside_files())
    
    print("\n=== Loading Inside Keywords ===")
    try:
        nouns = loader.load_inside_keywords("edit_type_add_noun_freq_by_length.json", max_length=2)
        print(f"Loaded {len(nouns)} nouns (length ≤ 2)")
        print("Top 10 nouns:", list(nouns.items())[:10])
    except FileNotFoundError as e:
        print(f"Error: {e}")
    
    print("\n=== Loading Outside Keywords ===")
    try:
        colors = loader.load_outside_keywords("keyword_color.json")
        print(f"Color categories: {list(colors.keys())}")
        print(f"Number of colors: {len(colors.get('colors', []))}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    
    print("\n=== Loading Text File ===")
    try:
        animals = loader.load_text_file("animal.txt")
        print(f"Loaded {len(animals)} animals")
        print("First 5 animals:", animals[:5])
    except FileNotFoundError as e:
        print(f"Error: {e}")
    
    print("\n=== Sampling Keywords ===")
    try:
        sample = loader.sample_keywords_from_outside(max_per_category=3)
        print("Sample keywords:")
        print(sample[:200] + "..." if len(sample) > 200 else sample)
    except Exception as e:
        print(f"Error: {e}")
