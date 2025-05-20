# data_utils.py
# This file contains utilities for loading and processing data from the Dakshina dataset

import os
import torch
import numpy as np
import pandas as pd
import subprocess
import time
import requests

def get_standard_language_code(language):
    """
    Convert language codes to the standard codes used in the Dakshina dataset.
    
    Args:
        language (str): Input language code
        
    Returns:
        str: Standardized language code
    """
    # Mapping of common language codes to Dakshina codes
    language_map = {
        'tel': 'te',  # Telugu
        'hin': 'hi',  # Hindi
        'tam': 'ta',  # Tamil
        'mal': 'ml',  # Malayalam
        'ben': 'bn',  # Bengali
        'kan': 'kn',  # Kannada
        'mar': 'mr',  # Marathi
        'guj': 'gu',  # Gujarati
        'pan': 'pa',  # Punjabi
        'urd': 'ur',  # Urdu
        'sin': 'si',  # Sinhala
        'snd': 'sd',  # Sindhi
        
        # Already standard codes (for convenience)
        'te': 'te',
        'hi': 'hi',
        'ta': 'ta',
        'ml': 'ml',
        'bn': 'bn',
        'kn': 'kn',
        'mr': 'mr',
        'gu': 'gu',
        'pa': 'pa',
        'ur': 'ur',
        'si': 'si',
        'sd': 'sd',
    }
    
    return language_map.get(language.lower(), language.lower())

def download_and_extract_dataset():
    """
    Download and extract the Dakshina dataset if not already present.
    
    This function checks if the dataset already exists in the current directory.
    If not, it downloads the dataset tar file and extracts it.
    
    Returns:
        str: Path to the extracted dataset directory
    """
    dataset_tar = "dakshina_dataset_v1.0.tar"
    dataset_dir = "dakshina_dataset_v1.0"
    
    # Check if the dataset directory already exists
    if not os.path.exists(dataset_dir):
        # Download dataset if tar file doesn't exist
        if not os.path.exists(dataset_tar):
            print("Downloading Dakshina dataset...")
            try:
                subprocess.run(["wget", "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar"], check=True)
            except subprocess.CalledProcessError:
                print("Error: Failed to download the dataset.")
                print("Trying alternative download method...")
                try:
                    # Alternative download with Python requests
                    url = "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar"
                    print(f"Downloading from {url}...")
                    r = requests.get(url, stream=True)
                    total_size = int(r.headers.get('content-length', 0))
                    block_size = 1024 * 1024  # 1 MB
                    with open(dataset_tar, 'wb') as f:
                        downloaded = 0
                        print(f"Total size: {total_size/1024/1024:.1f} MB")
                        for data in r.iter_content(block_size):
                            downloaded += len(data)
                            f.write(data)
                            done = int(50 * downloaded / total_size)
                            percent = int(100 * downloaded / total_size)
                            print(f"\r[{'=' * done}{'.' * (50-done)}] {percent}% ({downloaded/1024/1024:.1f}/{total_size/1024/1024:.1f} MB)", end='')
                    print("\nDownload completed!")
                except Exception as e:
                    print(f"Failed to download the dataset: {e}")
                    print("Please download it manually and place it in the current directory.")
        
        # Extract dataset
        if os.path.exists(dataset_tar):
            print("Extracting Dakshina dataset...")
            try:
                subprocess.run(["tar", "-xf", dataset_tar], check=True)
                print("Dataset extracted successfully.")
            except subprocess.CalledProcessError:
                print("Error: Failed to extract the dataset.")
                print("Please extract it manually with: tar -xf dakshina_dataset_v1.0.tar")
        else:
            print("Warning: Dataset tar file not found.")
    else:
        print("Dakshina dataset already exists.")
    
    # Verify dataset exists
    if os.path.exists(dataset_dir):
        return dataset_dir
    else:
        print("Warning: Dataset directory not found after download/extract attempts.")
        print("Will attempt to locate dataset files directly.")
        return None

def read_tsv(file_path):
    """
    Read a tab-separated file with source and target text.
    
    The Dakshina dataset stores data in TSV format with the target language 
    (e.g., Telugu) as the first column and English as the second column.
    
    Args:
        file_path (str): Path to the TSV file
        
    Returns:
        tuple: Two lists containing English words and target language words
    """
    eng_words = []
    tel_words = []
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"TSV file not found: {file_path}")
    
    # Open and read the TSV file
    try:
        with open(file_path, encoding='utf-8') as f:
            for ln in f:
                parts = ln.strip().split('\t')
                if len(parts) >= 2:
                    tel_words.append(parts[0])  # Dakshina format has target first
                    eng_words.append(parts[1])  # Source (English) second
    except UnicodeDecodeError:
        # Try with a different encoding if UTF-8 fails
        with open(file_path, encoding='latin-1') as f:
            for ln in f:
                parts = ln.strip().split('\t')
                if len(parts) >= 2:
                    tel_words.append(parts[0])
                    eng_words.append(parts[1])
    
    # Check if we got any data
    if not eng_words or not tel_words:
        print(f"Warning: No data found in {file_path}")
        print(f"First few lines of the file:")
        try:
            with open(file_path, encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i < 5:  # Print first 5 lines
                        print(f"  {line.strip()}")
                    else:
                        break
        except:
            print("  Unable to read file for debugging.")
    
    print(f"Read {len(eng_words)} word pairs from {file_path}")
    return eng_words, tel_words

def find_dataset_path(language='te'):
    """
    Find the correct path to the Dakshina dataset files.
    
    Args:
        language (str): Language code (e.g., 'te' for Telugu)
        
    Returns:
        str: Path to the dataset directory for the specified language
    """
    # Try different possible locations for the Dakshina dataset
    possible_paths = [
        f'./dakshina_dataset_v1.0/{language}/lexicons',
        f'../dakshina_dataset_v1.0/{language}/lexicons',
        f'/content/dakshina_dataset_v1.0/{language}/lexicons',
        f'./dakshina_dataset_v1.0/dakshina_dataset_v1.0/{language}/lexicons',
        f'/kaggle/working/dakshina_dataset_v1.0/{language}/lexicons',
        f'{language}/lexicons',
        f'dakshina_dataset_v1.0/{language}/lexicons'
    ]
    
    # Find the first path that contains the training file
    for path in possible_paths:
        train_file = os.path.join(path, f"{language}.translit.sampled.train.tsv")
        if os.path.exists(train_file):
            return path
    
    # If we get here, we didn't find the dataset
    print("Error: Could not find Dakshina dataset files.")
    print(f"Looked in the following locations:")
    for path in possible_paths:
        print(f"- {path} ({'exists' if os.path.exists(path) else 'not found'})")
        if os.path.exists(path):
            print(f"  Contents: {os.listdir(path)}")
    
    # Look for any lexicons directories to help with debugging
    found_dirs = []
    search_dirs = ['.', '..', '/content', '/kaggle/working']
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir, topdown=True):
                if 'lexicons' in dirs:
                    found_dirs.append(os.path.join(root, 'lexicons'))
                # Limit depth to avoid excessive searching
                if root.count(os.sep) - search_dir.count(os.sep) >= 3:
                    dirs.clear()  # Don't go deeper
    
    if found_dirs:
        print("\nFound potential lexicons directories:")
        for d in found_dirs:
            print(f"- {d}")
            if os.path.exists(d):
                print(f"  Contents: {os.listdir(d)}")
    
    return None

def load_dakshina_data(language='tel', base_path=None):
    """
    Load transliteration data from Dakshina TSV files.
    
    This function loads training, validation, and test data for the specified language.
    It also builds vocabularies and computes maximum word lengths.
    
    Args:
        language (str): Language code (e.g., 'tel' for Telugu)
        base_path (str, optional): Base path to the dataset
        
    Returns:
        tuple: Contains English and target language word lists, vocabularies, and max lengths
    """
    # Convert to standard language code used by Dakshina
    std_language = get_standard_language_code(language)
    print(f"Using language code '{std_language}' (from input '{language}')")
    
    # Check if we're using a non-standard code
    if std_language != language:
        print(f"Note: Dakshina dataset uses '{std_language}' as the language code for '{language}'")
    
    # If base_path is not provided, try to find it
    if base_path is None:
        base_path = find_dataset_path(std_language)
    
    if base_path is None:
        raise FileNotFoundError(f"Dakshina dataset files for language '{language}' (standardized as '{std_language}') not found in any of the expected locations")
    
    print(f"Found dataset at: {base_path}")
    
    # Construct paths to data files
    train_file = os.path.join(base_path, f"{std_language}.translit.sampled.train.tsv")
    valid_file = os.path.join(base_path, f"{std_language}.translit.sampled.dev.tsv")
    test_file = os.path.join(base_path, f"{std_language}.translit.sampled.test.tsv")
    
    # Check if files exist
    for path, name in [(train_file, "train"), (valid_file, "validation"), (test_file, "test")]:
        if not os.path.exists(path):
            print(f"Warning: {name} file not found: {path}")
            # Check if directory exists
            dir_path = os.path.dirname(path)
            if os.path.exists(dir_path):
                print(f"Directory exists, contents: {os.listdir(dir_path)}")
            else:
                print(f"Directory doesn't exist: {dir_path}")
    
    # Load data from TSV files
    print(f"Loading data from {train_file}")
    eng_list_train, tel_list_train = read_tsv(train_file)
    print(f"Loading data from {valid_file}")
    eng_list_valid, tel_list_valid = read_tsv(valid_file)
    print(f"Loading data from {test_file}")
    eng_list_test, tel_list_test = read_tsv(test_file)
    
    # Build vocabularies and find maximum word lengths
    eng_vocab = []  # List of unique letters in English words
    tel_vocab = []  # List of unique letters in target language words
    max_eng_len = -1
    max_tel_len = -1
    max_eng_word = ""
    max_tel_word = ""
    
    # Process training data for English vocabulary
    print("Building English vocabulary...")
    for word in eng_list_train:
        # Track maximum length
        max_eng_len = max(max_eng_len, len(word))
        if max_eng_len == len(word):
            max_eng_word = word
        # Add letters to vocabulary
        for letter in word:
            eng_vocab.append(letter)
    
    # Create a unique, sorted list of English characters
    eng_vocab = list(set(eng_vocab))
    eng_vocab.sort()
    
    # Process training data for target language vocabulary
    print(f"Building {language} vocabulary...")
    for word in tel_list_train:
        # Track maximum length
        max_tel_len = max(max_tel_len, len(word))
        if max_tel_len == len(word):
            max_tel_word = word
        # Add letters to vocabulary
        for letter in word:
            tel_vocab.append(letter)
    
    # Create a unique, sorted list of target language characters
    tel_vocab = list(set(tel_vocab))
    tel_vocab.sort()
    
    # Update maximum lengths from validation and test sets
    # This ensures we have proper padding for all datasets
    print("Computing maximum word lengths...")
    for word in eng_list_valid:
        max_eng_len = max(max_eng_len, len(word))
    for word in eng_list_test:
        max_eng_len = max(max_eng_len, len(word))
    for word in tel_list_test:
        max_tel_len = max(max_tel_len, len(word))
    for word in tel_list_valid:
        max_tel_len = max(max_tel_len, len(word))
    
    # Print dataset statistics
    print(f"English vocabulary size: {len(eng_vocab)}")
    print(f"Target language vocabulary size: {len(tel_vocab)}")
    print(f"Max English length: {max_eng_len}")
    print(f"Max target language length: {max_tel_len}")
    print(f"Training examples: {len(eng_list_train)}")
    print(f"Validation examples: {len(eng_list_valid)}")
    print(f"Test examples: {len(eng_list_test)}")
    
    return (eng_list_train, tel_list_train, eng_list_valid, tel_list_valid, 
            eng_list_test, tel_list_test, eng_vocab, tel_vocab, 
            max_eng_len, max_tel_len)

def word_to_vector(language, word, eng_vocab, tel_vocab, max_eng_len, max_tel_len):
    """
    Convert a word to its vectorial representation.
    
    This function converts a word into a sequence of indices that can be used by the model.
    It adds special tokens for start and end, and pads to the maximum length.
    
    Args:
        language (str): "english" or another language identifier
        word (str): The word to convert
        eng_vocab (list): English vocabulary
        tel_vocab (list): Target language vocabulary
        max_eng_len (int): Maximum length of English words
        max_tel_len (int): Maximum length of target language words
        
    Returns:
        list: Vector representation of the word
    """
    vec = []
    
    if language == "english":
        # Start token
        vec.append(len(eng_vocab) + 1)
        
        # Word content - convert each letter to its index in the vocabulary
        for letter in word:
            found = False
            for albt in range(len(eng_vocab)):
                if eng_vocab[albt] == letter:
                    vec.append(albt + 1)
                    found = True
                    break  # Stop once we find the matching letter
            
            # If character not in vocabulary, use special unknown token
            if not found:
                vec.append(0)  # Using 0 as the unknown token
        
        # Padding - fill with zeros to reach the maximum length
        while len(vec) < (max_eng_len + 1):
            vec.append(0)
        
        # End token
        vec.append(0)
    else:
        # Start token
        vec.append(len(tel_vocab) + 1)
        
        # Word content - convert each letter to its index in the vocabulary
        for letter in word:
            found = False
            for albt in range(len(tel_vocab)):
                if tel_vocab[albt] == letter:
                    vec.append(albt + 1)
                    found = True
                    break  # Stop once we find the matching letter
            
            # If character not in vocabulary, use special unknown token
            if not found:
                vec.append(0)  # Using 0 as the unknown token
        
        # Padding - fill with zeros to reach the maximum length
        while len(vec) < (max_tel_len + 1):
            vec.append(0)
        
        # End token
        vec.append(0)
    
    return vec

def prepare_matrices(eng_list, tel_list, eng_vocab, tel_vocab, max_eng_len, max_tel_len):
    """
    Create tensor matrices from word lists.
    
    This function converts lists of words into tensor matrices that can be fed into the model.
    
    Args:
        eng_list (list): List of English words
        tel_list (list): List of target language words
        eng_vocab (list): English vocabulary
        tel_vocab (list): Target language vocabulary
        max_eng_len (int): Maximum length of English words
        max_tel_len (int): Maximum length of target language words
        
    Returns:
        tuple: Tensors of English and target language word vectors
    """
    eng_matrix = []
    tel_matrix = []
    
    # Process English words
    for word in eng_list:
        eng_matrix.append(word_to_vector("english", word, eng_vocab, tel_vocab, max_eng_len, max_tel_len))
    
    # Process target language words
    for word in tel_list:
        tel_matrix.append(word_to_vector("telugu", word, eng_vocab, tel_vocab, max_eng_len, max_tel_len))
    
    # Convert lists to tensors
    return torch.tensor(eng_matrix), torch.tensor(tel_matrix)
