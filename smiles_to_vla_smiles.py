"""
SMILES to VLA-SMILES Converter
This script is designed to process files containing SMILES (Simplified Molecular Input Line Entry System) strings. 
It first converts each SMILES string into a corresponding binary sequence. 
Subsequently, it pads these sequences to ensure uniform length, matching the longest sequence in the set. 
Finally, the script transforms these binary and zero-padded SMILES into VLA-SMILES (Variable Length Array SMILES),
using a range of divisors to determine the transformation logic.

If you are utilizing the current descriptors, please ensure to properly cite the following source:
Nazarova, A.L.; Nakano, A. "VLA-SMILES: Variable-Length-Array SMILES Descriptors in Neural Network-Based QSAR Modeling." 
Machine Learning and Knowledge Extraction, 2022, Vol. 4, pp. 715-737. Available online: https://doi.org/10.3390/make4030034.

Usage:
    python smiles_to_vla_smiles.py input.dat

Author:
    Dr. Antonina L. Nazarova (aka Cover Girl)
"""

import sys
import numpy as np

def smiles_to_binary(smiles, max_length):
    """
    Convert a SMILES string to a binary sequence and pad with zeros to the specified max_length.

    Args:
    smiles (str): The SMILES string to convert.
    max_length (int): The length to which the binary sequence should be padded.

    Returns:
    str: The padded binary sequence.
    """
    binary_sequence = ''.join(f'{ord(c):08b}' for c in smiles)
    padding_length = max_length - len(binary_sequence)
    return binary_sequence + '0' * padding_length

def read_binary_smiles(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

def find_divisors(n):
    divisors = []
    for i in range(1, n + 1):
        if n % i == 0:
            divisors.append(i)
    return divisors

def transform_to_vla(smiles, k):
    num_samples = len(smiles)
    sequence_length = len(smiles[0])
    vla_smiles = np.zeros((num_samples, sequence_length // k))

    for i1 in range(num_samples):
        for i in range(sequence_length // k):
            bits = [int(smiles[i1][j]) for j in range(i * k, (i + 1) * k)]
            bits = [bit if bit >= 0 else 2 for bit in bits]
            r20 = sum(bit * 2 ** (k - idx - 1) for idx, bit in enumerate(bits))
            if r20 == 0:
                continue
            bits = [0 if bit == 2 else bit for bit in bits]
            r20 = sum(bit * 2 ** (k - idx - 1) for idx, bit in enumerate(bits))
            # Apply any additional transformation here if needed
            vla_smiles[i1][i] = r20
    return vla_smiles

def process_smiles_file(input_file):
    """
    Main processing function to convert SMILES to binary and then to VLA-SMILES.
    """
    try:
        with open(input_file, 'r') as file:
            smiles_list = file.read().splitlines()

        max_smiles_length = max(len(smiles) for smiles in smiles_list) * 8
        binary_sequences = [smiles_to_binary(smiles, max_smiles_length) for smiles in smiles_list]

        binary_file = 'binary_output.dat'
        with open(binary_file, 'w') as file:
            for sequence in binary_sequences:
                file.write(sequence + '\n')

        sequence_length = len(binary_sequences[0])
        divisors = find_divisors(sequence_length)

        for k in divisors:
            vla_smiles = transform_to_vla(binary_sequences, k)
            output_file = f'vla_output_{k}.dat'
            np.savetxt(output_file, vla_smiles, fmt='%d')

    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
        return

def main():
    if len(sys.argv) != 2:
        print("Usage: python smiles_to_vla_smiles.py input.dat")
        sys.exit(1)

    input_file = sys.argv[1]
    process_smiles_file(input_file)

if __name__ == "__main__":
    main()
