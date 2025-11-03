"""
Utilities for manipulating embedding geometry.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_token_ids(tokenizer, words):
    """Get token IDs for a list of words.

    Args:
        tokenizer: HuggingFace tokenizer
        words: List of words to tokenize

    Returns:
        List of token IDs (one per word)

    Raises:
        ValueError: If any word maps to multiple tokens
    """
    token_ids = []
    for word in words:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if len(tokens) != 1:
            raise ValueError(f"Word '{word}' maps to {len(tokens)} tokens: {tokens}")
        token_ids.append(tokens[0])
    return token_ids


def compute_centroid(embeddings, token_ids):
    """Compute the centroid of embeddings for given token IDs.

    Args:
        embeddings: Embedding layer (model.get_input_embeddings())
        token_ids: List of token IDs

    Returns:
        Tensor representing the centroid (mean of embeddings)
    """
    vectors = [embeddings.weight[tid] for tid in token_ids]
    centroid = torch.stack(vectors).mean(dim=0)
    return centroid


def modify_embeddings(model, tokenizer, exemplars, scale_factor):
    """Modify exemplar embeddings by scaling distance to centroid.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        exemplars: List of exemplar words (e.g., ["dog", "cat", "hamster"])
        scale_factor: Scaling factor for distances
            - 0.5 = move 50% closer (tight cluster)
            - 2.0 = move 100% farther (loose cluster)

    Returns:
        dict with modification statistics
    """
    embeddings = model.get_input_embeddings()
    token_ids = get_token_ids(tokenizer, exemplars)

    # Get original embeddings
    original_vecs = [embeddings.weight[tid].clone() for tid in token_ids]

    # Compute centroid
    centroid = compute_centroid(embeddings, token_ids)

    # Modify each exemplar embedding
    stats = {
        "original_distances": [],
        "new_distances": [],
        "token_ids": token_ids,
        "exemplars": exemplars,
    }

    with torch.no_grad():
        for tid, orig_vec in zip(token_ids, original_vecs):
            # Vector from centroid to exemplar
            offset = orig_vec - centroid

            # Original distance
            orig_dist = offset.norm().item()
            stats["original_distances"].append(orig_dist)

            # Scale the offset
            if scale_factor is None:
                # Baseline - no change
                new_vec = orig_vec
            else:
                new_vec = centroid + (scale_factor * offset)

            # Update embedding
            embeddings.weight[tid] = new_vec

            # New distance
            new_dist = (new_vec - centroid).norm().item()
            stats["new_distances"].append(new_dist)

    return stats


def print_modification_stats(stats, condition):
    """Print statistics about embedding modifications."""
    print(f"\n{'='*60}")
    print(f"Condition: {condition}")
    print(f"{'='*60}")

    for exemplar, tid, orig_d, new_d in zip(
        stats["exemplars"],
        stats["token_ids"],
        stats["original_distances"],
        stats["new_distances"]
    ):
        print(f"{exemplar:10} (token {tid:5d}): {orig_d:.4f} -> {new_d:.4f}")

    avg_orig = sum(stats["original_distances"]) / len(stats["original_distances"])
    avg_new = sum(stats["new_distances"]) / len(stats["new_distances"])
    print(f"\nAverage distance: {avg_orig:.4f} -> {avg_new:.4f}")
