import time
import torch
import numpy as np
from tqdm import tqdm
import gc
import copy
from ..trainer import predict, predict_dual
import torch.nn.functional as F


def evaluate(config,
             model,
             query_loader,
             gallery_loader,
             ranks=[1, 5, 10],
             step_size=1000,
             is_autocast=True,
             is_dual=False,
             cleanup=True):#评估阶段新增混合精度训练 避免FIND was unable to find an engine
    """
    Evaluate Recall@K for University-1652 dataset (cross-view image retrieval).
    Extract features for query and gallery sets, compute cosine similarity, and calculate Recall@K.
    Returns the Recall@1 value (float percentage).
    """
    # Feature Extraction
    print("\nExtract Features:")
    start_time = time.time()
    if is_dual:
        # Dual-mode prediction (model processes query and gallery together)
        query_features, gallery_features, query_labels, gallery_labels = predict_dual(config, model, query_loader,
                                                                                      gallery_loader,
                                                                                      is_autocast=is_autocast)
    else:
        # Separate feature extraction for query and gallery
        query_features, query_labels = predict(config, model, query_loader, is_autocast=is_autocast, input_id=1)
        gallery_features, gallery_labels = predict(config, model, gallery_loader, is_autocast=is_autocast, input_id=2)
    end_time = time.time()
    # Log extraction time
    print('Extract Features time: {:.2f} min'.format((end_time - start_time) / 60.0))

    # Optional L2 normalization of features
    if hasattr(config, 'normalize_features') and config.normalize_features:
        query_features = F.normalize(query_features, dim=-1)
        gallery_features = F.normalize(gallery_features, dim=-1)

    # Compute Recall@K using cosine similarity
    print("\nCompute Scores:")
    r1 = _calculate_scores(query_features, gallery_features, query_labels, gallery_labels, step_size=step_size,
                           ranks=ranks)

    # Cleanup GPU memory if needed
    if cleanup:
        del query_features, gallery_features, query_labels, gallery_labels
        gc.collect()

    return r1


def _calculate_scores(query_features, gallery_features, query_labels, gallery_labels, step_size=1000, ranks=[1, 5, 10]):
    """
    Compute Recall@K metrics given query and gallery features and labels.
    Uses cosine similarity (dot product of L2-normalized features) to rank gallery items for each query.
    """
    topk = copy.deepcopy(ranks)
    Q = len(query_features)
    R = len(gallery_features)

    # Compute similarity matrix in chunks to handle large matrices
    steps = Q // step_size + 1
    similarity_scores = []
    for i in range(steps):
        start = step_size * i
        end = start + step_size
        sim_batch = query_features[start:end] @ gallery_features.T  # (batch_q x R) similarity
        similarity_scores.append(sim_batch.cpu())
    # Combine all similarity chunks into a full matrix (Q x R) on CPU
    similarity = torch.cat(similarity_scores, dim=0)

    # Prepare labels
    query_labels_np = query_labels.cpu().numpy()
    gallery_labels_np = gallery_labels.cpu().numpy()

    # Map each gallery label to list of indices (to handle multiple gallery images per class)
    label_to_indices = {}
    for idx, label in enumerate(gallery_labels_np):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    # Include Recall@top1% (1% of gallery size) in evaluation
    top1_percent_rank = R // 100
    topk.append(top1_percent_rank)

    results = np.zeros(len(topk))
    valid_queries = 0  # count of queries with at least one valid gallery match

    # Iterate over each query to compute its rank position of the correct gallery match
    bar = tqdm(range(Q), ncols=100, position=0, leave=True)
    for i in bar:
        label = query_labels_np[i]
        # Skip query if it has no corresponding gallery labels or invalid label
        if label == -1 or label not in label_to_indices:
            continue
        valid_queries += 1
        # Get similarities for all gallery images with the same label (correct matches)
        correct_indices = label_to_indices[label]
        # Highest similarity among all correct gallery images for this query
        gt_sim = similarity[i, correct_indices].max().item()
        # Count how many gallery images have higher similarity than this best match
        higher_sim = (similarity[i, :] > gt_sim)
        ranking = int(higher_sim.sum().item())
        # Update counts for each rank threshold
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.0
    bar.close()
    # Brief pause to ensure progress bar is properly closed
    time.sleep(0.1)

    # Calculate percentage recalls (if no valid_queries (edge case), results remain zero)
    if valid_queries > 0:
        results = results / valid_queries * 100.0

    # Format and print Recall@K results
    output_strings = []
    for idx in range(len(topk) - 1):
        output_strings.append(f"Recall@{topk[idx]}: {results[idx]:.4f}")
    output_strings.append(f"Recall@top1: {results[-1]:.4f}")
    print(" - ".join(output_strings))

    # Return Recall@1 (as percentage)
    return results[0]
