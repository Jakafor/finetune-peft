import numpy as np
import cv2
import torch
from skimage.measure import label
from scipy.spatial.distance import cdist
from typing import Tuple, Optional


def get_first_prompt(mask_cls: np.ndarray, 
                     dist_thre_ratio: float = 0.2,
                     sample_num: int = 5,
                     neg_prompt_ratio: float = 0.3,
                     max_points_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate point prompts for each blob in the mask.
    
    Args:
        mask_cls: Binary mask with multiple blobs
        dist_thre_ratio: Ratio for distance threshold calculation
        prompt_num: Desired number of prompts per blob
        neg_prompt_ratio: Ratio of negative (background) prompts
        max_points_ratio: If available points exceed prompt_num, use prompt_num / max_points_ratio
    
    Returns:
        Tuple of (prompts, labels) where prompts is shape (B, N, 2) and labels is (B, N)
    """
    # Find all disconnected regions
    label_msk = mask_cls.astype(np.int32)

    # Classes present (exclude background 0)
    classes = [int(c) for c in np.unique(label_msk) if c != 0]
    classes.sort()
    
    all_prompts = []
    all_labels = []

    if len(classes) > 0:
        # Process each blob
        for cls_id in classes:
            # Create binary mask for current blob
            binary_msk = (label_msk == cls_id).astype(np.uint8)
            
            # Calculate distance transform
            padded_mask = np.pad(binary_msk, ((1, 1), (1, 1)), 'constant')
            dist_img = cv2.distanceTransform(padded_mask, cv2.DIST_L2, 5).astype(np.float32)[1:-1, 1:-1]
            
            # Calculate threshold
            dist_array = np.sort(dist_img.flatten())[::-1]
            num_positive = np.sum(dist_array > 0)
            if num_positive > 0:
                dis_thre = max(dist_array[int(dist_thre_ratio * num_positive)], 1)
            else:
                dis_thre = 1
            
            # Find candidate points within threshold
            candidate_points = np.argwhere(dist_img >= dis_thre)
            max_available = len(candidate_points)
            
            # Determine actual number of prompts to use
            if max_available == 0:
                # No valid points, add dummy
                blob_prompts = np.array([[0, 0]])
                blob_labels = np.array([-1])
            else:
                # Calculate number of prompts
                if sample_num <= max_available:
                    actual_prompt_num = sample_num
                else:
                    actual_prompt_num = min(int(max_available * max_points_ratio), max_available)
                
                # Split into positive and negative prompts
                num_neg = int(actual_prompt_num * neg_prompt_ratio)
                num_pos = actual_prompt_num - num_neg
                
                blob_prompts = []
                blob_labels = []
                
                # Generate positive prompts (well-distributed inside blob)
                if num_pos > 0 and len(candidate_points) > 0:
                    pos_prompts = _select_distributed_points(candidate_points, num_pos)
                    blob_prompts.extend(pos_prompts)
                    blob_labels.extend([1] * num_pos)
                
                # Generate negative prompts (near boundary)
                if num_neg > 0:
                    neg_prompts = _select_boundary_points(binary_msk, num_neg)
                    blob_prompts.extend(neg_prompts)
                    blob_labels.extend([0] * num_neg)
                
                blob_prompts = np.array(blob_prompts)
                blob_labels = np.array(blob_labels)
            
            all_prompts.append(blob_prompts)
            all_labels.append(blob_labels)
    else:
        # No blobs found, only background
        all_prompts.append(np.array([[0, 0]]))
        all_labels.append(np.array([-1]))
    
    # Convert to tensors with consistent shape
    max_n = max(p.shape[0] for p in all_prompts)
    
    # Pad to ensure consistent N dimension
    padded_prompts = []
    padded_labels = []
    
    for prompts, labels in zip(all_prompts, all_labels):
        n_current = prompts.shape[0]
        if n_current < max_n:
            # Pad with last point and -1 label
            pad_prompts = np.tile(prompts[-1:], (max_n - n_current, 1))
            pad_labels = np.full(max_n - n_current, -1)
            prompts = np.vstack([prompts, pad_prompts])
            labels = np.concatenate([labels, pad_labels])
        
        padded_prompts.append(prompts)
        padded_labels.append(labels)
    
    # Stack into tensors
    prompts_tensor = torch.tensor(np.stack(padded_prompts), dtype=torch.float32)
    labels_tensor = torch.tensor(np.stack(padded_labels), dtype=torch.long)
    
    # Return as (prompts, labels) tuple for SAM
    return (prompts_tensor, labels_tensor)


def _select_distributed_points(candidates: np.ndarray, n: int) -> list:
    """Select n well-distributed points from candidates using farthest point sampling."""
    if len(candidates) <= n:
        return candidates.tolist()
    
    # Start with a random point
    selected_indices = [np.random.randint(len(candidates))]
    selected_points = [candidates[selected_indices[0]]]
    
    # Iteratively select farthest points
    for _ in range(n - 1):
        # Calculate distances from all candidates to selected points
        distances = cdist(candidates, selected_points).min(axis=1)
        # Select the farthest point
        farthest_idx = np.argmax(distances)
        selected_indices.append(farthest_idx)
        selected_points.append(candidates[farthest_idx])
    
    # Convert to (x, y) format for SAM
    return [(int(p[1]), int(p[0])) for p in selected_points]


def _select_boundary_points(binary_mask: np.ndarray, n: int) -> list:
    """Select n well-distributed points just outside the mask boundary."""
    # Find boundary using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary_mask, kernel, iterations=1)
    boundary = dilated - binary_mask
    
    # Get boundary points (outside the mask)
    boundary_points = np.argwhere(boundary > 0)
    
    if len(boundary_points) == 0:
        # Fallback: use random points outside mask
        outside_points = np.argwhere(binary_mask == 0)
        if len(outside_points) > 0:
            indices = np.random.choice(len(outside_points), min(n, len(outside_points)), replace=False)
            return [(int(p[1]), int(p[0])) for p in outside_points[indices]]
        else:
            return [(0, 0)] * n
    
    # Select well-distributed boundary points
    if len(boundary_points) <= n:
        selected = boundary_points
    else:
        # Use farthest point sampling for distribution
        selected = []
        indices = [np.random.randint(len(boundary_points))]
        selected.append(boundary_points[indices[0]])
        
        for _ in range(n - 1):
            distances = cdist(boundary_points, selected).min(axis=1)
            farthest_idx = np.argmax(distances)
            indices.append(farthest_idx)
            selected.append(boundary_points[farthest_idx])
        
        selected = np.array(selected)
    
    # Convert to (x, y) format
    return [(int(p[1]), int(p[0])) for p in selected]


def get_top_boxes(mask_cls: np.ndarray,
                  area_ratio_threshold: float = 0.6) -> Optional[torch.Tensor]:
    """
    Generate bounding boxes for each blob if it meets area criteria.
    
    Args:
        mask_cls: Binary mask with multiple blobs
        area_ratio_threshold: Minimum ratio of mask area to bbox area
    
    Returns:
        Tensor of shape (B, 4) with bounding boxes, or None for blobs that don't meet criteria
    """
    # Find all disconnected regions
    label_msk = mask_cls.astype(np.int32)
    classes = [int(c) for c in np.unique(label_msk) if c != 0]
    classes.sort()

    if not classes:
        return None
    
    all_boxes = []
    
    if len(classes) > 0:
        for cls_id in classes:
            # Create binary mask for current blob
            binary_msk = (label_msk == cls_id).astype(np.uint8)
            
            # Calculate bounding box
            rows, cols = np.where(binary_msk)
            if len(rows) > 0:
                y0, y1 = rows.min(), rows.max()
                x0, x1 = cols.min(), cols.max()
                
                # Calculate areas
                mask_area = np.sum(binary_msk)
                bbox_area = (x1 - x0 + 1) * (y1 - y0 + 1)
                
                # Check if mask fills enough of the bbox
                if mask_area / bbox_area >= area_ratio_threshold:
                    all_boxes.append([x0, y0, x1, y1])
                else:
                    # Doesn't meet criteria - SAM will handle None boxes
                    all_boxes.append(None)
            else:
                all_boxes.append(None)
    else:
        # No blobs found
        all_boxes.append(None)
    
    # Convert to tensor, handling None values
    # SAM's prompt encoder expects None for missing boxes
    if all(box is None for box in all_boxes):
        return None
    
    # Create tensor with valid boxes, keep track of None indices
    valid_boxes = []
    for i, box in enumerate(all_boxes):
        if box is not None:
            valid_boxes.append(box)
        else:
            # For None boxes, the prompt encoder will handle it
            # We can either pass None or handle it differently based on your pipeline
            valid_boxes.append([0, 0, 0, 0])  # Dummy box that SAM will ignore with proper masking
    
    return torch.tensor(valid_boxes, dtype=torch.float32)


# Alternative implementation that returns boxes with validity mask
def get_top_boxes_with_mask(mask_cls: np.ndarray,
                            area_ratio_threshold: float = 0.6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate bounding boxes with validity mask for proper handling in SAM.
    
    Returns:
        Tuple of (boxes, valid_mask) where valid_mask indicates which boxes are valid
    """
    label_msk, region_ids = label(mask_cls, connectivity=2, return_num=True)
    
    all_boxes = []
    valid_mask = []
    
    if region_ids > 0:
        for region_id in range(1, region_ids + 1):
            binary_msk = (label_msk == region_id).astype(np.uint8)
            rows, cols = np.where(binary_msk)
            
            if len(rows) > 0:
                y0, y1 = rows.min(), rows.max()
                x0, x1 = cols.min(), cols.max()
                
                mask_area = np.sum(binary_msk)
                bbox_area = (x1 - x0 + 1) * (y1 - y0 + 1)
                
                if mask_area / bbox_area >= area_ratio_threshold:
                    all_boxes.append([x0, y0, x1, y1])
                    valid_mask.append(True)
                else:
                    all_boxes.append([0, 0, 0, 0])
                    valid_mask.append(False)
            else:
                all_boxes.append([0, 0, 0, 0])
                valid_mask.append(False)
    else:
        all_boxes.append([0, 0, 0, 0])
        valid_mask.append(False)
    
    return torch.tensor(all_boxes, dtype=torch.float32), torch.tensor(valid_mask, dtype=torch.bool)