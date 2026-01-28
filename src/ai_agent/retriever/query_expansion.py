from typing import List, Set
import re


# Task synonyms: mapping from common user terms to variations
TASK_SYNONYMS = {
    # Segmentation family - includes OCR/text segmentation
    "segment": ["segment", "segmentation", "mask", "contour", "extract", "extraction", "delineate", "separate"],
    "segmentation": ["segmentation", "segment", "mask", "contour", "extract", "extraction", "delineate", "text-segmentation", "OCR"],
    "mask": ["mask", "segment", "segmentation", "contour", "extract"],
    "extraction": ["extraction", "extract", "segment", "segmentation", "mask", "isolate", "text-extraction", "OCR"],
    "extract": ["extract", "extraction", "segment", "segmentation", "mask", "isolate", "text-extraction"],
    
    # OCR / Text recognition family - fully bidirectional with segmentation
    "ocr": ["OCR", "text-recognition", "character-recognition", "text-extraction", "segmentation", "text-segmentation", "extract"],
    "text-recognition": ["text-recognition", "OCR", "character-recognition", "text-extraction", "segmentation", "text-segmentation"],
    "character-recognition": ["character-recognition", "OCR", "text-recognition", "text-extraction", "segmentation"],
    "text-extraction": ["text-extraction", "OCR", "text-recognition", "character-recognition", "segmentation", "extraction", "extract"],
    "text-segmentation": ["text-segmentation", "segmentation", "OCR", "text-recognition", "text-extraction", "segment"],
    
    # Denoising family
    "denoise": ["denoise", "denoising", "filter", "filtering", "clean", "cleaning", "enhance", "enhancement"],
    "denoising": ["denoising", "denoise", "filter", "filtering", "clean", "enhancement"],
    "filter": ["filter", "filtering", "denoise", "clean", "smooth", "smoothing"],
    "enhance": ["enhance", "enhancement", "improve", "denoise", "sharpen"],
    
    # Registration family
    "register": ["register", "registration", "align", "alignment", "match", "matching"],
    "registration": ["registration", "register", "align", "alignment", "match", "matching"],
    "align": ["align", "alignment", "register", "registration", "match"],
    
    # Detection family
    "detect": ["detect", "detection", "find", "identify", "locate", "recognition"],
    "detection": ["detection", "detect", "find", "identify", "locate", "recognition"],
    "identify": ["identify", "identification", "detect", "detection", "recognize", "recognition"],
    
    # Reconstruction family
    "reconstruct": ["reconstruct", "reconstruction", "build", "generate", "synthesis"],
    "reconstruction": ["reconstruction", "reconstruct", "build", "generate", "synthesis"],
    
    # Classification family
    "classify": ["classify", "classification", "categorize", "predict", "prediction"],
    "classification": ["classification", "classify", "categorize", "predict", "prediction"],
}

# Anatomy synonyms
ANATOMY_SYNONYMS = {
    "lung": ["lung", "pulmonary", "respiratory"],
    "lungs": ["lungs", "pulmonary", "respiratory"],
    "pulmonary": ["pulmonary", "lung", "lungs", "respiratory"],
    
    "brain": ["brain", "cerebral", "neural", "cranial"],
    "cerebral": ["cerebral", "brain", "neural"],
    
    "heart": ["heart", "cardiac", "cardiovascular"],
    "cardiac": ["cardiac", "heart", "cardiovascular"],
    
    "liver": ["liver", "hepatic"],
    "hepatic": ["hepatic", "liver"],
    
    "kidney": ["kidney", "renal"],
    "renal": ["renal", "kidney"],
    
    "vessel": ["vessel", "vascular", "artery", "vein"],
    "vessels": ["vessels", "vascular", "arteries", "veins"],
    "vascular": ["vascular", "vessel", "vessels", "artery"],
    
    "bone": ["bone", "skeletal", "osseous"],
    "bones": ["bones", "skeletal", "osseous"],
    
    "cell": ["cell", "cellular"],
    "cells": ["cells", "cellular"],
    "nuclei": ["nuclei", "nucleus", "cell"],
    "nucleus": ["nucleus", "nuclei", "cell"],
    
    "text": ["text", "document", "character", "word", "handwriting", "OCR", "historical"],
    "document": ["document", "text", "page", "manuscript", "historical", "OCR"],
    "character": ["character", "text", "letter", "OCR", "glyph"],
    "handwriting": ["handwriting", "manuscript", "text", "OCR", "historical"],
    "manuscript": ["manuscript", "document", "historical", "handwriting", "text", "OCR"],
}

# Modality synonyms
MODALITY_SYNONYMS = {
    "ct": ["CT", "computed-tomography", "computed tomography", "CAT"],
    "mri": ["MRI", "magnetic-resonance", "magnetic resonance"],
    # Put OCR first for historical documents - it's the most important cross-vocabulary bridge
    "historical-documents": ["OCR", "text", "historical-documents", "historical", "document", "manuscript", "archive"],
    "historical": ["OCR", "text", "historical", "historical-documents", "document", "manuscript", "archive"],
    "xray": ["X-ray", "xray", "radiography", "radiograph"],
    "x-ray": ["X-ray", "xray", "radiography", "radiograph"],
    "ultrasound": ["ultrasound", "US", "sonography", "echo"],
    "pet": ["PET", "positron-emission", "positron emission"],
    "microscopy": ["microscopy", "microscope", "imaging"],
    "fluorescence": ["fluorescence", "fluorescent", "fluor"],
}

# Dimension synonyms
DIMENSION_SYNONYMS = {
    "2d": ["2D", "2-D", "two-dimensional", "planar", "slice", "image"],
    "3d": ["3D", "3-D", "three-dimensional", "volumetric", "volume", "stack", "tomography"],
    "4d": ["4D", "4-D", "four-dimensional", "temporal", "time-series", "timeseries", "dynamic"],
    "volume": ["volume", "volumetric", "3D", "3-D", "stack"],
    "volumetric": ["volumetric", "volume", "3D", "3-D", "stack"],
    "stack": ["stack", "volume", "volumetric", "3D", "3-D"],
}


def expand_query(query: str, max_expansions_per_term: int = 3) -> str:
    """
    Expand query with synonyms to improve recall.
    
    Keeps original query intact and appends synonym terms.
    Limits expansions to avoid query bloat.
    
    Args:
        query: Original user query
        max_expansions_per_term: Maximum number of synonym expansions per matched term
    
    Returns:
        Expanded query string
    
    Example:
        >>> expand_query("segment the lungs")
        "segment the lungs segmentation mask pulmonary respiratory"
    """
    # Normalize to lowercase for matching
    query_lower = query.lower()
    words = re.findall(r'\b\w+\b', query_lower)
    
    # Collect expansions (using sets to avoid duplicates)
    expansions: Set[str] = set()
    
    # Check each word against synonym dictionaries
    for word in words:
        # Task synonyms
        if word in TASK_SYNONYMS:
            synonyms = TASK_SYNONYMS[word][:max_expansions_per_term]
            expansions.update(s for s in synonyms if s.lower() != word)
        
        # Anatomy synonyms
        if word in ANATOMY_SYNONYMS:
            synonyms = ANATOMY_SYNONYMS[word][:max_expansions_per_term]
            expansions.update(s for s in synonyms if s.lower() != word)
        
        # Modality synonyms
        if word in MODALITY_SYNONYMS:
            synonyms = MODALITY_SYNONYMS[word][:max_expansions_per_term]
            expansions.update(s for s in synonyms if s.lower() != word)
        
        # Dimension synonyms
        if word in DIMENSION_SYNONYMS:
            synonyms = DIMENSION_SYNONYMS[word][:max_expansions_per_term]
            expansions.update(s for s in synonyms if s.lower() != word)
    
    # Build expanded query: original + expansions
    if expansions:
        expansion_str = " ".join(sorted(expansions))
        return f"{query} {expansion_str}"
    
    return query


def expand_terms(terms: List[str]) -> List[str]:
    """
    Expand a list of terms with their synonyms.
    
    Used internally for document indexing to add synonym terms
    to the retrieval text.
    
    Args:
        terms: List of terms to expand
    
    Returns:
        Expanded list including original terms and synonyms
    """
    expanded = set(terms)  # Start with originals
    
    for term in terms:
        term_lower = term.lower()
        
        # Check all synonym dictionaries
        for synonym_dict in [TASK_SYNONYMS, ANATOMY_SYNONYMS, MODALITY_SYNONYMS, DIMENSION_SYNONYMS]:
            if term_lower in synonym_dict:
                # Add top 2 synonyms per term to avoid bloat
                synonyms = synonym_dict[term_lower][:2]
                expanded.update(synonyms)
    
    return list(expanded)
