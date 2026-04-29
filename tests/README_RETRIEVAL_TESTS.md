# Retrieval Pipeline Test Suite

Comprehensive test coverage for the RAGImagingPipeline after removing query expansion.

## Test Summary

**Total Tests:** 34  
**Status:** ✅ All Passing  
**Runtime:** ~20 minutes (includes model loading)

## Quick Start

```bash
# Run all tests
pytest tests/test_retrieval_pipeline.py -v

# Run specific test class
pytest tests/test_retrieval_pipeline.py::TestMedicalRequests -v

# Run with verbose logging
pytest tests/test_retrieval_pipeline.py -v -s

# Run single test
pytest tests/test_retrieval_pipeline.py::TestMedicalRequests::test_lung_segmentation_ct -v
```

## Test Organization

### 1. Medical Imaging Requests (4 tests)
Tests retrieval for medical imaging tasks with domain-specific terminology.

- `test_lung_segmentation_ct` - Precise medical request with modality
- `test_brain_mri_registration` - Medical registration task  
- `test_medical_abbreviation` - Medical abbreviation understanding (CT scan)
- `test_dicom_format_hint` - DICOM format-specific request with file hints

**Key Verification:** Medical terms, anatomical structures, imaging modalities (CT, MRI) are correctly matched.

### 2. Non-Medical Requests (4 tests)
Tests retrieval for general computer vision and image processing tasks.

- `test_ocr_text_extraction` - OCR request (may not be in catalog)
- `test_image_classification` - General computer vision task
- `test_deblurring_restoration` - Image restoration task
- `test_jpeg_format_hint` - JPEG image processing with format hints

**Key Verification:** Domain-agnostic retrieval works, non-medical terms properly matched.

### 3. Vague vs. Precise Spectrum (4 tests)
Tests queries ranging from very vague to highly specific.

- `test_vague_analyze_image` - Very vague request ("analyze image")
- `test_vague_segment` - Vague task without context ("segment")
- `test_precise_3d_liver_segmentation_dicom` - Very precise with multiple constraints
- `test_moderate_precision_nifti_viewer` - Moderately precise request

**Key Verification:** System handles both broad and narrow queries appropriately.

### 4. Out of Catalog Requests (4 tests)
Tests queries for tasks likely not in the imaging tool catalog.

- `test_video_editing` - Video editing (out of scope)
- `test_audio_processing` - Audio processing (definitely out of scope)
- `test_3d_rendering_animation` - 3D rendering/animation task
- `test_document_layout_analysis` - Document analysis task

**Key Verification:** System returns nearest matches gracefully, doesn't fail on out-of-scope queries.

### 5. Retrieval Modes (4 tests)
Tests different retrieval configurations and modes.

- `test_retrieve_no_rerank` - Retrieval without CrossEncoder reranking
- `test_retrieve_with_rerank` - Full retrieval with reranking
- `test_rerank_improves_precision` - Verify reranking improves result quality
- `test_exclusion_filter` - Exclusion filter works correctly

**Key Verification:** Reranking improves precision, exclusions work, both modes return valid results.

### 6. Image Metadata Integration (4 tests)
Tests image metadata hint generation and integration.

- `test_format_hint_dicom` - DICOM format hint added to query
- `test_format_hint_nifti` - NIfTI format hint added
- `test_format_hint_tiff_stack` - TIFF stack hint for microscopy
- `test_multiple_formats` - Multiple file formats in one request

**Key Verification:** Format tokens (format:dicom, format:nifti) correctly enhance retrieval.

### 7. Edge Cases (5 tests)
Tests error conditions and boundary cases.

- `test_empty_query` - Empty query string
- `test_very_long_query` - Extremely long query
- `test_special_characters_query` - Query with special characters
- `test_top_k_zero` - Request zero results
- `test_top_k_large` - Request more results than available

**Key Verification:** System handles edge cases gracefully without crashes.

### 8. Retry Mechanism (2 tests)
Tests the retry mechanism for insufficient results.

- `test_retry_broadens_query` - Very specific query triggers retry
- `test_obscure_term_retry` - Obscure medical term needs retry

**Key Verification:** Retry mechanism activates when needed, broadens search appropriately.

### 9. Semantic Understanding (3 tests)
Tests BGE-M3's semantic understanding capabilities.

- `test_synonym_understanding_visualize_display` - Synonyms (visualize/display/show)
- `test_related_concepts_segmentation` - Related concept understanding (partition→segment)
- `test_acronym_vs_full_form` - Acronym vs full form (CT vs Computed Tomography)

**Key Verification:** Semantic embeddings handle vocabulary variations naturally.

## What Changed

These tests verify the **new simplified retrieval pipeline** that:

1. ✅ **Removed query expansion** - No more hardcoded synonym dictionaries
2. ✅ **Relies on BGE-M3** - Semantic embeddings handle vocabulary naturally
3. ✅ **Uses CrossEncoder reranking** - Precision layer after vector search
4. ✅ **Integrates image metadata** - Format tokens and metadata hints enhance retrieval
5. ✅ **Domain-agnostic** - Works for medical and non-medical tasks

## Key Assertions

Each test verifies:
- Results are returned (non-empty list)
- Top results are relevant (name matching, description content)
- Scores are properly set (similarity, rerank scores)
- Edge cases handled gracefully (no crashes)
- Semantic understanding works (synonyms, acronyms, related concepts)

## Performance Notes

- **First test is slowest** (~40s) - Loads BGE-M3 model and builds FAISS index
- **Subsequent tests are faster** - Models stay in memory (module-scoped fixture)
- **Full suite takes ~20 minutes** - Due to 34 tests × ~35s average per test
- **Optimize:** Use `-k` to run subset, or `--lf` to run last failed

## Debugging Failed Tests

```bash
# Run with full traceback
pytest tests/test_retrieval_pipeline.py::TestName::test_name -v --tb=long

# Run with print statements visible
pytest tests/test_retrieval_pipeline.py::TestName::test_name -v -s

# Stop at first failure
pytest tests/test_retrieval_pipeline.py -x

# Run last failed tests only
pytest tests/test_retrieval_pipeline.py --lf
```

## Adding New Tests

When adding tests:
1. Choose appropriate test class (or create new one)
2. Use descriptive test names: `test_<what>_<scenario>`
3. Log key results for debugging: `log.info(f"Result: {result}")`
4. Assert meaningful conditions (not just "len > 0")
5. Document expected behavior in docstring

Example:
```python
def test_new_scenario(self, pipeline):
    """Test: Brief description of what this tests."""
    results = pipeline.retrieve("query here", top_k=5)
    
    assert len(results) > 0, "Should find results"
    
    # Check specific behavior
    result_names = [r["doc"].name for r in results]
    log.info(f"Found: {result_names[:3]}")
    
    assert some_condition, "Explain why this should be true"
```

## Continuous Integration

To run in CI/CD:
```bash
# Fast smoke test (3 tests, ~2 min)
pytest tests/test_retrieval_pipeline.py -k "lung_segmentation or ocr or empty_query"

# Medium coverage (10 tests, ~6 min)  
pytest tests/test_retrieval_pipeline.py -k "Medical or NonMedical or edge"

# Full suite (34 tests, ~20 min)
pytest tests/test_retrieval_pipeline.py
```

## Related Files

- **Pipeline:** [`src/ai_agent/api/pipeline.py`](../src/ai_agent/api/pipeline.py)
- **Embedder:** [`src/ai_agent/retriever/text_embedder.py`](../src/ai_agent/retriever/text_embedder.py)
- **Reranker:** [`src/ai_agent/retriever/reranker.py`](../src/ai_agent/retriever/reranker.py)
- **Vector Index:** [`src/ai_agent/retriever/vector_index.py`](../src/ai_agent/retriever/vector_index.py)
- **Image Metadata:** [`src/ai_agent/utils/image_meta.py`](../src/ai_agent/utils/image_meta.py)
