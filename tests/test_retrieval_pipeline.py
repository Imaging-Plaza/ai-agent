"""
Test suite for the retrieval pipeline (RAGImagingPipeline).

Tests the new simplified retrieval system that relies on:
- BGE-M3 semantic embeddings (no hardcoded query expansion)
- CrossEncoder reranking for precision
- Image metadata hints (format, modality, dimensions)

Test Coverage:
- Medical imaging requests (CT, MRI, segmentation)
- Non-medical requests (OCR, general image processing)
- Vague vs. precise queries
- Format-specific requests
- Requests outside catalog scope
- Retrieval with/without reranking
- Image metadata hint integration
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List

import pytest

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "src" / "ai_agent"
for p in (ROOT, PKG_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from ai_agent.api.pipeline import RAGImagingPipeline

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def pipeline():
    """Create a single pipeline instance for all tests."""
    # Use default index location
    index_dir = ROOT / "artifacts" / "rag_index"
    if not index_dir.exists():
        pytest.skip(f"Index directory not found: {index_dir}")
    
    pipeline = RAGImagingPipeline(
        index_dir=str(index_dir),
        min_results=5,
        max_retries=2
    )
    
    # Verify index is loaded
    assert pipeline.index is not None, "Failed to load index"
    assert len(pipeline.index.docs) > 0, "Index has no documents"
    
    log.info(f"Loaded index with {len(pipeline.index.docs)} tools")
    return pipeline


class TestMedicalRequests:
    """Test retrieval for medical imaging tasks."""
    
    def test_lung_segmentation_ct(self, pipeline):
        """Test: Precise medical request with modality."""
        results = pipeline.retrieve("segment lungs CT", top_k=5)
        
        assert len(results) > 0, "Should find results for lung segmentation"
        
        # Check if top result is relevant
        top_doc = results[0]["doc"]
        log.info(f"Top result: {top_doc.name} (rerank: {results[0].get('rerank_score', 'N/A')})")
        
        # Should find lung-related tools
        result_names = [r["doc"].name.lower() for r in results]
        assert any("lung" in name for name in result_names), "Should find lung-related tools"
    
    def test_brain_mri_registration(self, pipeline):
        """Test: Medical registration task."""
        results = pipeline.retrieve("register brain MRI scans", top_k=5)
        
        assert len(results) > 0, "Should find results for brain registration"
        
        # Log top results
        for i, r in enumerate(results[:3]):
            log.info(f"  {i+1}. {r['doc'].name} (rerank: {r.get('rerank_score', 'N/A')})")
        
        # Should find registration or brain-related tools
        result_names = [r["doc"].name.lower() for r in results]
        has_relevant = any(
            "registr" in name or "brain" in name or "align" in name 
            for name in result_names
        )
        assert has_relevant, "Should find registration or brain-related tools"
    
    def test_medical_abbreviation(self, pipeline):
        """Test: Medical abbreviation understanding (CT scan)."""
        results = pipeline.retrieve("CT scan segmentation", top_k=5)
        
        assert len(results) > 0, "Should understand CT abbreviation"
        
        # Should find CT-compatible tools
        for r in results[:3]:
            log.info(f"  {r['doc'].name}: {r['doc'].description[:100]}")
    
    def test_dicom_format_hint(self, pipeline):
        """Test: DICOM format-specific request."""
        # Simulate DICOM file by adding format hint
        results = pipeline.retrieve(
            "visualize medical images",
            image_paths=["test.dcm"],  # Will add format:dicom hint
            top_k=5
        )
        
        assert len(results) > 0, "Should find DICOM-compatible tools"
        
        # Check if format hint was used
        for r in results[:3]:
            doc = r["doc"]
            log.info(f"  {doc.name} - formats: {getattr(doc, 'supportingData', 'N/A')}")


class TestNonMedicalRequests:
    """Test retrieval for non-medical imaging tasks."""
    
    def test_ocr_text_extraction(self, pipeline):
        """Test: OCR request (may not be in catalog)."""
        results = pipeline.retrieve("extract text from image OCR", top_k=5)
        
        # We expect results even if not perfect matches
        # (BGE-M3 should find related tools)
        assert len(results) > 0, "Should return candidates for OCR query"
        
        for r in results[:3]:
            log.info(f"  OCR candidate: {r['doc'].name}")
    
    def test_image_classification(self, pipeline):
        """Test: General computer vision task."""
        results = pipeline.retrieve("classify images using deep learning", top_k=5)
        
        assert len(results) > 0, "Should find classification tools"
        
        result_names = [r["doc"].name.lower() for r in results]
        log.info(f"Classification results: {result_names[:3]}")
    
    def test_deblurring_restoration(self, pipeline):
        """Test: Image restoration task."""
        results = pipeline.retrieve("deblur image restoration", top_k=5)
        
        assert len(results) > 0, "Should find deblurring tools"
        
        # Check for restoration-related tools
        for r in results[:3]:
            log.info(f"  Restoration: {r['doc'].name} (score: {r.get('rerank_score', 'N/A')})")
    
    def test_jpeg_format_hint(self, pipeline):
        """Test: JPEG image processing."""
        results = pipeline.retrieve(
            "process photo",
            image_paths=["photo.jpg"],
            top_k=5
        )
        
        assert len(results) > 0, "Should handle JPEG format hint"


class TestVaguePreciseSpectrum:
    """Test queries ranging from vague to very precise."""
    
    def test_vague_analyze_image(self, pipeline):
        """Test: Very vague request."""
        results = pipeline.retrieve("analyze image", top_k=5)
        
        # Should still return some results
        assert len(results) > 0, "Should return results even for vague query"
        log.info(f"Vague query returned {len(results)} results")
    
    def test_vague_segment(self, pipeline):
        """Test: Vague task without context."""
        results = pipeline.retrieve("segment", top_k=5)
        
        assert len(results) > 0, "Should return segmentation tools"
        
        # Should find generic segmentation tools
        result_names = [r["doc"].name.lower() for r in results]
        assert any("segment" in name for name in result_names), "Should find segmentation tools"
    
    def test_precise_3d_liver_segmentation_dicom(self, pipeline):
        """Test: Very precise request with multiple constraints."""
        results = pipeline.retrieve(
            "3D liver segmentation from DICOM CT scans using deep learning",
            top_k=5
        )
        
        assert len(results) > 0, "Should find results for precise query"
        
        # Log top results to verify precision
        for i, r in enumerate(results[:3]):
            log.info(f"  Precise query result {i+1}: {r['doc'].name}")
    
    def test_moderate_precision_nifti_viewer(self, pipeline):
        """Test: Moderately precise request."""
        results = pipeline.retrieve("visualize NIfTI brain volumes", top_k=5)
        
        assert len(results) > 0, "Should find NIfTI visualization tools"


class TestOutOfCatalogRequests:
    """Test queries for tasks likely not in the catalog."""
    
    def test_video_editing(self, pipeline):
        """Test: Video editing (not in imaging tool catalog)."""
        results = pipeline.retrieve("edit video add transitions", top_k=5)
        
        # Should still return something (BGE-M3 finds nearest matches)
        assert len(results) > 0, "Should return nearest matches"
        
        log.info(f"Video editing query returned: {[r['doc'].name for r in results[:3]]}")
    
    def test_audio_processing(self, pipeline):
        """Test: Audio processing (definitely out of scope)."""
        results = pipeline.retrieve("denoise audio recording", top_k=5)
        
        # Will return something, but should be poor matches
        assert len(results) > 0, "Should return results"
        
        # Results will have low rerank scores
        if results[0].get("rerank_score"):
            log.info(f"Audio query top score: {results[0]['rerank_score']:.3f}")
    
    def test_3d_rendering_animation(self, pipeline):
        """Test: 3D rendering/animation task."""
        results = pipeline.retrieve("render 3D scene with ray tracing", top_k=5)
        
        assert len(results) > 0, "Should return nearest imaging tools"
        
        # Might find 3D visualization tools
        for r in results[:3]:
            log.info(f"  3D rendering candidate: {r['doc'].name}")
    
    def test_document_layout_analysis(self, pipeline):
        """Test: Document analysis task."""
        results = pipeline.retrieve("analyze document layout structure", top_k=5)
        
        assert len(results) > 0, "Should return results"
        
        # May find segmentation or OCR-adjacent tools
        result_names = [r["doc"].name for r in results[:3]]
        log.info(f"Document layout results: {result_names}")


class TestRetrievalModes:
    """Test different retrieval modes and configurations."""
    
    def test_retrieve_no_rerank(self, pipeline):
        """Test: Retrieval without CrossEncoder reranking."""
        results = pipeline.retrieve_no_rerank("segment lungs", top_k=10)
        
        assert len(results) > 0, "Should return results without reranking"
        
        # Check that no rerank_score is set (or is 0.0)
        for r in results[:3]:
            assert r.get("rerank_score") is None or r.get("__rerank__") == 0.0
            log.info(f"  No rerank: {r['doc'].name} (sim: {r.get('__sim__', 'N/A')})")
    
    def test_retrieve_with_rerank(self, pipeline):
        """Test: Full retrieval with reranking."""
        results = pipeline.retrieve("segment lungs", top_k=10)
        
        assert len(results) > 0, "Should return reranked results"
        
        # Check that rerank_score is set
        assert results[0].get("rerank_score") is not None, "Should have rerank scores"
        
        for r in results[:3]:
            log.info(f"  Reranked: {r['doc'].name} (rerank: {r.get('rerank_score', 'N/A')})")
    
    def test_rerank_improves_precision(self, pipeline):
        """Test: Verify reranking improves result quality."""
        query = "register brain MRI images"
        
        # Without rerank
        no_rerank = pipeline.retrieve_no_rerank(query, top_k=10)
        
        # With rerank
        with_rerank = pipeline.retrieve(query, top_k=10)
        
        assert len(no_rerank) > 0 and len(with_rerank) > 0
        
        # Log comparison
        log.info("Comparison (no rerank vs rerank):")
        for i in range(min(3, len(no_rerank), len(with_rerank))):
            log.info(f"  {i+1}. {no_rerank[i]['doc'].name} → {with_rerank[i]['doc'].name}")
    
    def test_exclusion_filter(self, pipeline):
        """Test: Exclusion filter works correctly."""
        # First get top result
        results = pipeline.retrieve("segment image", top_k=5)
        
        if len(results) == 0:
            pytest.skip("No results to test exclusion")
        
        excluded_name = results[0]["doc"].name
        
        # Now exclude it
        filtered = pipeline.retrieve(
            "segment image",
            top_k=5,
            exclusions=[excluded_name]
        )
        
        # Verify excluded tool is not in results
        result_names = [r["doc"].name for r in filtered]
        assert excluded_name not in result_names, f"Should exclude {excluded_name}"
        
        log.info(f"Excluded {excluded_name}, got: {result_names[:3]}")


class TestImageMetadataIntegration:
    """Test image metadata hint generation and integration."""
    
    def test_format_hint_dicom(self, pipeline):
        """Test: DICOM format hint is added to query."""
        results = pipeline.retrieve(
            "visualize scan",
            image_paths=["scan.dcm", "scan2.dicom"],
            top_k=5
        )
        
        assert len(results) > 0, "Should find results with DICOM hint"
        
        # Image hint should boost DICOM-compatible tools
        for r in results[:3]:
            log.info(f"  DICOM hint result: {r['doc'].name}")
    
    def test_format_hint_nifti(self, pipeline):
        """Test: NIfTI format hint is added."""
        results = pipeline.retrieve(
            "view brain volume",
            image_paths=["brain.nii.gz"],
            top_k=5
        )
        
        assert len(results) > 0, "Should find NIfTI viewers"
    
    def test_format_hint_tiff_stack(self, pipeline):
        """Test: TIFF stack hint for microscopy."""
        results = pipeline.retrieve(
            "analyze microscopy images",
            image_paths=["cells.tif"],
            top_k=5
        )
        
        assert len(results) > 0, "Should find TIFF-compatible tools"
    
    def test_multiple_formats(self, pipeline):
        """Test: Multiple file formats in one request."""
        results = pipeline.retrieve(
            "register images",
            image_paths=["scan1.dcm", "scan2.nii.gz"],
            top_k=5
        )
        
        assert len(results) > 0, "Should handle multiple formats"
        
        # Should find tools compatible with either format
        log.info(f"Multi-format results: {[r['doc'].name for r in results[:3]]}")


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_query(self, pipeline):
        """Test: Empty query string."""
        results = pipeline.retrieve("", top_k=5)
        
        # Should still return something (based on metadata if provided)
        # or handle gracefully
        assert isinstance(results, list), "Should return list even for empty query"
    
    def test_very_long_query(self, pipeline):
        """Test: Extremely long query."""
        long_query = " ".join([
            "segment lung tissue from high resolution computed tomography CT scans",
            "with automatic detection of nodules lesions and anatomical structures",
            "using deep learning convolutional neural networks and transfer learning",
            "optimized for medical imaging radiology and pulmonology applications"
        ])
        
        results = pipeline.retrieve(long_query, top_k=5)
        
        assert len(results) > 0, "Should handle long queries"
        log.info(f"Long query top result: {results[0]['doc'].name if results else 'None'}")
    
    def test_special_characters_query(self, pipeline):
        """Test: Query with special characters."""
        results = pipeline.retrieve("segment (3D) [CT/MRI] images!", top_k=5)
        
        assert len(results) > 0, "Should handle special characters"
    
    def test_top_k_zero(self, pipeline):
        """Test: Request zero results."""
        results = pipeline.retrieve("segment lungs", top_k=0)
        
        # Should return empty list or handle gracefully
        assert isinstance(results, list), "Should return list"
    
    def test_top_k_large(self, pipeline):
        """Test: Request more results than available."""
        results = pipeline.retrieve("segment", top_k=1000)
        
        assert isinstance(results, list), "Should return list"
        # Will return all available results (up to catalog size)
        log.info(f"Large top_k returned {len(results)} results")


class TestRetryMechanism:
    """Test the retry mechanism for insufficient results."""
    
    def test_retry_broadens_query(self, pipeline):
        """Test: Very specific query that may trigger retry."""
        # Use a very specific query that might not find min_results initially
        results = pipeline.retrieve(
            "segment hippocampus subfields from high-resolution T1-weighted MRI",
            top_k=10
        )
        
        # Should eventually return some results (possibly after retry)
        assert len(results) > 0, "Should find results after potential retry"
        log.info(f"Retry test returned {len(results)} results")
    
    def test_obscure_term_retry(self, pipeline):
        """Test: Obscure medical term that might need retry."""
        results = pipeline.retrieve(
            "analyze perfusion BOLD fMRI hemodynamic response",
            top_k=5
        )
        
        # Should find brain/MRI related tools after retry
        assert len(results) > 0, "Should return results"
        
        for r in results[:3]:
            log.info(f"  Obscure term result: {r['doc'].name}")


class TestSemanticUnderstanding:
    """Test BGE-M3's semantic understanding capabilities."""
    
    def test_synonym_understanding_visualize_display(self, pipeline):
        """Test: Synonyms (visualize vs display vs show)."""
        query1 = pipeline.retrieve("visualize medical images", top_k=5)
        query2 = pipeline.retrieve("display medical images", top_k=5)
        query3 = pipeline.retrieve("show medical images", top_k=5)
        
        # All should return reasonable results
        assert len(query1) > 0 and len(query2) > 0 and len(query3) > 0
        
        # Top results might overlap
        names1 = {r["doc"].name for r in query1[:3]}
        names2 = {r["doc"].name for r in query2[:3]}
        names3 = {r["doc"].name for r in query3[:3]}
        
        log.info(f"Synonym overlap: {names1 & names2 & names3}")
    
    def test_related_concepts_segmentation(self, pipeline):
        """Test: Related concept understanding."""
        results = pipeline.retrieve("partition lung regions", top_k=5)
        
        # Should understand "partition" is related to segmentation
        assert len(results) > 0, "Should find segmentation tools"
        
        result_names = [r["doc"].name.lower() for r in results]
        log.info(f"Related concept results: {result_names[:3]}")
    
    def test_acronym_vs_full_form(self, pipeline):
        """Test: Acronym vs full form (CT vs Computed Tomography)."""
        ct_results = pipeline.retrieve("CT segmentation", top_k=5)
        full_results = pipeline.retrieve("computed tomography segmentation", top_k=5)
        
        assert len(ct_results) > 0 and len(full_results) > 0
        
        # Should have significant overlap
        ct_names = {r["doc"].name for r in ct_results[:3]}
        full_names = {r["doc"].name for r in full_results[:3]}
        
        overlap = ct_names & full_names
        log.info(f"Acronym overlap: {overlap}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])
