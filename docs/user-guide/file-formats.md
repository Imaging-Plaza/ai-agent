# Supported File Formats

The AI Imaging Agent supports a wide range of file formats for medical and scientific imaging, as well as general data files.

## Image Formats

### Standard Images

| Format | Extensions | Description |
|--------|-----------|-------------|
| PNG | `.png` | Portable Network Graphics - lossless compression |
| JPEG | `.jpg`, `.jpeg` | Joint Photographic Experts Group - lossy compression |
| WebP | `.webp` | Modern web image format |
| BMP | `.bmp` | Bitmap image file |
| GIF | `.gif` | Graphics Interchange Format |

**Best for**: General photographs, screenshots, web images

### Medical Imaging Formats

#### DICOM

| Format | Extensions | Description |
|--------|-----------|-------------|
| DICOM | `.dcm`, `.dicom` | Digital Imaging and Communications in Medicine |

**Features**:

- Industry standard for medical imaging
- Contains rich metadata (patient info, acquisition parameters)
- Supports multiple modalities (CT, MRI, X-ray, etc.)
- Can store 2D images or 3D volumes

**Metadata Extracted**:

- Patient ID, Study Instance UID
- Modality (CT, MR, CR, DX, etc.)
- Image dimensions and spacing
- Acquisition date/time
- Manufacturer and model

**Example Usage**:
```
Upload a CT DICOM file and ask:
"Segment the lungs from this scan"
```

#### NIfTI

| Format | Extensions | Description |
|--------|-----------|-------------|
| NIfTI | `.nii`, `.nii.gz` | Neuroimaging Informatics Technology Initiative |

**Features**:

- Standard for neuroimaging research
- Supports 3D and 4D (time-series) volumes
- Compact storage with optional gzip compression
- Contains spatial orientation information

**Metadata Extracted**:

- Volume dimensions (x, y, z, time)
- Voxel spacing
- Data type and bit depth
- Orientation matrix

**Example Usage**:
```
Upload a brain MRI NIfTI file:
"Register this brain scan to MNI space"
```

### Scientific Imaging Formats

#### TIFF/TIFF Stacks

| Format | Extensions | Description |
|--------|-----------|-------------|
| TIFF | `.tif`, `.tiff` | Tagged Image File Format |

**Features**:

- Supports multi-page/multi-frame images
- Common in microscopy and scientific imaging
- Can store extensive metadata
- Lossless compression options

**Metadata Extracted**:

- Number of pages/frames (for stacks)
- Dimensions (width, height, channels)
- Color mode (RGB, grayscale, etc.)
- Compression method
- DPI/resolution information

**Example Usage**:
```
Upload a microscopy TIFF stack:
"Analyze cell structures in this z-stack"
```

## Data Formats

<!-- ### Structured Data

| Format | Extensions | Description |
|--------|-----------|-------------|
| CSV | `.csv` | Comma-separated values |
| JSON | `.json` | JavaScript Object Notation |
| XML | `.xml` | Extensible Markup Language |

**Best for**: Metadata, annotations, measurements, structured results

## Media Formats

| Format | Extensions | Description |
|--------|-----------|-------------|
| Audio | `.mp3` | MPEG Audio Layer 3 |
| Video | `.mp4` | MPEG-4 video |

**Note**: Currently supported for upload but limited analysis capabilities. -->

## Format Detection

The agent automatically detects file formats using:

1. **File Extension**: Primary detection method
2. **Magic Bytes**: Header inspection for validation
3. **Content Analysis**: Fallback for ambiguous cases

## Metadata Extraction

### What Gets Extracted

For each uploaded file, the agent extracts:

#### Image Metadata
- **Dimensions**: Width, height, depth (for volumes)
- **Channels**: Grayscale, RGB, RGBA
- **Data Type**: uint8, int16, float32, etc.
- **File Size**: Storage size

#### Medical Image Metadata
- **Modality**: CT, MRI, X-ray, Ultrasound, PET, etc.
- **Patient Info**: Anonymized IDs
- **Study Info**: Study UID, dates
- **Acquisition Parameters**: Slice thickness, spacing, orientation
- **Equipment**: Manufacturer, model, software version

#### Format-Specific Metadata
- **DICOM Tags**: Full DICOM header information
- **NIfTI Header**: Spatial orientation, timing information
- **TIFF Tags**: IFD entries, compression, photometric interpretation

### Why Metadata Matters

Metadata is used for:

1. **Format Matching**: Recommend tools that support your file format
2. **Compatibility Scoring**: Prioritize tools that work with your specific format
3. **Context Understanding**: Help VLM understand image characteristics
4. **Demo Execution**: Ensure tools can process your data

## Preview Generation

### Automatic Conversion

Medical and scientific images are converted to PNG previews for VLM analysis:

| Original Format | Preview Generation |
|----------------|-------------------|
| DICOM (2D) | Single-frame converted to PNG |
| DICOM / NIfTI 3D volumes | Orthogonal 3‑view composite PNG (axial, sagittal, coronal) using middle slices and/or maximum intensity projections (MIPs) |
| NIfTI 4D (time series) | Middle timepoint volume rendered as an orthogonal 3‑view composite (middle slices and/or MIPs) |
| TIFF Stack | Orthogonal 3‑view composite for 3D stacks; otherwise contact sheet or animated GIF preview when appropriate |
| Standard Images | Single-view PNG (content preserved; may be resized/normalized) |

**Important**: Preview generation is for visual analysis only. Original format metadata is preserved and used for compatibility matching.

### Multi-Slice Handling

For 3D volumes, the agent typically builds an orthogonal 3‑view composite preview:

- **Axial**: Horizontal slices (z-axis)
- **Sagittal**: Side view (x-axis)
- **Coronal**: Front view (y-axis)

Each view may combine the middle slice with a maximum intensity projection (MIP) to capture both anatomical context and bright structures. When a 3‑view composite cannot be generated (e.g., unusual stack layout), the agent may fall back to a contact sheet or an animated GIF preview of multiple slices.

## Format Compatibility Matching

### How It Works

The retrieval system adds format tokens to your query:

```
Original query: "segment lungs"
Enhanced query: "segment lungs format:DICOM format:3D"
```

Tools are matched based on:

1. **Direct Format Support**: Tool explicitly supports your format
2. **Format Category**: Tool supports format family (e.g., medical imaging)
3. **Conversion Capability**: Tool can convert from your format

### IO Compatibility Scoring

The VLM considers:

- **Input Format Match**: Can the tool read your file?
- **Output Format**: What format does the tool produce?
- **Dimension Compatibility**: 2D tool for 2D images, 3D for volumes
- **Modality Specificity**: CT tools for CT images, MRI for MRI

## File Size Limits

Default limits (configurable):

| Category | Limit | Notes |
|----------|-------|-------|
| Images | 100 MB | Per file |
| DICOM | 200 MB | Medical images can be larger |
| NIfTI | 500 MB | Volumes can be very large |
| TIFF Stacks | 200 MB | Multi-frame images |
| Other Files | 50 MB | General limit |

!!! warning "Large Files"
    Very large files may take longer to process. Consider downsampling or cropping if possible.

## Unsupported Formats

Currently not supported:

- **Proprietary Formats**: Manufacturer-specific formats (e.g., .PAR/.REC)
- **Video Processing**: Limited video analysis capability
- **Raw Data**: Unformatted binary dumps without headers

## Format Best Practices

!!! tip "Use Standard Formats"
    Stick to standard formats (DICOM, NIfTI, PNG, TIFF) for best tool compatibility.

!!! tip "Include Metadata"
    Use formats that preserve metadata (DICOM, NIfTI) rather than exporting to PNG/JPEG.

!!! tip "Check Compatibility"
    If a tool doesn't work, check the format compatibility in the recommendation metadata.

!!! tip "Convert When Needed"
    Some tools prefer specific formats. Convert using standard tools (ITK-SNAP, 3D Slicer) before upload.

## Example Workflows by Format

### DICOM Workflow
```
1. Upload: chest_ct.dcm
2. Query: "Segment lungs"
3. Agent detects: DICOM, CT modality, 3D volume
4. Results: CT-compatible lung segmentation tools
```

### NIfTI Workflow
```
1. Upload: brain_mri.nii.gz
2. Query: "Skull stripping"
3. Agent detects: NIfTI, 3D volume, likely MRI
4. Results: Brain extraction tools supporting NIfTI
```

### TIFF Stack Workflow
```
1. Upload: microscopy_stack.tif
2. Query: "Cell counting"
3. Agent detects: Multi-frame TIFF, 3D stack
4. Results: Microscopy analysis tools
```

## Next Steps

- Learn about [Understanding Recommendations](recommendations.md)
- Explore [Running Demos](running-demos.md)
- Check [Advanced Features](advanced-features.md)
