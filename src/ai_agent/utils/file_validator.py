from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import zipfile
import imghdr
import nibabel as nib

class FileValidator:
    # Supported file extensions and their descriptions
    SUPPORTED_EXTENSIONS = {
        # Images
        '.jpg': 'JPEG image',
        '.jpeg': 'JPEG image',
        '.png': 'PNG image',
        '.tif': 'TIFF image',
        '.tiff': 'TIFF image',
        # Medical formats
        '.nii': 'NIfTI image',
        '.nii.gz': 'Compressed NIfTI image',
        '.dcm': 'DICOM image',
        # Directories
        'dir': 'Directory (for DICOM series)',
        # Archives
        '.zip': 'ZIP archive (for DICOM series)'
    }

    @classmethod
    def validate_files(cls, paths: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate a list of file paths.
        Returns (valid_paths, error_messages)
        """
        valid_paths = []
        errors = []

        for path in paths:
            if not path:
                continue
                
            try:
                p = Path(path)
                
                # Check if path exists
                if not p.exists():
                    errors.append(f"File not found: {path}")
                    continue

                # Handle ZIP files first (potential DICOM archives)
                if p.suffix.lower() == '.zip':
                    try:
                        with zipfile.ZipFile(p) as zf:
                            # Check if zip contains .dcm files
                            has_dicom = any(name.lower().endswith('.dcm') for name in zf.namelist())
                            if has_dicom:
                                valid_paths.append(path)
                                continue
                            else:
                                errors.append(f"ZIP file contains no DICOM files: {path}")
                                continue
                    except Exception as e:
                        errors.append(f"Invalid ZIP file {path}: {str(e)}")
                        continue

                # Handle directories
                if p.is_dir():
                    # Use _is_dicom_file for better validation
                    if any(cls._is_dicom_file(f) for f in p.glob('*')):
                        valid_paths.append(path)
                    else:
                        errors.append(f"Directory contains no valid DICOM files: {path}")
                    continue

                # For single DICOM files, use proper validation
                if p.suffix.lower() == '.dcm' or cls._is_dicom_file(p):
                    valid_paths.append(path)
                    continue

                # Check file extension - Fix NIfTI handling
                ext = p.suffix.lower()
                name = p.name.lower()
                
                # Handle compound extensions
                if name.endswith('.nii.gz'):
                    ext = '.nii.gz'
                elif ext == '.gz' and p.stem.lower().endswith('.nii'):
                    ext = '.nii.gz'
                elif name.endswith('.nii'):
                    ext = '.nii'

                # Special validation for NIfTI files
                if ext in ['.nii', '.nii.gz']:
                    try:
                        img = nib.load(str(p))
                        img.header  # Try to access header to verify file
                        valid_paths.append(path)
                    except Exception as e:
                        errors.append(f"Invalid NIfTI file {path}: {str(e)}")
                    continue
                
                # Check file extension
                if ext not in cls.SUPPORTED_EXTENSIONS:
                    errors.append(
                        f"Unsupported file type: {path}\n"
                        f"Supported formats: {', '.join(cls.SUPPORTED_EXTENSIONS.keys())}"
                    )
                    continue

                # Additional validation for image files
                if ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                    img_type = imghdr.what(path)
                    if not img_type:
                        errors.append(f"Invalid or corrupted image file: {path}")
                        continue

                valid_paths.append(path)

            except Exception as e:
                errors.append(f"Error processing {path}: {str(e)}")

        return valid_paths, errors

    @classmethod
    def _is_dicom_file(cls, path: Path) -> bool:
        """Check if file is DICOM by extension or DICM magic number"""
        if path.suffix.lower() == '.dcm':
            return True
        try:
            with open(path, 'rb') as f:
                f.seek(128)
                return f.read(4) == b'DICM'
        except Exception:
            return False

    @classmethod
    def get_supported_formats_md(cls) -> str:
        """Return markdown-formatted string of supported formats"""
        md = "**Supported File Formats:**\n\n"
        for ext, desc in cls.SUPPORTED_EXTENSIONS.items():
            md += f"- `{ext}`: {desc}\n"
        return md