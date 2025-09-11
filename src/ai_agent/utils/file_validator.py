from pathlib import Path
from typing import List, Tuple
import os
import imghdr

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

                # Handle directories
                if p.is_dir():
                    # Check if directory contains supported files
                    if any(f.suffix.lower() in ['.dcm'] for f in p.glob('*')):
                        valid_paths.append(path)
                    else:
                        errors.append(f"Directory contains no supported DICOM files: {path}")
                    continue

                # Check file extension
                ext = p.suffix.lower()
                if p.name.endswith('.nii.gz'):
                    ext = '.nii.gz'

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
    def get_supported_formats_md(cls) -> str:
        """Return markdown-formatted string of supported formats"""
        md = "**Supported File Formats:**\n\n"
        for ext, desc in cls.SUPPORTED_EXTENSIONS.items():
            md += f"- `{ext}`: {desc}\n"
        return md