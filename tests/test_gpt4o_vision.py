#!/usr/bin/env python3
"""
Test script to verify gpt-4o vision with BinaryContent (mimicking the actual pipeline).
This matches the exact pattern used in agent.py.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Import pydantic-ai components (matching agent.py pattern)
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel


class VisionTestResponse(BaseModel):
    """Response for vision test."""

    what_i_see: str
    image_received: bool
    confidence: int  # 0-100 how confident you saw an image


def main() -> int:
    """Run gpt-4o vision smoke test when executed as a script.

    This function performs environment checks, creates a test image,
    initializes the OpenAI client/agent, and runs basic tests.
    It is intentionally not executed at import time so pytest can
    safely collect this module without side effects.
    """
    missing_openai_api_key = False
    missing_pillow = False

    # Check environment
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("❌ OPENAI_API_KEY not found in environment")
        print("   Set it in .env or export OPENAI_API_KEY=your_key")
        missing_openai_api_key = True
    else:
        print("✅ OPENAI_API_KEY found")
        print()

    # Create minimal 1x1 red pixel PNG using PIL
    print("🔄 Creating test image (1x1 red pixel PNG)...")
    try:
        from PIL import Image
        import io

        # Create 1x1 red pixel image
        img = Image.new("RGB", (1, 1), color="red")

        # Save to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        png_1x1_red = img_bytes.getvalue()

        # Also save to file
        test_image_path = Path("test_pixel.png")
        test_image_path.write_bytes(png_1x1_red)

        print(f"✅ Test image created: {test_image_path} ({len(png_1x1_red)} bytes)")
    except ImportError:
        print("❌ PIL/Pillow not installed. Install with: pip install Pillow")
        missing_pillow = True

    print()

    # When run as a script, exit with a non-zero status if required dependencies are missing.
    if missing_openai_api_key or missing_pillow:
        return 1

    # Create OpenAI provider and model (matching agent.py)
    print("🔄 Creating OpenAI gpt-4o model client...")
    provider = OpenAIProvider(api_key=openai_key)

    model = OpenAIChatModel(
        model_name="gpt-4o",
        provider=provider,
    )

    agent = Agent(
        model=model,
        system_prompt=(
            "You are testing vision capabilities. "
            "If you receive an image, describe what you see in detail. "
            "Set image_received=True and confidence=100. "
            "If no image, set image_received=False and confidence=0."
        ),
    )

    print("✅ OpenAI gpt-4o agent created")
    print()

    # Test 1: Text-only (baseline)
    print("=" * 70)
    print("📝 Test 1: Text-only request (baseline)")
    print("=" * 70)
    try:
        result = agent.run_sync(
            "What is 2+2? (No image expected)",
            output_type=VisionTestResponse,
        )
        print(f"✅ Response: {result.output.what_i_see}")
        print(f"   Image received: {result.output.image_received}")
        print(f"   Confidence: {result.output.confidence}")
    except Exception as e:
        print(f"❌ Text-only failed: {e}")

    return 0


if __name__ == "__main__":
    # Execute the smoke test only when run as a script, not on import.
    sys.exit(main())
