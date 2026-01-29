#!/usr/bin/env python3
"""
Test script to check if EPFL openai/gpt-oss-120b model supports vision/images.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Check environment
epfl_key = os.getenv("EPFL_API_KEY")
if not epfl_key:
    print("❌ EPFL_API_KEY not found in environment")
    print("   Set it in .env or export EPFL_API_KEY=your_key")
    sys.exit(1)

print("✅ EPFL_API_KEY found")
print()

# Test with a simple image
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.messages import ImageUrl
from pydantic import BaseModel

class SimpleResponse(BaseModel):
    """Simple response for testing."""
    description: str
    has_image: bool

# Create EPFL provider and model
print("🔄 Creating EPFL model client...")
provider = OpenAIProvider(
    base_url="https://inference.rcp.epfl.ch/v1",
    api_key=epfl_key,
)

model = OpenAIChatModel(
    model_name="openai/gpt-oss-120b",
    provider=provider,
)

agent = Agent(
    model=model,
    system_prompt="You are a helpful assistant. If you receive an image, describe what you see. If no image, say so.",
)

print("✅ EPFL agent created")
print()

# Test 1: Text-only
print("📝 Test 1: Text-only request...")
try:
    result = agent.run_sync(
        "What is 2+2?",
        output_type=SimpleResponse,
    )
    print(f"✅ Text-only works: {result.output.description}")
except Exception as e:
    print(f"❌ Text-only failed: {e}")

print()

# Test 2: With image
print("📝 Test 2: Multimodal request with image...")
# Create a minimal 1x1 red pixel PNG data URL
red_pixel = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

try:
    result = agent.run_sync(
        [
            "Describe what you see in this image.",
            ImageUrl(red_pixel, media_type="image/png", vendor_metadata={"detail": "high"}),
        ],
        output_type=SimpleResponse,
    )
    print(f"✅ Multimodal request completed")
    print(f"   Response: {result.output.description}")
    print(f"   Model detected image: {result.output.has_image}")
    
    # Check for negative responses indicating image not seen
    negative_phrases = ["no image", "not attached", "no picture", "can't see", "cannot see", "didn't receive"]
    response_lower = result.output.description.lower()
    
    if result.output.has_image and not any(phrase in response_lower for phrase in negative_phrases):
        print()
        print("✅ SUCCESS: openai/gpt-oss-120b SUPPORTS vision!")
        print("   The model received and processed the image.")
    else:
        print()
        print("❌ FAILED: openai/gpt-oss-120b does NOT support vision")
        print("   The model accepted the API call but ignored the image.")
        print("   Response indicates no image was seen.")
        
except Exception as e:
    print(f"❌ Multimodal request failed: {e}")
    print()
    print("⚠️  LIKELY ISSUE: openai/gpt-oss-120b does NOT support vision")
    print("   The EPFL model may be text-only.")
    print()
    print("Solutions:")
    print("  1. Use OpenAI's gpt-4o (supports vision)")
    print("  2. Check EPFL docs for vision-capable models")
    print("  3. Ask EPFL if gpt-oss-120b supports multimodal inputs")

print()
print("=" * 70)
print("SUMMARY:")
print("  - EPFL endpoint: https://inference.rcp.epfl.ch/v1")
print("  - Model: openai/gpt-oss-120b")
print("  - This is NOT OpenAI API (so OpenAI billing shows 0 images)")
print("  - Run this test to check if EPFL model supports vision")
print("=" * 70)
