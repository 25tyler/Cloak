#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT Vision Adversarial Attack System
===================================

This module uses GPT Vision to iteratively test and modify individual pixels
in text glyphs until they are misclassified. Much lighter than traditional
ML adversarial attacks and potentially more effective.
"""

import openai
import base64
import io
import random
import time
from PIL import Image, ImageDraw
from typing import Tuple, List, Dict, Optional
import fitz


class GPTAdversarialAttacker:
    """
    Adversarial attack system using GPT Vision to test and modify glyphs.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the GPT adversarial attacker.
        
        Args:
            api_key: OpenAI API key (if None, will try to get from environment)
        """
        if api_key:
            openai.api_key = api_key
        else:
            # Try to get from environment
            import os
            openai.api_key = os.getenv('OPENAI_API_KEY')
        
        if not openai.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string for GPT Vision."""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    
    def test_glyph_with_gpt(self, glyph_image: Image.Image) -> Tuple[str, float]:
        """
        Test a glyph image with GPT Vision to see what character it recognizes.
        
        Args:
            glyph_image: PIL Image of the glyph
            
        Returns:
            Tuple of (recognized_character, confidence_score)
        """
        try:
            # Convert image to base64
            img_base64 = self.image_to_base64(glyph_image)
            
            # Create GPT Vision request
            response = openai.ChatCompletion.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What character do you see in this image? Respond with just the single character you recognize, nothing else."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1,
                temperature=0.1
            )
            
            recognized_char = response.choices[0].message.content.strip()
            
            # For now, return a simple confidence (in a real implementation, 
            # you might want to ask GPT for confidence or use multiple queries)
            confidence = 0.8  # Placeholder confidence
            
            return recognized_char, confidence
            
        except Exception as e:
            print(f"Error testing glyph with GPT: {e}")
            return "?", 0.0
    
    def modify_pixel(self, image: Image.Image, x: int, y: int, modification_type: str = "flip") -> Image.Image:
        """
        Modify a single pixel in the image.
        
        Args:
            image: Original PIL Image
            x, y: Pixel coordinates
            modification_type: Type of modification ("flip", "brighten", "darken")
            
        Returns:
            Modified PIL Image
        """
        # Create a copy to avoid modifying original
        modified = image.copy()
        pixels = modified.load()
        
        if x >= modified.width or y >= modified.height or x < 0 or y < 0:
            return modified
        
        current_pixel = pixels[x, y]
        
        if modification_type == "flip":
            # Flip the pixel value (0->255, 255->0, etc.)
            if isinstance(current_pixel, tuple):
                # RGB image
                new_pixel = tuple(255 - val for val in current_pixel)
            else:
                # Grayscale image
                new_pixel = 255 - current_pixel
        elif modification_type == "brighten":
            if isinstance(current_pixel, tuple):
                new_pixel = tuple(min(255, val + 50) for val in current_pixel)
            else:
                new_pixel = min(255, current_pixel + 50)
        elif modification_type == "darken":
            if isinstance(current_pixel, tuple):
                new_pixel = tuple(max(0, val - 50) for val in current_pixel)
            else:
                new_pixel = max(0, current_pixel - 50)
        
        pixels[x, y] = new_pixel
        return modified
    
    def create_adversarial_glyph(self, 
                                original_glyph: Image.Image,
                                target_character: str = None,
                                max_iterations: int = 50,
                                max_pixel_changes: int = 100) -> Tuple[Image.Image, Dict]:
        """
        Create an adversarial version of a glyph using GPT Vision feedback.
        
        Args:
            original_glyph: Original glyph image
            target_character: Character to try to make it look like (random if None)
            max_iterations: Maximum number of GPT queries
            max_pixel_changes: Maximum number of pixels to modify
            
        Returns:
            Tuple of (adversarial_glyph, attack_info)
        """
        print(f"🎯 Creating adversarial glyph with GPT Vision...")
        
        # Get original recognition
        original_char, original_conf = self.test_glyph_with_gpt(original_glyph)
        print(f"📖 Original character recognized: '{original_char}' (confidence: {original_conf:.2f})")
        
        # Set target character
        if target_character is None:
            # Choose a random different character
            different_chars = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789" 
                             if c != original_char]
            target_character = random.choice(different_chars) if different_chars else "X"
        
        print(f"🎯 Target character: '{target_character}'")
        
        # Start with original image
        current_image = original_glyph.copy()
        best_image = current_image.copy()
        best_score = 0
        
        attack_info = {
            'original_character': original_char,
            'original_confidence': original_conf,
            'target_character': target_character,
            'iterations': 0,
            'pixels_changed': 0,
            'success': False,
            'final_character': original_char,
            'final_confidence': original_conf
        }
        
        # Iterative pixel modification
        for iteration in range(max_iterations):
            print(f"🔄 Iteration {iteration + 1}/{max_iterations}")
            
            # Test current image
            current_char, current_conf = self.test_glyph_with_gpt(current_image)
            
            # Check if we've achieved the target
            if current_char == target_character:
                print(f"✅ Success! Achieved target character '{target_character}'")
                attack_info['success'] = True
                attack_info['final_character'] = current_char
                attack_info['final_confidence'] = current_conf
                break
            
            # If we've changed too many pixels, stop
            if attack_info['pixels_changed'] >= max_pixel_changes:
                print(f"⚠️ Reached maximum pixel changes ({max_pixel_changes})")
                break
            
            # Try modifying a random pixel
            x = random.randint(0, current_image.width - 1)
            y = random.randint(0, current_image.height - 1)
            modification_type = random.choice(["flip", "brighten", "darken"])
            
            # Create modified version
            modified_image = self.modify_pixel(current_image, x, y, modification_type)
            
            # Test the modified version
            modified_char, modified_conf = self.test_glyph_with_gpt(modified_image)
            
            # If it's better (closer to target or different from original), keep it
            if (modified_char != original_char and modified_char != current_char) or \
               (modified_char == target_character):
                current_image = modified_image
                attack_info['pixels_changed'] += 1
                print(f"📝 Modified pixel ({x},{y}) -> '{modified_char}' (confidence: {modified_conf:.2f})")
            
            # Keep track of best result
            if modified_char == target_character:
                best_image = modified_image.copy()
                best_score = modified_conf
            elif modified_char != original_char and best_score == 0:
                best_image = modified_image.copy()
                best_score = modified_conf
            
            attack_info['iterations'] = iteration + 1
            
            # Add small delay to avoid rate limiting
            time.sleep(0.1)
        
        # Final test
        final_char, final_conf = self.test_glyph_with_gpt(current_image)
        attack_info['final_character'] = final_char
        attack_info['final_confidence'] = final_conf
        
        print(f"🏁 Final result: '{final_char}' (confidence: {final_conf:.2f})")
        print(f"📊 Changed {attack_info['pixels_changed']} pixels in {attack_info['iterations']} iterations")
        
        return current_image, attack_info
    
    def apply_adversarial_to_text(self, 
                                 text: str,
                                 font_path: str,
                                 max_iterations: int = 30,
                                 max_pixel_changes: int = 50) -> List[Tuple[str, Image.Image, Dict]]:
        """
        Apply adversarial attacks to each character in text using GPT Vision.
        
        Args:
            text: Input text string
            font_path: Path to font file
            max_iterations: Max iterations per character
            max_pixel_changes: Max pixel changes per character
            
        Returns:
            List of (character, adversarial_glyph, attack_info) tuples
        """
        results = []
        
        # Load font
        font = fitz.Font(fontfile=font_path)
        
        for char in text:
            if char.isspace():
                # Skip whitespace characters
                results.append((char, None, None))
                continue
            
            try:
                print(f"🔤 Processing character: '{char}'")
                
                # Create glyph image for the character
                glyph_image = self._create_glyph_image(char, font)
                
                # Apply adversarial attack
                adversarial_glyph, attack_info = self.create_adversarial_glyph(
                    glyph_image, 
                    max_iterations=max_iterations,
                    max_pixel_changes=max_pixel_changes
                )
                
                results.append((char, adversarial_glyph, attack_info))
                
            except Exception as e:
                print(f"⚠️ Failed to create adversarial glyph for '{char}': {e}")
                results.append((char, None, None))
        
        return results
    
    def _create_glyph_image(self, char: str, font: fitz.Font) -> Image.Image:
        """Create a PIL Image of a character using the specified font."""
        # Create a temporary document to render the character
        doc = fitz.open()
        page = doc.new_page(width=100, height=100)
        
        # Set font and size
        font_size = 48
        
        # Insert the character
        page.insert_text(
            (10, 60),  # Position
            char,
            fontsize=font_size,
            font=font
        )
        
        # Get the character's bounding box
        text_dict = page.get_text("dict")
        char_rect = None
        
        for block in text_dict["blocks"]:
            if block["type"] == 0:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        if char in span["text"]:
                            char_rect = fitz.Rect(span["bbox"])
                            break
                    if char_rect:
                        break
                if char_rect:
                    break
        
        if char_rect:
            # Crop the page to the character
            char_rect = char_rect.inflate(5)  # Add some padding
            char_pixmap = page.get_pixmap(clip=char_rect)
            char_image = Image.frombytes("RGB", [char_pixmap.width, char_pixmap.height], char_pixmap.samples)
        else:
            # Fallback: create a simple character image
            char_image = Image.new('L', (64, 64), 255)
        
        doc.close()
        return char_image


# Example usage
if __name__ == "__main__":
    # Initialize the attacker
    attacker = GPTAdversarialAttacker()
    
    # Test with a simple character
    from PIL import Image, ImageDraw, ImageFont
    
    # Create a simple test image
    test_image = Image.new('L', (64, 64), 255)
    draw = ImageDraw.Draw(test_image)
    
    # Draw a simple 'A'
    draw.text((20, 20), "A", fill=0)
    
    # Apply adversarial attack
    adversarial_image, attack_info = attacker.create_adversarial_glyph(test_image)
    
    print(f"Attack info: {attack_info}")
    
    # Save results
    test_image.save("original_glyph.png")
    adversarial_image.save("adversarial_glyph.png")
    print("Saved original and adversarial glyphs!")
