import json
import base64
import tempfile
import os
import sys
import importlib.util
from http.server import BaseHTTPRequestHandler

# Add the current directory to Python path for Vercel
sys.path.append(os.path.dirname(__file__))

# Import the module with hyphen in filename
spec = importlib.util.spec_from_file_location("encrypt_module", os.path.join(os.path.dirname(__file__), "EncTestNewTestF-2.py"))
encrypt_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(encrypt_module)
process_pdf = encrypt_module.process_pdf

class handler(BaseHTTPRequestHandler):
    """
    Vercel serverless function to process PDF files with character mapping encryption
    """
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_POST(self):
        """Handle POST requests for PDF processing"""
        try:
            # Set CORS headers
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            # Parse the request body
            body = json.loads(post_data.decode('utf-8'))
            
            # Get PDF data and filename
            pdf_data = body.get('pdfData')
            filename = body.get('filename', 'document.pdf')
            
            if not pdf_data:
                response = json.dumps({'error': 'No PDF data provided'})
                self.wfile.write(response.encode('utf-8'))
                return
            
            # Decode base64 PDF data
            try:
                pdf_bytes = base64.b64decode(pdf_data)
            except Exception as e:
                response = json.dumps({'error': f'Invalid base64 PDF data: {str(e)}'})
                self.wfile.write(response.encode('utf-8'))
                return
            
            # Validate PDF data
            if not pdf_bytes.startswith(b'%PDF-'):
                response = json.dumps({'error': 'Invalid PDF data: File does not start with PDF header'})
                self.wfile.write(response.encode('utf-8'))
                return
            
            # Create temporary files
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as input_file:
                input_file.write(pdf_bytes)
                input_pdf_path = input_file.name
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as output_file:
                output_pdf_path = output_file.name
            
            # Get the font file path (in the same directory as this script)
            font_path = os.path.join(os.path.dirname(__file__), 'Supertest.ttf')
            
            # Process the PDF
            try:
                process_pdf(input_pdf_path, font_path, output_pdf_path)
                
                # Apply adversarial attacks to the processed PDF if requested
                adversarial_attacks_applied = False
                adversarial_glyphs_applied = False
                
                if body.get('applyAdversarialAttacks', False) or body.get('applyAdversarialGlyphs', False):
                    # Import ML dependencies - if they fail, the whole request fails
                    import sys
                    sys.path.append('/Users/tyler/Cloak')
                    from adversarial_attacks import AdversarialAttackEngine
                    from adversarial_utils import ImageProcessor
                    from ocr_model import CharacterClassifier
                    import fitz  # PyMuPDF
                    from PIL import Image
                    import io
                    
                    print("🚀 Initializing full ML adversarial attack engines...")
                    # Initialize attack engines
                    attack_engine = AdversarialAttackEngine(model_name='resnet50')
                    ocr_classifier = CharacterClassifier()
                    
                    # Open the processed PDF
                    doc = fitz.open(output_pdf_path)
                    
                    # Apply adversarial attacks to images
                    if body.get('applyAdversarialAttacks', False):
                        print("🖼️ Applying advanced adversarial attacks to images...")
                        for page_num in range(len(doc)):
                            page = doc[page_num]
                            # Get images from the page
                            image_list = page.get_images()
                            
                            for img_index, img in enumerate(image_list):
                                # Get image data
                                xref = img[0]
                                pix = fitz.Pixmap(doc, xref)
                                
                                if pix.n - pix.alpha < 4:  # GRAY or RGB
                                    # Convert to PIL Image
                                    img_data = pix.tobytes("png")
                                    pil_image = Image.open(io.BytesIO(img_data))
                                    
                                    # Convert to tensor and apply adversarial attack
                                    image_tensor = attack_engine.load_image_from_pil(pil_image)
                                    orig_pred, _ = attack_engine.predict(image_tensor)
                                    
                                    # Create adversarial example
                                    adversarial_image, attack_info = attack_engine.create_adversarial_example(
                                        image_tensor, orig_pred, attack_method='FGSM', epsilons=0.1
                                    )
                                    
                                    # Convert back to PIL and replace in PDF
                                    adv_pil = ImageProcessor.tensor_to_pil(adversarial_image)
                                    
                                    # Convert back to bytes
                                    img_buffer = io.BytesIO()
                                    adv_pil.save(img_buffer, format='PNG')
                                    img_bytes = img_buffer.getvalue()
                                    
                                    # Replace image in PDF
                                    new_pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, adv_pil.width, adv_pil.height), img_bytes)
                                    page.replace_image(xref, pixmap=new_pix)
                                    
                                    adversarial_attacks_applied = True
                                
                                pix = None
                    
                    # Apply adversarial attacks to text glyphs if requested
                    if body.get('applyAdversarialGlyphs', False):
                        print("🔤 Applying advanced adversarial attacks to text glyphs...")
                        for page_num in range(len(doc)):
                            page = doc[page_num]
                            
                            # Get all text from the page
                            text_dict = page.get_text("dict")
                            
                            for block in text_dict["blocks"]:
                                if block["type"] == 0:  # Text block
                                    for line in block["lines"]:
                                        for span in line["spans"]:
                                            text = span["text"]
                                            if text.strip():  # Skip empty text
                                                # Apply adversarial attacks to each character
                                                adversarial_results = attack_engine.apply_adversarial_to_text(
                                                    text, font_path, attack_method='FGSM', epsilons=0.05
                                                )
                                                
                                                # Replace text with adversarial glyphs
                                                self._replace_text_with_adversarial_glyphs(
                                                    page, span, adversarial_results, font_path
                                                )
                                                
                                                adversarial_glyphs_applied = True
                    
                    if adversarial_attacks_applied or adversarial_glyphs_applied:
                        # Save the modified PDF
                        doc.save(output_pdf_path)
                    doc.close()
                
            except Exception as e:
                # Clean up temporary files
                if os.path.exists(input_pdf_path):
                    os.unlink(input_pdf_path)
                if os.path.exists(output_pdf_path):
                    os.unlink(output_pdf_path)
                
                response = json.dumps({
                    'success': False,
                    'error': f'PDF processing failed: {str(e)}'
                })
                self.wfile.write(response.encode('utf-8'))
                return
            
            # Read the processed PDF
            with open(output_pdf_path, 'rb') as f:
                processed_pdf_bytes = f.read()
            
            # Encode the processed PDF as base64
            processed_pdf_base64 = base64.b64encode(processed_pdf_bytes).decode('utf-8')
            
            # Clean up temporary files
            os.unlink(input_pdf_path)
            os.unlink(output_pdf_path)
            
            # Return success response
            response_data = {
                'success': True,
                'processedPdf': processed_pdf_base64,
                'message': 'PDF processed successfully'
            }
            
            if adversarial_attacks_applied:
                response_data['message'] = 'PDF processed successfully with adversarial attacks applied'
                response_data['adversarialAttacksApplied'] = True
            
            if adversarial_glyphs_applied:
                response_data['message'] = 'PDF processed successfully with adversarial glyph attacks applied'
                response_data['adversarialGlyphsApplied'] = True
            
            if adversarial_attacks_applied and adversarial_glyphs_applied:
                response_data['message'] = 'PDF processed successfully with both image and glyph adversarial attacks applied'
                response_data['adversarialAttacksApplied'] = True
                response_data['adversarialGlyphsApplied'] = True
            
            response = json.dumps(response_data)
            self.wfile.write(response.encode('utf-8'))
            
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            self.send_response(500)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            response = json.dumps({
                'success': False,
                'error': f'Error processing PDF: {str(e)}'
            })
            self.wfile.write(response.encode('utf-8'))
    
    def do_GET(self):
        """Handle GET requests - return method not allowed"""
        self.send_response(405)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        
        response = json.dumps({'error': 'Method not allowed'})
        self.wfile.write(response.encode('utf-8'))
    
    def _replace_text_with_adversarial_glyphs(self, page, span, adversarial_results, font_path):
        """
        Replace text span with adversarial glyphs.
        
        Args:
            page: PDF page object
            span: Text span dictionary
            adversarial_results: List of (character, adversarial_glyph, attack_info) tuples
            font_path: Path to font file
        """
        try:
            # Get span position and properties
            bbox = span["bbox"]
            font_size = span["size"]
            font_name = span["font"]
            
            # Create a new text writer for this span
            tw = fitz.TextWriter(page.rect)
            
            # Load font
            font = fitz.Font(fontfile=font_path)
            
            # Position for first character
            x, y = bbox[0], bbox[3]  # Start at left edge, baseline
            
            for i, (char, adversarial_glyph, attack_info) in enumerate(adversarial_results):
                if adversarial_glyph is not None:
                    # Create a temporary image from the adversarial glyph
                    import io
                    img_buffer = io.BytesIO()
                    adversarial_glyph.save(img_buffer, format='PNG')
                    img_bytes = img_buffer.getvalue()
                    
                    # Insert the adversarial glyph as an image
                    img_rect = fitz.Rect(x, y - font_size, x + font_size, y)
                    page.insert_image(img_rect, stream=img_bytes)
                    
                    # Move to next character position
                    x += font_size * 0.6  # Approximate character width
                else:
                    # Use original character for whitespace
                    tw.append(
                        pos=(x, y),
                        text=char,
                        font=font,
                        fontsize=font_size
                    )
                    x += font_size * 0.6
            
            # Write any remaining text
            tw.write_text(page, overlay=True)
            
        except Exception as e:
            print(f"Warning: Failed to replace text with adversarial glyphs: {e}")
            # Fallback: keep original text
