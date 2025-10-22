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
            response = json.dumps({
                'success': True,
                'processedPdf': processed_pdf_base64,
                'message': 'PDF processed successfully'
            })
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
