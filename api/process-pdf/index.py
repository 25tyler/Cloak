import json
import base64
import tempfile
import os
import sys

# Add the current directory to Python path for Vercel
sys.path.append(os.path.dirname(__file__))

from EncTestNewTestF import process_pdf

def handler(request):
    """
    Vercel serverless function to process PDF files with character mapping encryption
    """
    # Handle CORS preflight requests
    if request.method == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type',
            },
            'body': ''
        }
    
    # Only allow POST requests
    if request.method != 'POST':
        return {
            'statusCode': 405,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({'error': 'Method not allowed'})
        }
    
    try:
        # Parse the request body
        body = json.loads(request.body)
        
        # Get PDF data and filename
        pdf_data = body.get('pdfData')
        filename = body.get('filename', 'document.pdf')
        
        if not pdf_data:
            return {
                'statusCode': 400,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Content-Type': 'application/json'
                },
                'body': json.dumps({'error': 'No PDF data provided'})
            }
        
        # Decode base64 PDF data
        pdf_bytes = base64.b64decode(pdf_data)
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as input_file:
            input_file.write(pdf_bytes)
            input_pdf_path = input_file.name
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as output_file:
            output_pdf_path = output_file.name
        
        # Get the font file path (in the same directory as this script)
        font_path = os.path.join(os.path.dirname(__file__), 'Supertest.ttf')
        
        # Process the PDF
        process_pdf(input_pdf_path, font_path, output_pdf_path)
        
        # Read the processed PDF
        with open(output_pdf_path, 'rb') as f:
            processed_pdf_bytes = f.read()
        
        # Encode the processed PDF as base64
        processed_pdf_base64 = base64.b64encode(processed_pdf_bytes).decode('utf-8')
        
        # Clean up temporary files
        os.unlink(input_pdf_path)
        os.unlink(output_pdf_path)
        
        # Return success response
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'success': True,
                'processedPdf': processed_pdf_base64,
                'message': 'PDF processed successfully'
            })
        }
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'success': False,
                'error': f'Error processing PDF: {str(e)}'
            })
        }
