# PDF Encryption Tool

A web application that allows users to upload PDF files and process them with character mapping encryption. The processed PDFs can then be downloaded.

## Features

- **PDF Upload**: Drag and drop or click to upload PDF files
- **Character Mapping Encryption**: Uses custom character mapping to encrypt text in PDFs
- **Real-time Processing**: Shows processing status with visual feedback
- **Download Processed PDF**: Download the encrypted PDF file
- **Responsive Design**: Works on desktop and mobile devices

## How It Works

1. **Upload**: Select or drag a PDF file to the upload area
2. **Processing**: The PDF is sent to a serverless function that:
   - Extracts text from the PDF
   - Applies character mapping encryption
   - Redacts original text and overlays encrypted text
   - Returns the processed PDF
3. **Download**: Download the processed PDF with encrypted content

## Character Mapping

The encryption uses a custom character mapping system:
- **Uppercase letters**: Mapped to different uppercase letters
- **Lowercase letters**: Mapped to different lowercase letters  
- **Special characters**: Mapped to specific characters
- **Ligatures**: Expanded to their component characters

## Deployment

### Prerequisites
- Node.js installed
- Vercel CLI installed (`npm install -g vercel`)

### Deploy to Vercel
```bash
# Install dependencies
npm install

# Deploy to Vercel
npm run deploy
```

Or use the deploy script:
```bash
chmod +x deploy.sh
./deploy.sh
```

## Local Development

To run locally with Vercel:
```bash
npm run dev
```

## File Structure

```
├── index.html          # Main HTML file
├── script.js           # Frontend JavaScript
├── style.css           # CSS styles
├── api/
│   └── process-pdf/
│       └── index.py    # Serverless function for PDF processing
├── requirements.txt    # Python dependencies
├── vercel.json        # Vercel configuration
├── package.json       # Node.js dependencies
└── deploy.sh          # Deployment script
```

## Technical Details

- **Frontend**: Vanilla JavaScript with modern ES6+ features
- **Backend**: Python serverless function using PyMuPDF
- **Hosting**: Vercel (supports both static files and serverless functions)
- **Processing**: Character-by-character text replacement with font preservation

## Browser Support

- Chrome/Chromium (recommended)
- Firefox
- Safari
- Edge

## Security Note

This tool processes PDFs on the server side. The processed files are not stored permanently and are only returned to the client.
