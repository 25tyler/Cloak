// Global variable to store the uploaded PDF file and processed PDF
let uploadedFile = null;
let processedPdfData = null;

// DOM elements
const pdfInput = document.getElementById('pdfInput');
const fileInfo = document.getElementById('fileInfo');
const processingStatus = document.getElementById('processingStatus');
const downloadBtn = document.getElementById('downloadBtn');
const downloadInfo = document.getElementById('downloadInfo');
const previewContainer = document.getElementById('previewContainer');

// Check if DOM elements are loaded
console.log('PDF Input element:', pdfInput);
console.log('File Info element:', fileInfo);
console.log('Processing Status element:', processingStatus);
console.log('Download Button element:', downloadBtn);
console.log('Download Info element:', downloadInfo);
console.log('Preview Container element:', previewContainer);

// Event listener for file input change
if (pdfInput) {
    pdfInput.addEventListener('change', handleFileUpload);
    console.log('File input change listener added');
} else {
    console.error('PDF input element not found!');
}

// Event listener for download button
if (downloadBtn) {
    downloadBtn.addEventListener('click', handleDownload);
    console.log('Download button click listener added');
} else {
    console.error('Download button element not found!');
}


/**
 * Handles file upload when user selects a PDF file
 * This function is called when the file input changes
 */
async function handleFileUpload(event) {
    console.log('handleFileUpload called');
    console.log('Event:', event);
    console.log('Event target:', event.target);
    console.log('Files:', event.target.files);
    
    // Get the selected file from the input
    const file = event.target.files[0];
    console.log('Selected file:', file);
    
    // Check if a file was selected
    if (!file) {
        console.log('No file selected, resetting state');
        resetFileState();
        return;
    }
    
    console.log('File details:', {
        name: file.name,
        type: file.type,
        size: file.size
    });
    
    // Check if the file is a PDF
    if (file.type !== 'application/pdf') {
        console.log('File is not a PDF:', file.type);
        alert('Please select a PDF file.');
        resetFileState();
        return;
    }
    
    console.log('PDF file validated, proceeding with upload');
    
    // Store the file globally
    uploadedFile = file;
    
    // Display file information
    displayFileInfo(file);
    
    // Show processing status
    showProcessingStatus('Processing PDF...', 'processing');
    
    try {
        // Process the PDF
        const result = await processPdf(file);
        
            // Show success status with ML mode information
            if (result.adversarialAttacksApplied && result.adversarialGlyphsApplied) {
                const mlStatus = result.mlStatus ? ` (${result.mlStatus})` : '';
                showProcessingStatus(`PDF processed successfully with both image and glyph adversarial attacks applied!${mlStatus}`, 'success');
            } else if (result.adversarialAttacksApplied) {
                const mlStatus = result.mlStatus ? ` (${result.mlStatus})` : '';
                showProcessingStatus(`PDF processed successfully with adversarial attacks applied!${mlStatus}`, 'success');
            } else if (result.adversarialGlyphsApplied) {
                const mlStatus = result.mlStatus ? ` (${result.mlStatus})` : '';
                showProcessingStatus(`PDF processed successfully with adversarial glyph attacks applied!${mlStatus}`, 'success');
            } else {
                showProcessingStatus('PDF processed successfully!', 'success');
            }
            
            // Update system status
            if (result.mlStatus) {
                document.getElementById('mlStatus').textContent = result.mlStatus;
            }
        
        // Enable download button
        downloadBtn.disabled = false;
        
        // Update preview
        updatePreview(file, true);
        
    } catch (error) {
        console.error('Error processing PDF:', error);
        showProcessingStatus('Error processing PDF. Please try again.', 'error');
        resetFileState();
    }
}

/**
 * Processes the PDF by sending it to the serverless function
 * @param {File} file - The PDF file to process
 */
async function processPdf(file) {
    // Convert file to base64
    const base64 = await fileToBase64(file);
    
        // Get adversarial attacks options
        const applyAdversarialAttacks = document.getElementById('applyAdversarialAttacks').checked;
        const applyAdversarialGlyphs = document.getElementById('applyAdversarialGlyphs').checked;
        
        // Send to processing endpoint
        const response = await fetch('/api/process-pdf', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                pdfData: base64,
                filename: file.name,
                applyAdversarialAttacks: applyAdversarialAttacks,
                applyAdversarialGlyphs: applyAdversarialGlyphs
            })
        });
    
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const result = await response.json();
    
    if (!result.success) {
        throw new Error('Failed to process PDF');
    }
    
    // Store the processed PDF data
    processedPdfData = result.processedPdf;
    
    // Return the result for status checking
    return result;
}

/**
 * Converts a file to base64 string
 * @param {File} file - The file to convert
 * @returns {Promise<string>} Base64 string
 */
function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => {
            // Remove the data URL prefix to get just the base64 data
            const base64 = reader.result.split(',')[1];
            resolve(base64);
        };
        reader.onerror = error => reject(error);
    });
}

/**
 * Shows processing status to the user
 * @param {string} message - Status message
 * @param {string} type - Status type (processing, success, error)
 */
function showProcessingStatus(message, type) {
    if (!processingStatus) {
        console.warn('Processing status element not found');
        return;
    }
    processingStatus.innerHTML = `
        <div class="status-message ${type}">
            <span class="status-icon">${getStatusIcon(type)}</span>
            <span class="status-text">${message}</span>
        </div>
    `;
    processingStatus.classList.add('show');
}

/**
 * Gets the appropriate icon for status type
 * @param {string} type - Status type
 * @returns {string} Icon emoji
 */
function getStatusIcon(type) {
    switch (type) {
        case 'processing': return '⏳';
        case 'success': return '✅';
        case 'error': return '❌';
        default: return 'ℹ️';
    }
}

/**
 * Displays information about the uploaded file
 * @param {File} file - The uploaded file object
 */
function displayFileInfo(file) {
    if (!fileInfo) {
        console.warn('File info element not found');
        return;
    }
    
    const fileSizeKB = (file.size / 1024).toFixed(2);
    const fileSizeMB = (file.size / (1024 * 1024)).toFixed(2);
    
    fileInfo.innerHTML = `
        <h3>✅ File Uploaded Successfully!</h3>
        <p><strong>Name:</strong> ${file.name}</p>
        <p><strong>Size:</strong> ${fileSizeKB} KB (${fileSizeMB} MB)</p>
        <p><strong>Type:</strong> ${file.type}</p>
        <p><strong>Last Modified:</strong> ${new Date(file.lastModified).toLocaleString()}</p>
    `;
    
    fileInfo.classList.add('show');
}

/**
 * Updates the preview section to show PDF information
 * @param {File} file - The uploaded file object
 * @param {boolean} processed - Whether the PDF has been processed
 */
function updatePreview(file, processed = false) {
    if (!previewContainer) {
        console.warn('Preview container element not found');
        return;
    }
    
    const statusText = processed ? 'Ready for download!' : 'Processing...';
    const statusIcon = processed ? '✅' : '⏳';
    
    previewContainer.innerHTML = `
        <div>
            <h3>📄 ${file.name}</h3>
            <p>${statusIcon} ${statusText}</p>
            <p>File size: ${(file.size / 1024).toFixed(2)} KB</p>
            ${processed ? '<p class="processed-indicator">🔒 Processed with encryption</p>' : ''}
        </div>
    `;
    previewContainer.classList.add('has-pdf');
}

/**
 * Handles the download functionality
 * This function is called when the download button is clicked
 */
function handleDownload() {
    // Check if there's a processed PDF to download
    if (!processedPdfData) {
        alert('No processed PDF to download. Please upload and process a PDF first.');
        return;
    }
    
    // Convert base64 to blob
    const byteCharacters = atob(processedPdfData);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: 'application/pdf' });
    
    // Create a download link
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    
    // Set the download attributes
    link.href = url;
    link.download = `processed_${uploadedFile.name}`;
    
    // Append to body, click, and remove
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    // Clean up the URL object
    URL.revokeObjectURL(url);
    
    // Show download confirmation
    showDownloadInfo();
}

/**
 * Shows download confirmation message
 */
function showDownloadInfo() {
    if (!downloadInfo) {
        console.warn('Download info element not found');
        return;
    }
    
    downloadInfo.innerHTML = `
        <h3>✅ Download Started!</h3>
        <p>Your PDF file "${uploadedFile.name}" is being downloaded.</p>
        <p>Check your browser's download folder.</p>
    `;
    
    downloadInfo.classList.add('show');
    
    // Hide the message after 5 seconds
    setTimeout(() => {
        if (downloadInfo) {
            downloadInfo.classList.remove('show');
        }
    }, 5000);
}

/**
 * Resets the file state when no file is selected
 */
function resetFileState() {
    uploadedFile = null;
    processedPdfData = null;
    if (fileInfo) fileInfo.classList.remove('show');
    if (processingStatus) processingStatus.classList.remove('show');
    if (downloadBtn) downloadBtn.disabled = true;
    if (previewContainer) {
        previewContainer.innerHTML = '<p>No PDF selected</p>';
        previewContainer.classList.remove('has-pdf');
    }
}

/**
 * Drag and drop functionality for better user experience
 */
// Prevent default drag behaviors
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    if (previewContainer) {
        previewContainer.addEventListener(eventName, preventDefaults, false);
    }
    document.body.addEventListener(eventName, preventDefaults, false);
});

// Highlight drop area when item is dragged over it
['dragenter', 'dragover'].forEach(eventName => {
    if (previewContainer) {
        previewContainer.addEventListener(eventName, highlight, false);
    }
});

['dragleave', 'drop'].forEach(eventName => {
    if (previewContainer) {
        previewContainer.addEventListener(eventName, unhighlight, false);
    }
});

// Handle dropped files
if (previewContainer) {
    previewContainer.addEventListener('drop', handleDrop, false);
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight(e) {
    if (previewContainer) {
        previewContainer.style.borderColor = '#4caf50';
        previewContainer.style.backgroundColor = '#e8f5e8';
    }
}

function unhighlight(e) {
    if (previewContainer) {
        previewContainer.style.borderColor = '#ddd';
        previewContainer.style.backgroundColor = '#fafafa';
    }
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    console.log('Drop event triggered, files:', files.length);
    
    if (files.length > 0) {
        const file = files[0];
        console.log('Dropped file:', file.name, 'Type:', file.type);
        
        if (file.type === 'application/pdf') {
            // Create a new file input event
            const input = document.getElementById('pdfInput');
            
            // Try to set the file directly (modern browsers)
            if (input.files && 'DataTransfer' in window) {
                try {
                    const dataTransfer = new DataTransfer();
                    dataTransfer.items.add(file);
                    input.files = dataTransfer.files;
                    
                    // Trigger the change event
                    const event = new Event('change', { bubbles: true });
                    input.dispatchEvent(event);
                } catch (error) {
                    console.error('Error setting file via DataTransfer:', error);
                    // Fallback: manually trigger the upload
                    handleFileUpload({ target: { files: [file] } });
                }
            } else {
                // Fallback for older browsers
                console.log('Using fallback method for file upload');
                handleFileUpload({ target: { files: [file] } });
            }
        } else {
            alert('Please drop a PDF file.');
        }
    }
}


