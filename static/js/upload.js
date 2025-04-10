document.addEventListener('DOMContentLoaded', function() {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const fileInfo = document.getElementById('file-info');
    const submitBtn = document.getElementById('submit-btn');
    const processingIcon = document.getElementById('processing-icon');
    const successIcon = document.getElementById('success-icon');
    
    // Gestion du glisser-déposer
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        dropArea.classList.add('active');
    }
    
    function unhighlight() {
        dropArea.classList.remove('active');
    }
    
    // Gestion du dépôt de fichier
    dropArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        fileInput.files = files;
        handleFiles(files);
    }
    
    // Gestion de la sélection de fichier
    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });
    
    function handleFiles(files) {
        if (files.length) {
            const file = files[0];
            displayFileInfo(file);
            showSubmitButton();
        }
    }
    
    function displayFileInfo(file) {
        fileInfo.innerHTML = `
            <i class="fas fa-file-alt"></i> ${file.name} 
            <span class="file-size">(${formatFileSize(file.size)})</span>
        `;
        successIcon.style.display = 'block';
    }
    
    function showSubmitButton() {
        submitBtn.classList.add('visible');
    }
    
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    // Soumission du formulaire
    document.getElementById('file-form').addEventListener('submit', function() {
        processingIcon.style.display = 'block';
        successIcon.style.display = 'none';
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Traitement...';
    });
});