// Fonction pour empêcher le comportement par défaut lors du glisser-déposer
function preventDefault(e) {
    e.preventDefault();
    e.stopPropagation();
}

const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('file-input');
const fileForm = document.getElementById('file-form');
const processingIcon = document.getElementById('processing-icon');
const successIcon = document.getElementById('success-icon');

// Empêcher l'événement par défaut lors du glisser et du déposer
dropArea.addEventListener('dragover', preventDefault, false);
dropArea.addEventListener('dragenter', preventDefault, false);
dropArea.addEventListener('drop', handleDrop, false);

// Gérer le fichier déposé
function handleDrop(e) {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        dropArea.classList.add('has-file');
        dropArea.querySelector('p').textContent = files[0].name; // Afficher le nom du fichier

        // Afficher l'icône de traitement
        processingIcon.classList.remove('d-none');
        successIcon.classList.add('d-none'); // Cacher l'icône de succès
        
        // Soumettre automatiquement le formulaire après un dépot
        fileForm.submit();
    }
}

// Optionnel : cliquer sur la zone de dépose pour ouvrir la fenêtre de sélection de fichier
dropArea.addEventListener('click', function () {
    fileInput.click();
});

// Affichage du nom du fichier sélectionné et soumission automatique
fileInput.addEventListener('change', function () {
    if (fileInput.files.length > 0) {
        dropArea.classList.add('has-file');
        dropArea.querySelector('p').textContent = fileInput.files[0].name;

        // Afficher l'icône de traitement
        processingIcon.classList.remove('d-none');
        successIcon.classList.add('d-none'); // Cacher l'icône de succès

        // Soumettre automatiquement le formulaire après sélection du fichier
        fileForm.submit();
    }
});




 // Animation d'entrée progressive
 document.addEventListener('DOMContentLoaded', function() {
    const cards = document.querySelectorAll('.summary-card');
    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'all 0.5s ease ' + (index * 0.1) + 's';
        
        setTimeout(() => {
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, 100);
    });
    
    // Animation au survol des boutons
    const buttons = document.querySelectorAll('.btn');
    buttons.forEach(btn => {
        btn.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-3px)';
        });
        btn.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
});