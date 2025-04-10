  // Animation au défilement
  document.addEventListener('DOMContentLoaded', function() {
    const animateOnScroll = () => {
        const elements = document.querySelectorAll('.service-card, .feature-item');
        
        elements.forEach(element => {
            const elementPosition = element.getBoundingClientRect().top;
            const screenPosition = window.innerHeight / 1.2;
            
            if(elementPosition < screenPosition) {
                element.style.opacity = '1';
                element.style.transform = 'translateY(0)';
            }
        });
    };
    
    // Initial state
    const services = document.querySelectorAll('.service-card');
    const features = document.querySelectorAll('.feature-item');
    
    services.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(30px)';
        card.style.transition = `all 0.5s ease ${index * 0.2}s`;
    });
    
    features.forEach((item, index) => {
        item.style.opacity = '0';
        item.style.transform = 'translateY(30px)';
        item.style.transition = `all 0.5s ease ${index * 0.1}s`;
    });
    
    // Trigger on load
    setTimeout(animateOnScroll, 500);
    
    // And on scroll
    window.addEventListener('scroll', animateOnScroll);
});