// MedAI Fusion - Interactive Frontend Logic

document.addEventListener('DOMContentLoaded', function() {
    // Form navigation
    const form = document.getElementById('health-assessment-form');
    const formSteps = document.querySelectorAll('.form-step');
    const progressSteps = document.querySelectorAll('.progress-step');
    const nextBtns = document.querySelectorAll('.next-step');
    const prevBtns = document.querySelectorAll('.prev-step');
    
    let currentStep = 1;
    
    // Next button functionality
    nextBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            if (validateStep(currentStep)) {
                currentStep++;
                updateFormSteps();
            }
        });
    });
    
    // Previous button functionality
    prevBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            currentStep--;
            updateFormSteps();
        });
    });
    
    // Update form steps display
    function updateFormSteps() {
        formSteps.forEach((step, index) => {
            step.classList.toggle('active', index + 1 === currentStep);
        });
        
        progressSteps.forEach((step, index) => {
            step.classList.toggle('active', index + 1 <= currentStep);
        });
    }
    
    // Validate current step
    function validateStep(step) {
        const currentFormStep = document.getElementById(`step-${step}`);
        const requiredFields = currentFormStep.querySelectorAll('[required]');
        
        for (let field of requiredFields) {
            if (!field.value) {
                field.classList.add('error');
                showNotification('Please fill all required fields', 'error');
                return false;
            }
            field.classList.remove('error');
        }
        return true;
    }
    
    // Handle form submission
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            // Show processing animation
            currentStep = 4;
            updateFormSteps();
            
            // Animate processing steps
            animateProcessingSteps();
            
            // Prepare form data
            const formData = new FormData(form);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    // Redirect to results page
                    setTimeout(() => {
                        window.location.href = result.redirect;
                    }, 3000);
                } else {
                    showNotification('Error processing data. Please try again.', 'error');
                }
            } catch (error) {
                console.error('Error:', error);
                showNotification('Network error. Please check your connection.', 'error');
            }
        });
    }
    
    // Animate processing steps
    function animateProcessingSteps() {
        const steps = ['proc-clinical', 'proc-image', 'proc-wearable', 'proc-fusion'];
        
        steps.forEach((stepId, index) => {
            setTimeout(() => {
                const step = document.getElementById(stepId);
                if (step) {
                    step.classList.add('completed');
                    step.querySelector('.proc-icon').textContent = '✓';
                }
            }, (index + 1) * 750);
        });
    }
    
    // File upload preview
    const fileInput = document.getElementById('medical_image');
    const preview = document.getElementById('image-preview');
    
    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file && file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    preview.innerHTML = `
                        <img src="${e.target.result}" alt="Medical image preview" 
                             style="max-width: 300px; margin-top: 1rem; border-radius: 8px;">
                        <p style="margin-top: 0.5rem; color: #10b981;">✓ Image uploaded successfully</p>
                    `;
                };
                reader.readAsDataURL(file);
            }
        });
    }
    
    // Language selector
    const langSelector = document.getElementById('language-selector');
    if (langSelector) {
        langSelector.addEventListener('change', (e) => {
            const lang = e.target.value;
            // In production, this would trigger translation
            window.location.href = `/?lang=${lang}`;
        });
    }
    
    // Real-time validation
    const inputs = document.querySelectorAll('input[type="number"]');
    inputs.forEach(input => {
        input.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            const min = parseFloat(e.target.min);
            const max = parseFloat(e.target.max);
            
            if (min && value < min) {
                e.target.classList.add('error');
                showNotification(`Value must be at least ${min}`, 'warning');
            } else if (max && value > max) {
                e.target.classList.add('error');
                showNotification(`Value must be at most ${max}`, 'warning');
            } else {
                e.target.classList.remove('error');
            }
        });
    });
    
    // BMI Calculator
    const heightInput = document.querySelector('input[name="height"]');
    const weightInput = document.querySelector('input[name="weight"]');
    const bmiInput = document.querySelector('input[name="bmi"]');
    
    function calculateBMI() {
        if (heightInput && weightInput && bmiInput) {
            const height = parseFloat(heightInput.value) / 100; // Convert cm to m
            const weight = parseFloat(weightInput.value);
            
            if (height && weight) {
                const bmi = (weight / (height * height)).toFixed(1);
                bmiInput.value = bmi;
            }
        }
    }
    
    if (heightInput) heightInput.addEventListener('input', calculateBMI);
    if (weightInput) weightInput.addEventListener('input', calculateBMI);
    
    // Notification system
    function showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        
        // Style the notification
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 1rem 1.5rem;
            background: ${type === 'error' ? '#ef4444' : type === 'warning' ? '#f59e0b' : '#10b981'};
            color: white;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 9999;
            animation: slideIn 0.3s ease;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
    
    // Add CSS animations
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        @keyframes slideOut {
            from {
                transform: translateX(0);
                opacity: 1;
            }
            to {
                transform: translateX(100%);
                opacity: 0;
            }
        }
        
        .error {
            border-color: #ef4444 !important;
        }
    `;
    document.head.appendChild(style);
    
    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
    
    // Add keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            // Submit form with Ctrl+Enter
            const submitBtn = document.getElementById('analyze-btn');
            if (submitBtn) submitBtn.click();
        }
    });
    
    // Print results functionality
    const printBtn = document.getElementById('print-results');
    if (printBtn) {
        printBtn.addEventListener('click', () => {
            window.print();
        });
    }
    
    // Share results functionality
    const shareBtn = document.getElementById('share-results');
    if (shareBtn) {
        shareBtn.addEventListener('click', async () => {
            const shareData = {
                title: 'MedAI Fusion Health Report',
                text: 'View my AI-powered health risk assessment',
                url: window.location.href
            };
            
            try {
                if (navigator.share) {
                    await navigator.share(shareData);
                } else {
                    // Copy to clipboard as fallback
                    navigator.clipboard.writeText(window.location.href);
                    showNotification('Link copied to clipboard!', 'success');
                }
            } catch (err) {
                console.error('Error sharing:', err);
            }
        });
    }
});

// Progressive Web App registration
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js').then(
            registration => console.log('SW registered:', registration),
            err => console.log('SW registration failed:', err)
        );
    });
}
