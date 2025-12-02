// ================================
// Theme Toggle Functionality
// ================================
const themeToggle = document.getElementById('theme-toggle');
const htmlElement = document.documentElement;

// Check for saved theme preference or default to 'light'
const currentTheme = localStorage.getItem('theme') || 'light';
htmlElement.setAttribute('data-theme', currentTheme);

themeToggle.addEventListener('click', () => {
    const theme = htmlElement.getAttribute('data-theme');
    const newTheme = theme === 'light' ? 'dark' : 'light';

    htmlElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);

    // Add a little animation to the toggle button
    themeToggle.style.transform = 'rotate(360deg)';
    setTimeout(() => {
        themeToggle.style.transform = 'rotate(0deg)';
    }, 300);
});

// ================================
// Mobile Navigation Menu
// ================================
const hamburger = document.getElementById('hamburger');
const navMenu = document.getElementById('nav-menu');
const navLinks = document.querySelectorAll('.nav-link');

// Toggle mobile menu
hamburger.addEventListener('click', () => {
    hamburger.classList.toggle('active');
    navMenu.classList.toggle('active');
});

// Close mobile menu when clicking on a link
navLinks.forEach(link => {
    link.addEventListener('click', () => {
        hamburger.classList.remove('active');
        navMenu.classList.remove('active');
    });
});

// Close mobile menu when clicking outside
document.addEventListener('click', (e) => {
    if (!navMenu.contains(e.target) && !hamburger.contains(e.target)) {
        hamburger.classList.remove('active');
        navMenu.classList.remove('active');
    }
});

// ================================
// Smooth Scrolling
// ================================
// Function to smooth scroll to a section
function smoothScrollToSection(targetId) {
    const targetSection = document.querySelector(targetId);
    if (targetSection) {
        const navbarHeight = document.getElementById('navbar').offsetHeight;
        const targetPosition = targetSection.offsetTop - navbarHeight;

        window.scrollTo({
            top: targetPosition,
            behavior: 'smooth'
        });
    }
}

// Apply to nav links
navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const targetId = link.getAttribute('href');
        smoothScrollToSection(targetId);
    });
});

// Apply to all internal links (including hero buttons)
document.querySelectorAll('a[href^="#"]').forEach(link => {
    link.addEventListener('click', (e) => {
        const targetId = link.getAttribute('href');
        if (targetId && targetId !== '#') {
            e.preventDefault();
            smoothScrollToSection(targetId);
        }
    });
});

// ================================
// Navbar Scroll Effect
// ================================
const navbar = document.getElementById('navbar');
let lastScrollTop = 0;

window.addEventListener('scroll', () => {
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;

    // Add shadow when scrolled
    if (scrollTop > 50) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }

    lastScrollTop = scrollTop;
});

// ================================
// Scroll Reveal Animation
// ================================
const revealElements = document.querySelectorAll('.project-card, .skill-category, .about-content, .contact-content');

const revealOnScroll = () => {
    revealElements.forEach(element => {
        const elementTop = element.getBoundingClientRect().top;
        const elementBottom = element.getBoundingClientRect().bottom;
        const windowHeight = window.innerHeight;

        // Reveal when element is 100px into viewport
        if (elementTop < windowHeight - 100 && elementBottom > 0) {
            element.classList.add('scroll-reveal', 'active');
        }
    });
};

// Initial check on page load
revealOnScroll();

// Check on scroll with throttling for performance
let scrollTimeout;
window.addEventListener('scroll', () => {
    if (scrollTimeout) {
        window.cancelAnimationFrame(scrollTimeout);
    }
    scrollTimeout = window.requestAnimationFrame(() => {
        revealOnScroll();
    });
});

// ================================
// Active Navigation Link
// ================================
const sections = document.querySelectorAll('section[id]');

const highlightNavLink = () => {
    const scrollY = window.pageYOffset;

    sections.forEach(section => {
        const sectionHeight = section.offsetHeight;
        const sectionTop = section.offsetTop - 100;
        const sectionId = section.getAttribute('id');
        const navLink = document.querySelector(`.nav-link[href="#${sectionId}"]`);

        if (scrollY > sectionTop && scrollY <= sectionTop + sectionHeight) {
            navLinks.forEach(link => link.classList.remove('active'));
            if (navLink) {
                navLink.classList.add('active');
            }
        }
    });
};

window.addEventListener('scroll', highlightNavLink);

// ================================
// Typing Effect for Hero Section (Optional)
// ================================
const heroSubtitle = document.querySelector('.hero-subtitle');
const roles = [
    'Full Stack Developer',
    'Problem Solver',
    'Tech Enthusiast',
    'Creative Thinker',
    'Lifelong Learner'
];

let roleIndex = 0;
let charIndex = 0;
let isDeleting = false;
let typingSpeed = 100;

function typeEffect() {
    const currentRole = roles[roleIndex];

    if (isDeleting) {
        heroSubtitle.textContent = currentRole.substring(0, charIndex - 1);
        charIndex--;
        typingSpeed = 50;
    } else {
        heroSubtitle.textContent = currentRole.substring(0, charIndex + 1);
        charIndex++;
        typingSpeed = 100;
    }

    if (!isDeleting && charIndex === currentRole.length) {
        // Pause at end of word
        typingSpeed = 2000;
        isDeleting = true;
    } else if (isDeleting && charIndex === 0) {
        isDeleting = false;
        roleIndex = (roleIndex + 1) % roles.length;
        typingSpeed = 500;
    }

    setTimeout(typeEffect, typingSpeed);
}

// ================================
// Intersection Observer for Better Performance
// ================================
// More efficient alternative to scroll event listeners
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -100px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('scroll-reveal', 'active');
            // Optional: unobserve after animation
            // observer.unobserve(entry.target);
        }
    });
}, observerOptions);

// Observe all project cards and skill categories
document.querySelectorAll('.project-card, .skill-category').forEach(el => {
    observer.observe(el);
});

// ================================
// Scroll to Top Button (Optional)
// ================================
const scrollToTopBtn = document.createElement('button');
scrollToTopBtn.innerHTML = '↑';
scrollToTopBtn.className = 'scroll-to-top';
scrollToTopBtn.setAttribute('aria-label', 'Scroll to top');
document.body.appendChild(scrollToTopBtn);

// Add styles for the button
const style = document.createElement('style');
style.textContent = `
    .scroll-to-top {
        position: fixed;
        bottom: 30px;
        right: 30px;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
        color: white;
        border: none;
        font-size: 1.5rem;
        cursor: pointer;
        opacity: 0;
        visibility: hidden;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-lg);
        z-index: 999;
    }

    .scroll-to-top.visible {
        opacity: 1;
        visibility: visible;
    }

    .scroll-to-top:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-xl);
    }
`;
document.head.appendChild(style);

// Show/hide scroll to top button
window.addEventListener('scroll', () => {
    if (window.pageYOffset > 300) {
        scrollToTopBtn.classList.add('visible');
    } else {
        scrollToTopBtn.classList.remove('visible');
    }
});

// Scroll to top on click
scrollToTopBtn.addEventListener('click', () => {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
});

// ================================
// Keyboard Navigation
// ================================
document.addEventListener('keydown', (e) => {
    // Press 'T' to toggle theme
    if (e.key === 't' || e.key === 'T') {
        if (document.activeElement.tagName !== 'INPUT' &&
            document.activeElement.tagName !== 'TEXTAREA') {
            themeToggle.click();
        }
    }

    // Press 'Escape' to close mobile menu
    if (e.key === 'Escape') {
        hamburger.classList.remove('active');
        navMenu.classList.remove('active');
    }
});

// ================================
// Preload Performance Optimization
// ================================
// Add loading class to body
document.body.classList.add('loading');

// Remove loading class when page is fully loaded
window.addEventListener('load', () => {
    document.body.classList.remove('loading');
});

// ================================
// PDF Slides Generator
// ================================

async function generateSlidesFromPDF() {
    const carouselSlides = document.querySelector('.carousel-slides');
    if (!carouselSlides) return;

    const pdfUrl = carouselSlides.getAttribute('data-pdf-url');
    if (!pdfUrl) return;

    const slideInfoAttr = carouselSlides.getAttribute('data-slide-info');
    const slideInfo = slideInfoAttr ? JSON.parse(slideInfoAttr) : [];

    // Configure PDF.js worker
    pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

    const slides = [];
    const dotsContainer = document.querySelector('.carousel-dots');

    try {
        // Show loading message
        carouselSlides.innerHTML = `
            <div class="carousel-slide active">
                <div class="slide-content">
                    <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-secondary); text-align: center; padding: 2rem;">
                        <div>
                            <h3>Loading PDF...</h3>
                            <p>Please wait while we load your research slides.</p>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Load the PDF with CORS mode
        const loadingTask = pdfjsLib.getDocument({
            url: pdfUrl,
            withCredentials: false
        });
        const pdf = await loadingTask.promise;
        const numPages = pdf.numPages;

        // Generate a slide for each page
        for (let pageNum = 1; pageNum <= numPages; pageNum++) {
            const page = await pdf.getPage(pageNum);
            const viewport = page.getViewport({ scale: 2.0 });

            // Create canvas for rendering
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            canvas.width = viewport.width;
            canvas.height = viewport.height;

            // Render PDF page to canvas
            await page.render({
                canvasContext: context,
                viewport: viewport
            }).promise;

            // Convert canvas to image data URL
            const imageDataUrl = canvas.toDataURL('image/png');

            // Get slide info for this page (if provided)
            const info = slideInfo[pageNum - 1] || {
                title: `Research Highlight ${pageNum}`,
                description: `Page ${pageNum} of ${numPages}`
            };

            // Create slide HTML
            const slideHTML = `
                <div class="carousel-slide ${pageNum === 1 ? 'active' : ''}">
                    <div class="slide-content">
                        <img src="${imageDataUrl}" alt="${info.title}" class="slide-image">
                        <div class="slide-info">
                            <h3 class="slide-title">${info.title}</h3>
                            <p class="slide-description">${info.description}</p>
                        </div>
                    </div>
                </div>
            `;

            slides.push(slideHTML);

            // Create dot indicator
            const dot = document.createElement('button');
            dot.className = `carousel-dot ${pageNum === 1 ? 'active' : ''}`;
            dot.setAttribute('data-slide', pageNum - 1);
            dot.setAttribute('aria-label', `Go to slide ${pageNum}`);
            dotsContainer.appendChild(dot);
        }

        // Insert all slides at once
        carouselSlides.innerHTML = slides.join('');

    } catch (error) {
        console.error('Error loading PDF:', error);

        // Show error message in carousel
        carouselSlides.innerHTML = `
            <div class="carousel-slide active">
                <div class="slide-content">
                    <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-secondary); text-align: center; padding: 2rem;">
                        <div>
                            <h3>Error loading PDF</h3>
                            <p>Please check the PDF URL and try again.</p>
                            <p style="font-size: 0.875rem; margin-top: 1rem;">Error: ${error.message}</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
}

// Initialize PDF slides before carousel
(async function() {
    await generateSlidesFromPDF();

    // Now initialize the carousel after slides are generated
    if (document.querySelector('.carousel-slide')) {
        window.carouselInstance = new ResearchCarousel();
    }
})();

// ================================
// Research Carousel Functionality
// ================================

class ResearchCarousel {
    constructor() {
        this.slides = document.querySelectorAll('.carousel-slide');
        this.dots = document.querySelectorAll('.carousel-dot');
        this.prevBtn = document.querySelector('.carousel-arrow-left');
        this.nextBtn = document.querySelector('.carousel-arrow-right');
        this.currentSlide = 0;
        this.autoPlayInterval = null;
        this.autoPlayDelay = 10000; // 10 seconds after last hover
        this.lastHoverTime = Date.now();

        this.init();
    }

    init() {
        if (!this.slides.length) return;

        // Set up event listeners
        this.prevBtn.addEventListener('click', () => {
            this.prevSlide();
            this.handleUserInteraction();
        });
        this.nextBtn.addEventListener('click', () => {
            this.nextSlide();
            this.handleUserInteraction();
        });

        // Dot navigation
        this.dots.forEach((dot, index) => {
            dot.addEventListener('click', () => {
                this.goToSlide(index);
                this.handleUserInteraction();
            });
        });

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') {
                this.prevSlide();
                this.handleUserInteraction();
            }
            if (e.key === 'ArrowRight') {
                this.nextSlide();
                this.handleUserInteraction();
            }
        });

        // Touch/swipe support
        this.setupTouchSupport();

        // Track hover on carousel
        const carousel = document.querySelector('.carousel-container');
        if (carousel) {
            carousel.addEventListener('mouseenter', () => this.handleUserInteraction());
            carousel.addEventListener('mousemove', () => this.handleUserInteraction());
        }

        // Start auto-advance check
        this.startAutoAdvance();
    }

    handleUserInteraction() {
        // Update last hover time whenever user interacts
        this.lastHoverTime = Date.now();
    }

    startAutoAdvance() {
        // Check every second if it's time to advance
        setInterval(() => {
            const timeSinceLastHover = Date.now() - this.lastHoverTime;
            if (timeSinceLastHover >= this.autoPlayDelay) {
                this.nextSlide();
                this.lastHoverTime = Date.now(); // Reset timer after auto-advance
            }
        }, 1000);
    }

    goToSlide(index) {
        // Remove active class from current slide and dot
        this.slides[this.currentSlide].classList.remove('active');
        this.dots[this.currentSlide].classList.remove('active');

        // Update current slide
        this.currentSlide = index;

        // Add active class to new slide and dot
        this.slides[this.currentSlide].classList.add('active');
        this.dots[this.currentSlide].classList.add('active');
    }

    nextSlide() {
        const nextIndex = (this.currentSlide + 1) % this.slides.length;
        this.goToSlide(nextIndex);
    }

    prevSlide() {
        const prevIndex = (this.currentSlide - 1 + this.slides.length) % this.slides.length;
        this.goToSlide(prevIndex);
    }

    setupTouchSupport() {
        const carousel = document.querySelector('.carousel-slides');
        if (!carousel) return;

        let touchStartX = 0;
        let touchEndX = 0;

        carousel.addEventListener('touchstart', (e) => {
            touchStartX = e.changedTouches[0].screenX;
        }, { passive: true });

        carousel.addEventListener('touchend', (e) => {
            touchEndX = e.changedTouches[0].screenX;
            this.handleSwipe();
            this.handleUserInteraction();
        });

        const handleSwipe = () => {
            const swipeThreshold = 50;
            const diff = touchStartX - touchEndX;

            if (Math.abs(diff) > swipeThreshold) {
                if (diff > 0) {
                    this.nextSlide();
                } else {
                    this.prevSlide();
                }
            }
        };

        this.handleSwipe = handleSwipe;
    }
}