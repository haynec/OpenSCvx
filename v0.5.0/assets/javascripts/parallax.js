/**
 * Minimal parallax effect using scroll position
 * Only handles essential parallax elements
 */
(function() {
  'use strict';

  // Check if parallax is supported
  if (!window.requestAnimationFrame) {
    return;
  }

  let ticking = false;
  let scrollY = 0;

  // Get all parallax elements
  const parallaxElements = document.querySelectorAll('[data-parallax-speed]');

  if (parallaxElements.length === 0) {
    return;
  }

  /**
   * Update parallax positions based on scroll
   */
  function updateParallax() {
    scrollY = window.pageYOffset || window.scrollY || 0;

    parallaxElements.forEach(function(element) {
      const speed = parseFloat(element.getAttribute('data-parallax-speed')) || 0.5;
      const translateY = scrollY * speed;
      
      element.style.transform = `translate3d(0, ${translateY}px, 0)`;
    });

    ticking = false;
  }

  /**
   * Request animation frame for smooth parallax
   */
  function requestTick() {
    if (!ticking) {
      window.requestAnimationFrame(updateParallax);
      ticking = true;
    }
  }

  // Initialize on scroll
  window.addEventListener('scroll', requestTick, { passive: true });
  
  // Initial update
  updateParallax();
})();
