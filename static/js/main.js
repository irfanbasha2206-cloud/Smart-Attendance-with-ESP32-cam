// Sidebar overlay close on mobile
document.addEventListener('click', function(e) {
  const sidebar = document.getElementById('sidebar');
  if (!sidebar) return;
  if (window.innerWidth <= 900 && sidebar.classList.contains('open')) {
    if (!e.target.closest('#sidebar') && !e.target.closest('.menu-toggle')) {
      sidebar.classList.remove('open');
    }
  }
});

// Stagger animate stat cards on load
document.querySelectorAll('.stat-card').forEach((card, i) => {
  card.style.animationDelay = (i * 0.07) + 's';
  card.classList.add('animate-in');
});

// Auto-animate table rows
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
    }
  });
}, { threshold: 0.1 });

document.querySelectorAll('.attend-table tbody tr').forEach(row => {
  observer.observe(row);
});

// Progress bars animated fill on load
window.addEventListener('load', () => {
  document.querySelectorAll('.progress-bar-fill').forEach(bar => {
    const target = bar.style.width;
    bar.style.width = '0';
    setTimeout(() => { bar.style.width = target; }, 100);
  });
});
