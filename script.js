// Smooth scroll navigation
document.querySelectorAll('a[href^="#"]').forEach(link => {
  link.addEventListener('click', function (e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute('href'));
    if (target) {
      target.scrollIntoView({ behavior: 'smooth' });
    }
  });
});

// Upload & simulate compliance check
document.getElementById('uploadForm').addEventListener('submit', function (e) {
  e.preventDefault();
  const file = document.getElementById('file').files[0];
  const report = document.getElementById('report');

  if (!file) {
    report.innerHTML = `<p style="color: red;">No file selected!</p>`;
    return;
  }

  const fakeResponse = {
    total: 8,
    non_compliant: 2,
    details: ['Helmet not worn (2-wheeler)', 'Red light crossed at timestamp 00:04']
  };

  let detailsHTML = '';
  if (fakeResponse.details.length > 0) {
    detailsHTML = '<ul>' + fakeResponse.details.map(d => `<li>${d}</li>`).join('') + '</ul>';
  }

  report.innerHTML = `
    <h3>Compliance Report</h3>
    <p>Total Users Detected: ${fakeResponse.total}</p>
    <p>Non-Compliant Users: ${fakeResponse.non_compliant}</p>
    ${detailsHTML}
  `;
});
