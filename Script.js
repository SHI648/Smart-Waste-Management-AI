document.addEventListener('DOMContentLoaded', () => {
    initDragAndDrop();
    initCamera();
    initDarkMode();
    initDashboardCharts();
});

function showToast(message, type = 'success') {
    const toastContainer = document.getElementById('toast-container') || createToastContainer();
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type === 'success' ? 'success' : 'danger'} border-0 show mb-2`;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');
    
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                <i class="bi ${type === 'success' ? 'bi-check-circle-fill' : 'bi-exclamation-triangle-fill'} me-2"></i>
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
    `;
    
    toastContainer.appendChild(toast);
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function createToastContainer() {
    const container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
    document.body.appendChild(container);
    return container;
}

function toggleLoading(show) {
    let loader = document.getElementById('ai-loader');
    if (show) {
        if (!loader) {
            loader = document.createElement('div');
            loader.id = 'ai-loader';
            loader.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(15,23,42,0.7);backdrop-filter:blur(8px);z-index:9999;display:flex;flex-direction:column;align-items:center;justify-content:center;color:#fff;';
            loader.innerHTML = `
                <div class="spinner-border text-success mb-3" style="width: 3rem; height: 3rem;" role="status"></div>
                <h5 class="fw-bold tracking-wide pulse-glow">AI Processing Target...</h5>
                <p class="text-muted small">Running inference matrix & extraction templates</p>
            `;
            document.body.appendChild(loader);
        }
    } else if (loader) {
        loader.remove();
    }
}

function initDragAndDrop() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');

    if (!dropZone || !fileInput) return;

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => e.preventDefault(), false);
        dropZone.addEventListener(eventName, (e) => e.stopPropagation(), false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('border-success'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('border-success'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) handleFileSelection(files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFileSelection(e.target.files[0]);
    });

    function handleFileSelection(file) {
        if (!file.type.startsWith('image/')) {
            showToast('Invalid file format. Please inject an image.', 'error');
            return;
        }

        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = () => {
            if (previewImage && previewContainer) {
                previewImage.src = reader.result;
                previewContainer.classList.remove('d-none');
                dropZone.classList.add('d-none');
            }
            executeAjaxUpload(file);
        };
    }
}

function executeAjaxUpload(file) {
    const formData = new FormData();
    formData.append('file', file);

    toggleLoading(true);

    fetch('/api/classify', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) throw new Error('Network execution sequence failure.');
        return response.json();
    })
    .then(data => {
        showToast('Classification operational matrix executed successfully.');
        window.location.href = `/result?id=${data.prediction_id}`;
    })
    .catch(error => {
        console.error(error);
        toggleLoading(false);
        showToast('Inference execution runtime error.', 'error');
    });
}

function initCamera() {
    const video = document.getElementById('webcam-view');
    const captureBtn = document.getElementById('capture-btn');
    const startCamBtn = document.getElementById('start-cam-btn');
    let streamInstance = null;

    if (!startCamBtn) return;

    startCamBtn.addEventListener('click', () => {
        navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } })
            .then(stream => {
                streamInstance = stream;
                if (video) {
                    video.srcObject = stream;
                    video.classList.remove('d-none');
                    if (captureBtn) captureBtn.classList.remove('d-none');
                    startCamBtn.classList.add('d-none');
                    showToast('Camera peripheral initialization complete.');
                }
            })
            .catch(err => {
                console.error(err);
                showToast('Unable to open hardware video layer.', 'error');
            });
    });

    if (captureBtn && video) {
        captureBtn.addEventListener('click', () => {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            canvas.toBlob((blob) => {
                if (streamInstance) {
                    streamInstance.getTracks().forEach(track => track.stop());
                }
                video.classList.add('d-none');
                captureBtn.classList.add('d-none');
                startCamBtn.classList.remove('d-none');
                
                const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
                executeAjaxUpload(file);
            }, 'image/jpeg');
        });
    }
}

function initDarkMode() {
    const themeToggle = document.getElementById('theme-toggle');
    if (!themeToggle) return;

    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-bs-theme', savedTheme);
    updateThemeIcon(savedTheme);

    themeToggle.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-bs-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-bs-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        updateThemeIcon(newTheme);
    });

    function updateThemeIcon(theme) {
        const icon = themeToggle.querySelector('i');
        if (!icon) return;
        if (theme === 'dark') {
            icon.className = 'bi bi-sun-fill';
        } else {
            icon.className = 'bi bi-moon-stars-fill';
        }
    }
}

function initDashboardCharts() {
    const distCtx = document.getElementById('wasteDistributionChart');
    if (distCtx && window.Chart) {
        const labels = JSON.parse(distCtx.getAttribute('data-labels') || '[]');
        const data = JSON.parse(distCtx.getAttribute('data-values') || '[]');
        new Chart(distCtx.getContext('2d'), {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: ['#fd7e14', '#0d6efd', '#6c757d', '#0dcaf0', '#198754', '#dc3545'],
                    borderWidth: 2
                }]
            },
            options: { responsive: true, maintainAspectRatio: false }
        });
    }

    const trendCtx = document.getElementById('monthlyPredictionsChart');
    if (trendCtx && window.Chart) {
        const labels = JSON.parse(trendCtx.getAttribute('data-labels') || '[]');
        const data = JSON.parse(trendCtx.getAttribute('data-values') || '[]');
        new Chart(trendCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Classifications',
                    data: data,
                    borderColor: '#198754',
                    backgroundColor: 'rgba(25, 135, 84, 0.05)',
                    fill: true,
                    tension: 0.3
                }]
            },
            options: { responsive: true, maintainAspectRatio: false }
        });
    }

    const confCtx = document.getElementById('confidenceDistributionChart');
    if (confCtx && window.Chart) {
        const data = JSON.parse(confCtx.getAttribute('data-values') || '[]');
        new Chart(confCtx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: ['50-60%', '60-70%', '70-80%', '80-90%', '90-100%'],
                datasets: [{
                    data: data,
                    backgroundColor: '#ffc107',
                    borderRadius: 4
                }]
            },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
        });
    }
}