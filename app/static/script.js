class NSFWScanner {
    constructor() {
        this.apiBase = '/api/v1';
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const scanBtn = document.getElementById('scanBtn');
        const batchScanBtn = document.getElementById('batchScanBtn');

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            this.handleFiles(files);
        });

        // File input
        fileInput.addEventListener('change', (e) => {
            this.handleFiles(e.target.files);
        });

        // Scan buttons
        scanBtn.addEventListener('click', () => this.scanImages());
        batchScanBtn.addEventListener('click', () => this.batchScan());
    }

    handleFiles(files) {
        const preview = document.getElementById('imagePreview');
        preview.innerHTML = '';
        
        Array.from(files).forEach(file => {
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    const img = document.createElement('img');
                    img.src = e.target.result;
                    img.style.maxWidth = '200px';
                    img.style.maxHeight = '200px';
                    img.style.margin = '0.5rem';
                    img.style.borderRadius = '0.5rem';
                    preview.appendChild(img);
                };
                reader.readAsDataURL(file);
            }
        });

        document.getElementById('results').style.display = 'none';
    }

    async scanImages() {
        const fileInput = document.getElementById('fileInput');
        const files = fileInput.files;
        
        if (files.length === 0) {
            this.showAlert('Please select at least one image', 'warning');
            return;
        }

        this.showLoading(true);

        try {
            const results = [];
            for (let file of files) {
                const result = await this.scanSingleImage(file);
                results.push(result);
            }

            this.displayResults(results);
        } catch (error) {
            this.showAlert('Scan failed: ' + error.message, 'danger');
        } finally {
            this.showLoading(false);
        }
    }

    async scanSingleImage(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${this.apiBase}/scan?detailed_analysis=true&return_heatmap=true`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    }

    displayResults(results) {
        const resultsDiv = document.getElementById('results');
        const resultsContent = document.getElementById('resultsContent');
        
        resultsContent.innerHTML = '';

        results.forEach((result, index) => {
            const resultCard = this.createResultCard(result, index);
            resultsContent.appendChild(resultCard);
        });

        resultsDiv.style.display = 'block';
        this.updateStats(results);
    }

    createResultCard(result, index) {
        const card = document.createElement('div');
        card.className = 'result-card';
        
        const status = result.is_nsfw ? 'NSFW Content Detected' : 'Safe Content';
        const statusClass = result.is_nsfw ? 'danger' : 'success';
        
        const confidencePercent = (result.confidence * 100).toFixed(1);
        const confidenceClass = result.confidence > 0.7 ? 'danger' : 
                              result.confidence > 0.3 ? 'warning' : 'safe';

        card.innerHTML = `
            <h3>Image ${index + 1} - <span class="text-${statusClass}">${status}</span></h3>
            <div class="confidence-meter">
                <div class="confidence-fill confidence-${confidenceClass}" 
                     style="width: ${confidencePercent}%"></div>
            </div>
            <p>Confidence: <strong>${confidencePercent}%</strong></p>
            <p>Processing Time: ${result.processing_time.toFixed(2)}s</p>
            
            <div class="category-grid">
                ${Object.entries(result.categories).map(([category, score]) => `
                    <div class="category-item">
                        <strong>${category}</strong>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${score * 100}%"></div>
                        </div>
                        <span>${(score * 100).toFixed(1)}%</span>
                    </div>
                `).join('')}
            </div>

            ${result.heatmap_url ? `
                <div class="heatmap-container">
                    <h4>Attention Heatmap</h4>
                    <img src="${result.heatmap_url}" alt="Heatmap" class="heatmap-image">
                </div>
            ` : ''}
        `;

        return card;
    }

    updateStats(results) {
        const totalScans = results.length;
        const nsfwCount = results.filter(r => r.is_nsfw).length;
        const avgConfidence = results.reduce((sum, r) => sum + r.confidence, 0) / totalScans;
        const totalTime = results.reduce((sum, r) => sum + r.processing_time, 0);

        document.getElementById('totalScans').textContent = totalScans;
        document.getElementById('nsfwCount').textContent = nsfwCount;
        document.getElementById('safeCount').textContent = totalScans - nsfwCount;
        document.getElementById('avgConfidence').textContent = (avgConfidence * 100).toFixed(1) + '%';
        document.getElementById('totalTime').textContent = totalTime.toFixed(2) + 's';
    }

    showLoading(show) {
        const loading = document.getElementById('loading');
        const scanBtn = document.getElementById('scanBtn');
        
        if (show) {
            loading.style.display = 'block';
            scanBtn.disabled = true;
            scanBtn.textContent = 'Scanning...';
        } else {
            loading.style.display = 'none';
            scanBtn.disabled = false;
            scanBtn.textContent = 'Scan Images';
        }
    }

    showAlert(message, type) {
        // Implement alert system
        console.log(`[${type}] ${message}`);
    }

    async batchScan() {
        // Implement batch scanning with progress tracking
        console.log('Batch scan feature coming soon...');
    }
}

// Initialize scanner when page loads
document.addEventListener('DOMContentLoaded', () => {
    new NSFWScanner();
});
