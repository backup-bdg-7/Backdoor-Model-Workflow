/**
 * Main JavaScript for the AI Model Training Dashboard
 */

// Global state
const state = {
    currentPage: 'dashboard',
    selectedTrainingId: null,
    selectedExportId: null,
    refreshInterval: null,
    refreshRate: 5000, // 5 seconds
    realTimeUpdates: true,
    metricsChart: null,
    pendingActions: {},
    settings: {
        maxMemory: 1024,
        defaultModelSize: 'small',
        defaultBatchSize: 8,
        defaultExportFormat: 'flask',
        defaultQuantize: true,
        updateInterval: 5
    }
};

// DOM Elements
const elements = {
    sidebar: document.querySelector('.sidebar'),
    sidebarToggle: document.getElementById('sidebar-toggle'),
    pageTitle: document.getElementById('page-title'),
    navItems: document.querySelectorAll('nav ul li'),
    pages: document.querySelectorAll('.page'),
    
    // Dashboard elements
    activeJobsCount: document.getElementById('active-jobs-count'),
    completedJobsCount: document.getElementById('completed-jobs-count'),
    exportsCount: document.getElementById('exports-count'),
    memoryUsage: document.getElementById('memory-usage'),
    activeJobsTable: document.getElementById('active-jobs-table').querySelector('tbody'),
    noActiveJobs: document.getElementById('no-active-jobs'),
    recentExportsTable: document.getElementById('recent-exports-table').querySelector('tbody'),
    noRecentExports: document.getElementById('no-recent-exports'),
    
    // Training page elements
    newTrainingBtn: document.getElementById('new-training-btn'),
    allJobsTable: document.getElementById('all-jobs-table').querySelector('tbody'),
    noAllJobs: document.getElementById('no-all-jobs'),
    
    // Exports page elements
    newExportBtn: document.getElementById('new-export-btn'),
    allExportsTable: document.getElementById('all-exports-table').querySelector('tbody'),
    noAllExports: document.getElementById('no-all-exports'),
    
    // Datasets page elements
    datasetsTable: document.getElementById('datasets-table').querySelector('tbody'),
    noDatasets: document.getElementById('no-datasets'),
    
    // Modals
    trainingDetailsModal: document.getElementById('training-details-modal'),
    exportDetailsModal: document.getElementById('export-details-modal'),
    newTrainingModal: document.getElementById('new-training-modal'),
    newExportModal: document.getElementById('new-export-modal'),
    
    // Settings
    maxMemory: document.getElementById('max-memory'),
    defaultModelSize: document.getElementById('default-model-size'),
    defaultBatchSize: document.getElementById('default-batch-size'),
    defaultExportFormat: document.getElementById('default-export-format'),
    defaultQuantize: document.getElementById('default-quantize'),
    enableRealtime: document.getElementById('enable-realtime'),
    updateInterval: document.getElementById('update-interval'),
    saveSettings: document.getElementById('save-settings'),
    resetSettings: document.getElementById('reset-settings'),
    
    // Status indicator
    statusIndicator: document.getElementById('status-indicator'),
    statusText: document.getElementById('status-text'),
};

// ===== Utility Functions =====

/**
 * Format a timestamp as a date string
 * 
 * @param {number} timestamp - Unix timestamp
 * @returns {string} - Formatted date string
 */
function formatTimestamp(timestamp) {
    if (!timestamp) return 'N/A';
    
    const date = new Date(timestamp * 1000);
    return date.toLocaleString();
}

/**
 * Format a timestamp as a relative time string
 * 
 * @param {number} timestamp - Unix timestamp
 * @returns {string} - Relative time string (e.g., "2 hours ago")
 */
function formatRelativeTime(timestamp) {
    if (!timestamp) return 'N/A';
    
    const seconds = Math.floor((Date.now() / 1000) - timestamp);
    
    if (seconds < 60) return `${seconds} seconds ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)} minutes ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)} hours ago`;
    return `${Math.floor(seconds / 86400)} days ago`;
}

/**
 * Format bytes as a human-readable string
 * 
 * @param {number} bytes - Number of bytes
 * @returns {string} - Formatted size string
 */
function formatBytes(bytes) {
    if (bytes === 0 || !bytes) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Clear the content of an element
 * 
 * @param {HTMLElement} element - Element to clear
 */
function clearElement(element) {
    while (element.firstChild) {
        element.removeChild(element.firstChild);
    }
}

/**
 * Show an element
 * 
 * @param {HTMLElement} element - Element to show
 */
function showElement(element) {
    element.style.display = 'block';
}

/**
 * Hide an element
 * 
 * @param {HTMLElement} element - Element to hide
 */
function hideElement(element) {
    element.style.display = 'none';
}

/**
 * Show a modal
 * 
 * @param {HTMLElement} modal - Modal element to show
 */
function showModal(modal) {
    modal.classList.add('active');
}

/**
 * Hide a modal
 * 
 * @param {HTMLElement} modal - Modal element to hide
 */
function hideModal(modal) {
    modal.classList.remove('active');
}

/**
 * Display a toast notification
 * 
 * @param {string} message - Notification message
 * @param {string} type - Notification type (success, error, info, warning)
 */
function showNotification(message, type = 'info') {
    // Check if the container exists, create it if not
    let container = document.querySelector('.notification-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'notification-container';
        document.body.appendChild(container);
    }
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    
    // Create notification content
    const content = document.createElement('div');
    content.className = 'notification-content';
    
    // Add icon based on type
    const icon = document.createElement('i');
    switch (type) {
        case 'success':
            icon.className = 'fas fa-check-circle';
            break;
        case 'error':
            icon.className = 'fas fa-exclamation-circle';
            break;
        case 'warning':
            icon.className = 'fas fa-exclamation-triangle';
            break;
        default:
            icon.className = 'fas fa-info-circle';
    }
    
    // Add text
    const text = document.createElement('span');
    text.textContent = message;
    
    // Add close button
    const closeBtn = document.createElement('button');
    closeBtn.innerHTML = '&times;';
    closeBtn.className = 'notification-close';
    closeBtn.addEventListener('click', () => {
        notification.classList.add('hiding');
        setTimeout(() => {
            notification.remove();
        }, 300);
    });
    
    // Assemble notification
    content.appendChild(icon);
    content.appendChild(text);
    notification.appendChild(content);
    notification.appendChild(closeBtn);
    
    // Add to container
    container.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        notification.classList.add('hiding');
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 5000);
}

// ===== Navigation =====

/**
 * Navigate to a page
 * 
 * @param {string} page - Page to navigate to
 */
function navigateToPage(page) {
    // Update active nav item
    elements.navItems.forEach(item => {
        if (item.dataset.page === page) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
    
    // Update visible page
    elements.pages.forEach(pageEl => {
        if (pageEl.id === page) {
            pageEl.classList.add('active');
        } else {
            pageEl.classList.remove('active');
        }
    });
    
    // Update page title
    elements.pageTitle.textContent = page.charAt(0).toUpperCase() + page.slice(1);
    
    // Update current page in state
    state.currentPage = page;
    
    // Load page data
    loadPageData(page);
}

/**
 * Toggle sidebar visibility
 */
function toggleSidebar() {
    elements.sidebar.classList.toggle('collapsed');
}

// ===== Data Loading =====

/**
 * Load data for the current page
 * 
 * @param {string} page - Page to load data for
 */
async function loadPageData(page) {
    try {
        switch (page) {
            case 'dashboard':
                await loadDashboardData();
                break;
            case 'training':
                await loadAllTrainings();
                break;
            case 'exports':
                await loadAllExports();
                break;
            case 'datasets':
                await loadDatasets();
                break;
            case 'settings':
                loadSettings();
                break;
        }
    } catch (error) {
        console.error(`Error loading data for ${page}:`, error);
        showNotification(`Error loading data: ${error.message}`, 'error');
    }
}

/**
 * Load dashboard data
 */
async function loadDashboardData() {
    // Load health status
    await updateSystemStatus();
    
    // Load active training jobs
    const trainingsResponse = await API.listTrainings();
    const trainings = trainingsResponse.data || [];
    
    // Filter active and completed jobs
    const activeJobs = trainings.filter(job => ['pending', 'running'].includes(job.status));
    const completedJobs = trainings.filter(job => job.status === 'completed');
    
    // Update counts
    elements.activeJobsCount.textContent = activeJobs.length;
    elements.completedJobsCount.textContent = completedJobs.length;
    
    // Update active jobs table
    clearElement(elements.activeJobsTable);
    
    if (activeJobs.length > 0) {
        hideElement(elements.noActiveJobs);
        
        activeJobs.forEach(job => {
            const row = document.createElement('tr');
            
            // Format row
            row.innerHTML = `
                <td>${job.id.substring(0, 8)}...</td>
                <td><div class="status-badge ${job.status}">${job.status}</div></td>
                <td>
                    <div class="progress-container">
                        <div class="progress">
                            <div class="progress-bar" style="width: ${job.progress || 0}%"></div>
                        </div>
                        <div class="progress-text">${Math.round(job.progress || 0)}%</div>
                    </div>
                </td>
                <td>${formatRelativeTime(job.updated_at)}</td>
                <td>
                    <button class="action-btn view" data-id="${job.id}" title="View Details">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button class="action-btn stop" data-id="${job.id}" title="Stop Training">
                        <i class="fas fa-stop"></i>
                    </button>
                </td>
            `;
            
            // Add event listeners
            row.querySelector('.action-btn.view').addEventListener('click', () => {
                viewTrainingDetails(job.id);
            });
            
            row.querySelector('.action-btn.stop').addEventListener('click', () => {
                stopTraining(job.id);
            });
            
            elements.activeJobsTable.appendChild(row);
        });
    } else {
        showElement(elements.noActiveJobs);
    }
    
    // Load exports
    const exportsResponse = await API.listExports();
    const exports = exportsResponse.data || [];
    
    // Update counts
    elements.exportsCount.textContent = exports.length;
    
    // Update recent exports table
    clearElement(elements.recentExportsTable);
    
    if (exports.length > 0) {
        hideElement(elements.noRecentExports);
        
        // Get 5 most recent exports
        const recentExports = exports.slice(0, 5);
        
        recentExports.forEach(exportItem => {
            const row = document.createElement('tr');
            
            // Format row
            row.innerHTML = `
                <td>${exportItem.id.substring(0, 8)}...</td>
                <td>${exportItem.config?.model_id?.substring(0, 8) || 'N/A'}</td>
                <td>${exportItem.config?.format || 'N/A'}</td>
                <td><div class="status-badge ${exportItem.status}">${exportItem.status}</div></td>
                <td>
                    <button class="action-btn view" data-id="${exportItem.id}" title="View Details">
                        <i class="fas fa-eye"></i>
                    </button>
                    ${exportItem.status === 'completed' ? `
                        <a href="${API.getExportDownloadUrl(exportItem.id)}" class="action-btn download" title="Download">
                            <i class="fas fa-download"></i>
                        </a>
                    ` : ''}
                </td>
            `;
            
            // Add event listeners
            row.querySelector('.action-btn.view').addEventListener('click', () => {
                viewExportDetails(exportItem.id);
            });
            
            elements.recentExportsTable.appendChild(row);
        });
    } else {
        showElement(elements.noRecentExports);
    }
}

/**
 * Load all training jobs
 */
async function loadAllTrainings() {
    const response = await API.listTrainings();
    const trainings = response.data || [];
    
    clearElement(elements.allJobsTable);
    
    if (trainings.length > 0) {
        hideElement(elements.noAllJobs);
        
        trainings.forEach(job => {
            const row = document.createElement('tr');
            
            // Format row
            row.innerHTML = `
                <td>${job.id.substring(0, 8)}...</td>
                <td>${job.config?.model_size || 'N/A'}</td>
                <td><div class="status-badge ${job.status}">${job.status}</div></td>
                <td>
                    <div class="progress-container">
                        <div class="progress">
                            <div class="progress-bar" style="width: ${job.progress || 0}%"></div>
                        </div>
                        <div class="progress-text">${Math.round(job.progress || 0)}%</div>
                    </div>
                </td>
                <td>${formatRelativeTime(job.created_at)}</td>
                <td>
                    <button class="action-btn view" data-id="${job.id}" title="View Details">
                        <i class="fas fa-eye"></i>
                    </button>
                    ${['pending', 'running'].includes(job.status) ? `
                        <button class="action-btn stop" data-id="${job.id}" title="Stop Training">
                            <i class="fas fa-stop"></i>
                        </button>
                    ` : ''}
                    ${job.status === 'completed' ? `
                        <button class="action-btn export" data-id="${job.id}" title="Export Model">
                            <i class="fas fa-file-export"></i>
                        </button>
                    ` : ''}
                </td>
            `;
            
            // Add event listeners
            row.querySelector('.action-btn.view').addEventListener('click', () => {
                viewTrainingDetails(job.id);
            });
            
            if (['pending', 'running'].includes(job.status)) {
                row.querySelector('.action-btn.stop').addEventListener('click', () => {
                    stopTraining(job.id);
                });
            }
            
            if (job.status === 'completed') {
                row.querySelector('.action-btn.export').addEventListener('click', () => {
                    openExportModal(job.id);
                });
            }
            
            elements.allJobsTable.appendChild(row);
        });
    } else {
        showElement(elements.noAllJobs);
    }
}

/**
 * Load all exports
 */
async function loadAllExports() {
    const response = await API.listExports();
    const exports = response.data || [];
    
    clearElement(elements.allExportsTable);
    
    if (exports.length > 0) {
        hideElement(elements.noAllExports);
        
        exports.forEach(exportItem => {
            const row = document.createElement('tr');
            
            // Format row
            row.innerHTML = `
                <td>${exportItem.id.substring(0, 8)}...</td>
                <td>${exportItem.config?.model_id?.substring(0, 8) || 'N/A'}</td>
                <td>${exportItem.config?.format || 'N/A'}</td>
                <td><div class="status-badge ${exportItem.status}">${exportItem.status}</div></td>
                <td>${formatRelativeTime(exportItem.created_at)}</td>
                <td>
                    <button class="action-btn view" data-id="${exportItem.id}" title="View Details">
                        <i class="fas fa-eye"></i>
                    </button>
                    ${exportItem.status === 'completed' ? `
                        <a href="${API.getExportDownloadUrl(exportItem.id)}" class="action-btn download" title="Download">
                            <i class="fas fa-download"></i>
                        </a>
                    ` : ''}
                    <button class="action-btn delete" data-id="${exportItem.id}" title="Delete">
                        <i class="fas fa-trash"></i>
                    </button>
                </td>
            `;
            
            // Add event listeners
            row.querySelector('.action-btn.view').addEventListener('click', () => {
                viewExportDetails(exportItem.id);
            });
            
            row.querySelector('.action-btn.delete').addEventListener('click', () => {
                // TODO: Implement delete export functionality
                showNotification('Delete functionality not implemented yet', 'info');
            });
            
            elements.allExportsTable.appendChild(row);
        });
    } else {
        showElement(elements.noAllExports);
    }
}

/**
 * Load available datasets
 */
async function loadDatasets() {
    const response = await API.listDatasets();
    const datasets = response.data || {};
    
    // Combine core and additional datasets
    const allDatasets = [
        ...(datasets.core || []),
        ...(datasets.additional || [])
    ];
    
    clearElement(elements.datasetsTable);
    
    if (allDatasets.length > 0) {
        hideElement(elements.noDatasets);
        
        allDatasets.forEach(dataset => {
            const row = document.createElement('tr');
            
            // Format row
            row.innerHTML = `
                <td>${dataset.name}</td>
                <td>${datasets.core?.includes(dataset) ? 'Core' : 'Additional'}</td>
                <td>${dataset.streaming ? 'Yes' : 'No'}</td>
                <td>${dataset.max_samples || 'All'}</td>
            `;
            
            elements.datasetsTable.appendChild(row);
        });
    } else {
        showElement(elements.noDatasets);
    }
}

/**
 * Load settings
 */
function loadSettings() {
    // Load settings from local storage
    const savedSettings = localStorage.getItem('settings');
    if (savedSettings) {
        Object.assign(state.settings, JSON.parse(savedSettings));
    }
    
    // Update form fields
    elements.maxMemory.value = state.settings.maxMemory;
    elements.defaultModelSize.value = state.settings.defaultModelSize;
    elements.defaultBatchSize.value = state.settings.defaultBatchSize;
    elements.defaultExportFormat.value = state.settings.defaultExportFormat;
    elements.defaultQuantize.checked = state.settings.defaultQuantize;
    elements.enableRealtime.checked = state.settings.realTimeUpdates !== false;
    elements.updateInterval.value = state.settings.updateInterval;
}

/**
 * Save settings
 */
function saveSettings() {
    // Update settings
    state.settings.maxMemory = parseInt(elements.maxMemory.value);
    state.settings.defaultModelSize = elements.defaultModelSize.value;
    state.settings.defaultBatchSize = parseInt(elements.defaultBatchSize.value);
    state.settings.defaultExportFormat = elements.defaultExportFormat.value;
    state.settings.defaultQuantize = elements.defaultQuantize.checked;
    state.settings.realTimeUpdates = elements.enableRealtime.checked;
    state.settings.updateInterval = parseInt(elements.updateInterval.value);
    
    // Update real-time setting
    state.realTimeUpdates = elements.enableRealtime.checked;
    
    // Update refresh rate
    state.refreshRate = parseInt(elements.updateInterval.value) * 1000;
    
    // Save to local storage
    localStorage.setItem('settings', JSON.stringify(state.settings));
    
    showNotification('Settings saved successfully', 'success');
}

/**
 * Reset settings to defaults
 */
function resetSettings() {
    // Reset state settings
    state.settings = {
        maxMemory: 1024,
        defaultModelSize: 'small',
        defaultBatchSize: 8,
        defaultExportFormat: 'flask',
        defaultQuantize: true,
        realTimeUpdates: true,
        updateInterval: 5
    };
    
    // Update real-time setting
    state.realTimeUpdates = true;
    
    // Update refresh rate
    state.refreshRate = 5000;
    
    // Update form fields
    elements.maxMemory.value = state.settings.maxMemory;
    elements.defaultModelSize.value = state.settings.defaultModelSize;
    elements.defaultBatchSize.value = state.settings.defaultBatchSize;
    elements.defaultExportFormat.value = state.settings.defaultExportFormat;
    elements.defaultQuantize.checked = state.settings.defaultQuantize;
    elements.enableRealtime.checked = state.settings.realTimeUpdates;
    elements.updateInterval.value = state.settings.updateInterval;
    
    // Save to local storage
    localStorage.setItem('settings', JSON.stringify(state.settings));
    
    showNotification('Settings reset to defaults', 'success');
}

/**
 * Update system status indicators
 */
async function updateSystemStatus() {
    try {
        // Get health status
        const healthResponse = await API.getHealthStatus();
        
        // Update status indicator
        elements.statusIndicator.className = 'status-indicator online';
        elements.statusText.textContent = 'System Online';
        
        // Update memory usage
        if (healthResponse.memory && healthResponse.memory.percent) {
            elements.memoryUsage.textContent = `${Math.round(healthResponse.memory.percent)}%`;
        }
    } catch (error) {
        console.error('Error getting health status:', error);
        
        // Update status indicator
        elements.statusIndicator.className = 'status-indicator offline';
        elements.statusText.textContent = 'System Offline';
    }
}

// ===== Training Details =====

/**
 * View training details
 * 
 * @param {string} trainingId - Training job ID
 */
async function viewTrainingDetails(trainingId) {
    try {
        state.selectedTrainingId = trainingId;
        
        // Get training status
        const response = await API.getTrainingStatus(trainingId);
        const training = response.data;
        
        if (!training) {
            showNotification('Training job not found', 'error');
            return;
        }
        
        // Update modal content
        document.getElementById('training-id').querySelector('span').textContent = trainingId;
        document.getElementById('training-status-badge').className = `status-badge ${training.status}`;
        document.getElementById('training-status-badge').textContent = training.status;
        document.getElementById('training-progress-bar').style.width = `${training.progress || 0}%`;
        document.getElementById('training-progress-text').textContent = `${Math.round(training.progress || 0)}%`;
        document.getElementById('training-status-message').textContent = training.message || 'No status message';
        
        // Update details
        document.getElementById('detail-model-size').textContent = training.config?.model_size || 'N/A';
        document.getElementById('detail-datasets').textContent = training.config?.datasets?.map(d => d.name).join(', ') || 'N/A';
        document.getElementById('detail-started').textContent = formatTimestamp(training.created_at) || 'N/A';
        
        // Calculate elapsed time
        if (training.created_at) {
            const startTime = new Date(training.created_at * 1000);
            const elapsedMs = training.status === 'completed' 
                ? new Date(training.updated_at * 1000) - startTime
                : Date.now() - startTime;
            
            const hours = Math.floor(elapsedMs / (1000 * 60 * 60));
            const minutes = Math.floor((elapsedMs % (1000 * 60 * 60)) / (1000 * 60));
            
            document.getElementById('detail-elapsed').textContent = `${hours}h ${minutes}m`;
        } else {
            document.getElementById('detail-elapsed').textContent = 'N/A';
        }
        
        // Update metrics
        document.getElementById('detail-loss').textContent = training.metrics?.loss?.toFixed(4) || 'N/A';
        document.getElementById('detail-lr').textContent = training.metrics?.learning_rate?.toExponential(2) || 'N/A';
        
        // Update action buttons
        const stopBtn = document.getElementById('stop-training-btn');
        const exportBtn = document.getElementById('export-model-btn');
        
        if (['pending', 'running'].includes(training.status)) {
            stopBtn.style.display = 'block';
            exportBtn.style.display = 'none';
        } else if (training.status === 'completed') {
            stopBtn.style.display = 'none';
            exportBtn.style.display = 'block';
        } else {
            stopBtn.style.display = 'none';
            exportBtn.style.display = 'none';
        }
        
        // Show modal
        showModal(elements.trainingDetailsModal);
        
        // Start real-time updates
        if (state.realTimeUpdates) {
            initMetricsChart(trainingId);
            API.subscribeToUpdates(trainingId, handleTrainingUpdate);
        }
    } catch (error) {
        console.error('Error viewing training details:', error);
        showNotification(`Error viewing training details: ${error.message}`, 'error');
    }
}

/**
 * Initialize metrics chart
 * 
 * @param {string} trainingId - Training job ID
 */
async function initMetricsChart(trainingId) {
    try {
        // Get metrics data
        const response = await API.getTrainingMetrics(trainingId);
        
        if (!response.data || !response.data.time_series) {
            return;
        }
        
        const timeSeriesData = response.data.time_series;
        
        // If chart already exists, destroy it
        if (state.metricsChart) {
            state.metricsChart.destroy();
        }
        
        // Get canvas context
        const ctx = document.getElementById('metrics-chart').getContext('2d');
        
        // Create chart
        state.metricsChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: timeSeriesData.timestamps.map(t => {
                    const minutes = Math.floor((t - timeSeriesData.timestamps[0]) / 60);
                    return `${minutes}m`;
                }),
                datasets: [
                    {
                        label: 'Loss',
                        data: timeSeriesData.loss,
                        borderColor: '#3f51b5',
                        backgroundColor: 'rgba(63, 81, 181, 0.1)',
                        tension: 0.3,
                        pointRadius: 2,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Loss'
                        },
                        beginAtZero: true
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                }
            }
        });
        
        // Switch between metrics
        document.querySelectorAll('.chart-toggle').forEach(button => {
            button.addEventListener('click', () => {
                // Update active state
                document.querySelectorAll('.chart-toggle').forEach(btn => {
                    btn.classList.remove('active');
                });
                button.classList.add('active');
                
                // Update chart data
                const chartType = button.dataset.chart;
                
                // Update dataset
                if (chartType === 'loss') {
                    state.metricsChart.data.datasets[0].label = 'Loss';
                    state.metricsChart.data.datasets[0].data = timeSeriesData.loss;
                    state.metricsChart.options.scales.y.title.text = 'Loss';
                } else if (chartType === 'lr') {
                    state.metricsChart.data.datasets[0].label = 'Learning Rate';
                    state.metricsChart.data.datasets[0].data = timeSeriesData.learning_rate;
                    state.metricsChart.options.scales.y.title.text = 'Learning Rate';
                }
                
                state.metricsChart.update();
            });
        });
    } catch (error) {
        console.error('Error initializing metrics chart:', error);
    }
}

/**
 * Handle real-time training updates
 * 
 * @param {Object} data - Update data
 */
function handleTrainingUpdate(data) {
    if (data.type === 'metrics_update' || data.type === 'initial_metrics') {
        const update = data.data;
        
        // Update status badge
        document.getElementById('training-status-badge').className = `status-badge ${update.status}`;
        document.getElementById('training-status-badge').textContent = update.status;
        
        // Update progress
        document.getElementById('training-progress-bar').style.width = `${update.progress || 0}%`;
        document.getElementById('training-progress-text').textContent = `${Math.round(update.progress || 0)}%`;
        
        // Update status message
        document.getElementById('training-status-message').textContent = update.message || 'No status message';
        
        // Update metrics
        if (update.metrics) {
            document.getElementById('detail-loss').textContent = update.metrics.loss?.toFixed(4) || 'N/A';
            document.getElementById('detail-lr').textContent = update.metrics.learning_rate?.toExponential(2) || 'N/A';
            
            // Update chart if it exists
            if (state.metricsChart) {
                // Add new data point
                const timestamps = state.metricsChart.data.labels;
                const newTimestamp = Math.floor((update.timestamp - timestamps[0]) / 60) + 'm';
                
                // Add new data if timestamp doesn't already exist
                if (!timestamps.includes(newTimestamp)) {
                    state.metricsChart.data.labels.push(newTimestamp);
                    
                    // Add data to active dataset
                    const activeDataset = document.querySelector('.chart-toggle.active').dataset.chart;
                    
                    if (activeDataset === 'loss' && update.metrics.loss) {
                        state.metricsChart.data.datasets[0].data.push(update.metrics.loss);
                    } else if (activeDataset === 'lr' && update.metrics.learning_rate) {
                        state.metricsChart.data.datasets[0].data.push(update.metrics.learning_rate);
                    }
                    
                    state.metricsChart.update();
                }
            }
        }
        
        // Update action buttons
        const stopBtn = document.getElementById('stop-training-btn');
        const exportBtn = document.getElementById('export-model-btn');
        
        if (['pending', 'running'].includes(update.status)) {
            stopBtn.style.display = 'block';
            exportBtn.style.display = 'none';
        } else if (update.status === 'completed') {
            stopBtn.style.display = 'none';
            exportBtn.style.display = 'block';
        } else {
            stopBtn.style.display = 'none';
            exportBtn.style.display = 'none';
        }
    }
}

/**
 * Stop a training job
 * 
 * @param {string} trainingId - Training job ID
 */
async function stopTraining(trainingId) {
    try {
        if (state.pendingActions[trainingId]) {
            return;
        }
        
        state.pendingActions[trainingId] = true;
        
        // Confirm stop
        if (!confirm('Are you sure you want to stop this training job?')) {
            delete state.pendingActions[trainingId];
            return;
        }
        
        // Stop training
        await API.stopTraining(trainingId);
        
        showNotification('Training job stopping', 'success');
        
        // Refresh data
        loadPageData(state.currentPage);
        
        // If details modal is open, update status
        if (state.selectedTrainingId === trainingId) {
            const response = await API.getTrainingStatus(trainingId);
            
            if (response.data) {
                document.getElementById('training-status-badge').className = `status-badge ${response.data.status}`;
                document.getElementById('training-status-badge').textContent = response.data.status;
                document.getElementById('training-status-message').textContent = response.data.message || 'No status message';
                
                // Update action buttons
                document.getElementById('stop-training-btn').style.display = 'none';
            }
        }
    } catch (error) {
        console.error('Error stopping training:', error);
        showNotification(`Error stopping training: ${error.message}`, 'error');
    } finally {
        delete state.pendingActions[trainingId];
    }
}

/**
 * Start a new training job
 */
async function startTraining() {
    try {
        // Get form values
        const modelSize = document.getElementById('model-size').value;
        const epochs = parseInt(document.getElementById('epochs').value);
        const batchSize = parseInt(document.getElementById('batch-size').value);
        const learningRate = parseFloat(document.getElementById('learning-rate').value);
        const saveSteps = parseInt(document.getElementById('save-steps').value);
        
        // Get selected datasets
        const selectedDatasets = [];
        document.querySelectorAll('.dataset-checkbox:checked').forEach(checkbox => {
            selectedDatasets.push(JSON.parse(checkbox.value));
        });
        
        if (selectedDatasets.length === 0) {
            showNotification('Please select at least one dataset', 'error');
            return;
        }
        
        // Create training config
        const config = {
            model_size: modelSize,
            datasets: selectedDatasets,
            epochs: epochs,
            batch_size: batchSize,
            learning_rate: learningRate,
            save_steps: saveSteps
        };
        
        // Start training
        const response = await API.startTraining(config);
        
        // Hide modal
        hideModal(elements.newTrainingModal);
        
        showNotification('Training job started successfully', 'success');
        
        // Refresh data
        loadPageData(state.currentPage);
        
        // View training details
        setTimeout(() => {
            viewTrainingDetails(response.data.training_id);
        }, 1000);
    } catch (error) {
        console.error('Error starting training:', error);
        showNotification(`Error starting training: ${error.message}`, 'error');
    }
}

// ===== Export Details =====

/**
 * View export details
 * 
 * @param {string} exportId - Export job ID
 */
async function viewExportDetails(exportId) {
    try {
        state.selectedExportId = exportId;
        
        // Get export status
        const response = await API.getExportStatus(exportId);
        const exportItem = response.data;
        
        if (!exportItem) {
            showNotification('Export job not found', 'error');
            return;
        }
        
        // Update modal content
        document.getElementById('export-id').querySelector('span').textContent = exportId;
        document.getElementById('export-status-badge').className = `status-badge ${exportItem.status}`;
        document.getElementById('export-status-badge').textContent = exportItem.status;
        document.getElementById('export-progress-bar').style.width = `${exportItem.progress || 0}%`;
        document.getElementById('export-progress-text').textContent = `${Math.round(exportItem.progress || 0)}%`;
        document.getElementById('export-status-message').textContent = exportItem.message || 'No status message';
        
        // Update details
        document.getElementById('detail-model-id').textContent = exportItem.config?.model_id || 'N/A';
        document.getElementById('detail-format').textContent = exportItem.config?.format || 'N/A';
        document.getElementById('detail-quantized').textContent = exportItem.config?.quantize ? 'Yes' : 'No';
        document.getElementById('detail-created').textContent = formatTimestamp(exportItem.created_at) || 'N/A';
        
        // Update file size if available
        if (exportItem.result && exportItem.result.export_size) {
            document.getElementById('detail-size').textContent = formatBytes(exportItem.result.export_size);
        } else {
            document.getElementById('detail-size').textContent = 'N/A';
        }
        
        // Update download button
        const downloadBtn = document.getElementById('download-export-btn');
        
        if (exportItem.status === 'completed') {
            downloadBtn.disabled = false;
            downloadBtn.onclick = () => {
                window.location.href = API.getExportDownloadUrl(exportId);
            };
        } else {
            downloadBtn.disabled = true;
        }
        
        // Show modal
        showModal(elements.exportDetailsModal);
    } catch (error) {
        console.error('Error viewing export details:', error);
        showNotification(`Error viewing export details: ${error.message}`, 'error');
    }
}

/**
 * Open export modal for a model
 * 
 * @param {string} modelId - Model ID
 */
function openExportModal(modelId) {
    // Set selected model
    document.getElementById('export-model-id').value = modelId;
    
    // Reset form
    document.getElementById('export-format').value = state.settings.defaultExportFormat;
    document.getElementById('quantize').checked = state.settings.defaultQuantize;
    document.getElementById('optimization-level').value = '1';
    
    // Show modal
    showModal(elements.newExportModal);
}

/**
 * Create new export
 */
async function createExport() {
    try {
        // Get form values
        const modelId = document.getElementById('export-model-id').value;
        const format = document.getElementById('export-format').value;
        const quantize = document.getElementById('quantize').checked;
        const optimizationLevel = parseInt(document.getElementById('optimization-level').value);
        
        // Create export config
        const config = {
            model_id: modelId,
            format: format,
            quantize: quantize,
            optimization_level: optimizationLevel
        };
        
        // Request export
        const response = await API.requestExport(config);
        
        // Hide modal
        hideModal(elements.newExportModal);
        
        showNotification('Export job started successfully', 'success');
        
        // Refresh data
        loadPageData(state.currentPage);
        
        // View export details
        setTimeout(() => {
            viewExportDetails(response.data.export_id);
        }, 1000);
    } catch (error) {
        console.error('Error creating export:', error);
        showNotification(`Error creating export: ${error.message}`, 'error');
    }
}

// ===== Modal Handling =====

/**
 * Initialize modal handling
 */
function initModals() {
    // Close modals when clicking outside or on close button
    document.querySelectorAll('.modal').forEach(modal => {
        // Close when clicking on the background
        modal.addEventListener('click', (event) => {
            if (event.target === modal) {
                hideModal(modal);
                
                // Unsubscribe from updates if training details modal
                if (modal === elements.trainingDetailsModal && state.selectedTrainingId) {
                    API.unsubscribeFromUpdates(state.selectedTrainingId);
                    state.selectedTrainingId = null;
                }
                
                // Reset selected export
                if (modal === elements.exportDetailsModal) {
                    state.selectedExportId = null;
                }
            }
        });
        
        // Close when clicking close button
        modal.querySelector('.modal-close').addEventListener('click', () => {
            hideModal(modal);
            
            // Unsubscribe from updates if training details modal
            if (modal === elements.trainingDetailsModal && state.selectedTrainingId) {
                API.unsubscribeFromUpdates(state.selectedTrainingId);
                state.selectedTrainingId = null;
            }
            
            // Reset selected export
            if (modal === elements.exportDetailsModal) {
                state.selectedExportId = null;
            }
        });
    });
    
    // Training modals
    elements.newTrainingBtn.addEventListener('click', () => {
        // Populate datasets selector
        populateDatasetsSelector();
        
        // Reset form
        document.getElementById('model-size').value = state.settings.defaultModelSize;
        document.getElementById('epochs').value = '3';
        document.getElementById('batch-size').value = state.settings.defaultBatchSize.toString();
        document.getElementById('learning-rate').value = '0.00003';
        document.getElementById('save-steps').value = '1000';
        
        // Show modal
        showModal(elements.newTrainingModal);
    });
    
    document.getElementById('cancel-training-btn').addEventListener('click', () => {
        hideModal(elements.newTrainingModal);
    });
    
    document.getElementById('new-training-form').addEventListener('submit', (event) => {
        event.preventDefault();
        startTraining();
    });
    
    document.getElementById('stop-training-btn').addEventListener('click', () => {
        if (state.selectedTrainingId) {
            stopTraining(state.selectedTrainingId);
        }
    });
    
    document.getElementById('export-model-btn').addEventListener('click', () => {
        if (state.selectedTrainingId) {
            // Hide training details modal
            hideModal(elements.trainingDetailsModal);
            
            // Unsubscribe from updates
            API.unsubscribeFromUpdates(state.selectedTrainingId);
            
            // Open export modal
            openExportModal(state.selectedTrainingId);
        }
    });
    
    // Export modals
    elements.newExportBtn.addEventListener('click', async () => {
        // Populate models dropdown
        await populateModelsDropdown();
        
        // Reset form
        document.getElementById('export-format').value = state.settings.defaultExportFormat;
        document.getElementById('quantize').checked = state.settings.defaultQuantize;
        document.getElementById('optimization-level').value = '1';
        
        // Show modal
        showModal(elements.newExportModal);
    });
    
    document.getElementById('cancel-export-btn').addEventListener('click', () => {
        hideModal(elements.newExportModal);
    });
    
    document.getElementById('new-export-form').addEventListener('submit', (event) => {
        event.preventDefault();
        createExport();
    });
}

/**
 * Populate datasets selector
 */
async function populateDatasetsSelector() {
    const datasetsSelector = document.getElementById('datasets-selector');
    
    // Show loading
    datasetsSelector.innerHTML = `
        <div class="loading-placeholder">
            <i class="fas fa-spinner fa-spin"></i> Loading available datasets...
        </div>
    `;
    
    try {
        // Get datasets
        const response = await API.listDatasets();
        const datasets = response.data || {};
        
        // Combine core and additional datasets
        const allDatasets = [
            ...(datasets.core || []).map(d => ({ ...d, type: 'Core' })),
            ...(datasets.additional || []).map(d => ({ ...d, type: 'Additional' }))
        ];
        
        // Clear selector
        clearElement(datasetsSelector);
        
        if (allDatasets.length > 0) {
            allDatasets.forEach(dataset => {
                const datasetItem = document.createElement('div');
                datasetItem.className = 'dataset-item';
                
                datasetItem.innerHTML = `
                    <input type="checkbox" class="dataset-checkbox" id="dataset-${dataset.name.replace(/\//g, '-')}" value='${JSON.stringify(dataset)}'>
                    <div class="dataset-info">
                        <div class="dataset-name">${dataset.name}</div>
                        <div class="dataset-meta">
                            ${dataset.type} dataset | 
                            ${dataset.streaming ? 'Streaming' : 'Static'} | 
                            Max samples: ${dataset.max_samples || 'All'}
                        </div>
                    </div>
                `;
                
                datasetsSelector.appendChild(datasetItem);
            });
            
            // Check first 3 datasets by default
            const checkboxes = datasetsSelector.querySelectorAll('.dataset-checkbox');
            for (let i = 0; i < Math.min(3, checkboxes.length); i++) {
                checkboxes[i].checked = true;
            }
        } else {
            datasetsSelector.innerHTML = `
                <div class="loading-placeholder">
                    <i class="fas fa-exclamation-circle"></i> No datasets available
                </div>
            `;
        }
    } catch (error) {
        console.error('Error loading datasets:', error);
        
        datasetsSelector.innerHTML = `
            <div class="loading-placeholder">
                <i class="fas fa-exclamation-circle"></i> Error loading datasets
            </div>
        `;
    }
}

/**
 * Populate models dropdown
 */
async function populateModelsDropdown() {
    const modelSelect = document.getElementById('export-model-id');
    
    // Clear options
    clearElement(modelSelect);
    
    // Add default option
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = 'Select a model';
    defaultOption.disabled = true;
    defaultOption.selected = true;
    modelSelect.appendChild(defaultOption);
    
    try {
        // Get completed training jobs
        const response = await API.listTrainings();
        const trainings = response.data || [];
        
        // Filter completed models
        const completedModels = trainings.filter(job => job.status === 'completed');
        
        if (completedModels.length > 0) {
            completedModels.forEach(model => {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = `${model.id.substring(0, 8)}... (${model.config?.model_size || 'Unknown'})`;
                modelSelect.appendChild(option);
            });
        } else {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No trained models available';
            option.disabled = true;
            modelSelect.appendChild(option);
        }
    } catch (error) {
        console.error('Error loading models:', error);
        
        const option = document.createElement('option');
        option.value = '';
        option.textContent = 'Error loading models';
        option.disabled = true;
        modelSelect.appendChild(option);
    }
}

// ===== Auto-refresh =====

/**
 * Start auto-refresh
 */
function startAutoRefresh() {
    // Clear existing interval if any
    if (state.refreshInterval) {
        clearInterval(state.refreshInterval);
    }
    
    // Set up refresh interval
    state.refreshInterval = setInterval(() => {
        // Only refresh if not viewing details
        if (!state.selectedTrainingId && !state.selectedExportId) {
            loadPageData(state.currentPage);
        }
        
        // Update system status
        updateSystemStatus();
    }, state.refreshRate);
}

// ===== Initialization =====

/**
 * Initialize the application
 */
function init() {
    // Load settings
    loadSettings();
    
    // Set up navigation
    elements.navItems.forEach(item => {
        item.addEventListener('click', () => {
            navigateToPage(item.dataset.page);
        });
    });
    
    // Set up sidebar toggle
    elements.sidebarToggle.addEventListener('click', toggleSidebar);
    
    // Initialize modals
    initModals();
    
    // Set up settings form
    elements.saveSettings.addEventListener('click', saveSettings);
    elements.resetSettings.addEventListener('click', resetSettings);
    
    // Load initial page data
    loadPageData(state.currentPage);
    
    // Start auto-refresh
    startAutoRefresh();
    
    // Set up refresh buttons
    document.getElementById('refresh-active-jobs').addEventListener('click', () => {
        loadDashboardData();
    });
    
    document.getElementById('refresh-recent-exports').addEventListener('click', () => {
        loadDashboardData();
    });
    
    document.getElementById('refresh-all-jobs').addEventListener('click', () => {
        loadAllTrainings();
    });
    
    document.getElementById('refresh-all-exports').addEventListener('click', () => {
        loadAllExports();
    });
    
    document.getElementById('refresh-datasets').addEventListener('click', () => {
        loadDatasets();
    });
}

// Initialize when document is loaded
document.addEventListener('DOMContentLoaded', init);
