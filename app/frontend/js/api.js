/**
 * API client for interacting with the AI model training backend
 */

const API = {
    // Base URLs for API endpoints
    baseUrl: window.location.hostname === 'localhost' ? 'http://localhost:8000/api' : '/api',
    monitorUrl: window.location.hostname === 'localhost' ? 'http://localhost:8081' : '/monitor',
    
    // Cache for API responses
    cache: {
        datasets: null,
        models: null,
        trainings: null,
        exports: null
    },
    
    // Websocket connections for real-time updates
    sockets: {},
    
    /**
     * Make an API request
     * 
     * @param {string} url - URL to request
     * @param {Object} options - Fetch options
     * @returns {Promise} - Promise that resolves to the API response
     */
    async request(url, options = {}) {
        try {
            const response = await fetch(url, {
                ...options,
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                }
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.message || 'An error occurred');
            }
            
            return data;
        } catch (error) {
            console.error('API request error:', error);
            throw error;
        }
    },
    
    /**
     * Check API health status
     * 
     * @returns {Promise} - Promise that resolves to the health status
     */
    async getHealthStatus() {
        return this.request(`${this.baseUrl}/health`);
    },
    
    /**
     * Get monitor health status
     * 
     * @returns {Promise} - Promise that resolves to the monitor health status
     */
    async getMonitorHealth() {
        return this.request(`${this.monitorUrl}/health`);
    },
    
    // ===== Training API =====
    
    /**
     * Start a new training job
     * 
     * @param {Object} config - Training configuration
     * @returns {Promise} - Promise that resolves to the training job info
     */
    async startTraining(config) {
        return this.request(`${this.baseUrl}/training/start`, {
            method: 'POST',
            body: JSON.stringify(config)
        });
    },
    
    /**
     * Get training status
     * 
     * @param {string} trainingId - Training job ID
     * @returns {Promise} - Promise that resolves to the training status
     */
    async getTrainingStatus(trainingId) {
        return this.request(`${this.baseUrl}/training/status/${trainingId}`);
    },
    
    /**
     * Stop a training job
     * 
     * @param {string} trainingId - Training job ID
     * @returns {Promise} - Promise that resolves to the stop result
     */
    async stopTraining(trainingId) {
        return this.request(`${this.baseUrl}/training/stop/${trainingId}`, {
            method: 'POST'
        });
    },
    
    /**
     * List all training jobs
     * 
     * @returns {Promise} - Promise that resolves to the list of training jobs
     */
    async listTrainings() {
        const result = await this.request(`${this.baseUrl}/training/list`);
        this.cache.trainings = result.data;
        return result;
    },
    
    // ===== Export API =====
    
    /**
     * Request a model export
     * 
     * @param {Object} config - Export configuration
     * @returns {Promise} - Promise that resolves to the export job info
     */
    async requestExport(config) {
        return this.request(`${this.baseUrl}/export/request`, {
            method: 'POST',
            body: JSON.stringify(config)
        });
    },
    
    /**
     * Get export status
     * 
     * @param {string} exportId - Export job ID
     * @returns {Promise} - Promise that resolves to the export status
     */
    async getExportStatus(exportId) {
        return this.request(`${this.baseUrl}/export/status/${exportId}`);
    },
    
    /**
     * List all exports
     * 
     * @returns {Promise} - Promise that resolves to the list of exports
     */
    async listExports() {
        const result = await this.request(`${this.baseUrl}/export/list`);
        this.cache.exports = result.data;
        return result;
    },
    
    /**
     * Get URL for downloading an export
     * 
     * @param {string} exportId - Export job ID
     * @returns {string} - Download URL
     */
    getExportDownloadUrl(exportId) {
        return `${this.baseUrl}/export/download/${exportId}`;
    },
    
    // ===== Dataset API =====
    
    /**
     * List available datasets
     * 
     * @returns {Promise} - Promise that resolves to the list of datasets
     */
    async listDatasets() {
        const result = await this.request(`${this.baseUrl}/datasets/list`);
        this.cache.datasets = result.data;
        return result;
    },
    
    // ===== Monitoring API =====
    
    /**
     * Get training metrics
     * 
     * @param {string} trainingId - Training job ID
     * @returns {Promise} - Promise that resolves to the training metrics
     */
    async getTrainingMetrics(trainingId) {
        return this.request(`${this.baseUrl}/monitoring/metrics/${trainingId}`);
    },
    
    /**
     * Get charts for a training job
     * 
     * @param {string} trainingId - Training job ID
     * @returns {Promise} - Promise that resolves to the chart data
     */
    async getCharts(trainingId) {
        return this.request(`${this.monitorUrl}/charts/${trainingId}`);
    },
    
    /**
     * List jobs with metrics
     * 
     * @returns {Promise} - Promise that resolves to the list of jobs with metrics
     */
    async listJobsWithMetrics() {
        return this.request(`${this.monitorUrl}/jobs`);
    },
    
    /**
     * Subscribe to real-time updates for a training job
     * 
     * @param {string} trainingId - Training job ID
     * @param {Function} onMessage - Callback function for messages
     * @returns {WebSocket} - WebSocket connection
     */
    subscribeToUpdates(trainingId, onMessage) {
        // Close existing connection if any
        if (this.sockets[trainingId]) {
            this.sockets[trainingId].close();
        }
        
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.hostname === 'localhost' ? 'localhost:8081' : window.location.host;
        const wsUrl = `${protocol}//${host}/monitor/ws/${trainingId}`;
        
        const socket = new WebSocket(wsUrl);
        
        socket.onopen = () => {
            console.log(`WebSocket connected for training ${trainingId}`);
            
            // Send ping every 30 seconds to keep connection alive
            socket.pingInterval = setInterval(() => {
                if (socket.readyState === WebSocket.OPEN) {
                    socket.send('ping');
                }
            }, 30000);
        };
        
        socket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                onMessage(data);
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };
        
        socket.onclose = () => {
            console.log(`WebSocket disconnected for training ${trainingId}`);
            clearInterval(socket.pingInterval);
            delete this.sockets[trainingId];
        };
        
        socket.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
        
        this.sockets[trainingId] = socket;
        return socket;
    },
    
    /**
     * Unsubscribe from real-time updates
     * 
     * @param {string} trainingId - Training job ID
     */
    unsubscribeFromUpdates(trainingId) {
        if (this.sockets[trainingId]) {
            this.sockets[trainingId].close();
            delete this.sockets[trainingId];
        }
    },
    
    /**
     * Close all WebSocket connections
     */
    closeAllConnections() {
        Object.values(this.sockets).forEach(socket => {
            socket.close();
        });
        this.sockets = {};
    }
};

// Clean up WebSocket connections when the page is unloaded
window.addEventListener('beforeunload', () => {
    API.closeAllConnections();
});
