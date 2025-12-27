/**
 * Popup Script for Browser Extension
 * 
 * Handles the popup interface and user interactions.
 */

// DOM elements
const enableToggle = document.getElementById('enableToggle');
const sensitivitySelect = document.getElementById('sensitivity');
const autoRunCheckbox = document.getElementById('autoRun');
const analyzeBtn = document.getElementById('analyzeBtn');
const settingsBtn = document.getElementById('settingsBtn');
const statusIndicator = document.getElementById('statusIndicator');
const pagesAnalyzed = document.getElementById('pagesAnalyzed');
const lastAnalysis = document.getElementById('lastAnalysis');

/**
 * Initialize popup when DOM is loaded
 */
document.addEventListener('DOMContentLoaded', async () => {
    await loadSettings();
    await updateStatus();
    setupEventListeners();
});

/**
 * Load settings from storage
 */
async function loadSettings() {
    try {
        const result = await chrome.storage.sync.get([
            'extensionEnabled',
            'sensitivity',
            'autoRun',
            'pagesAnalyzed',
            'lastAnalysis'
        ]);
        
        enableToggle.checked = result.extensionEnabled !== false;
        sensitivitySelect.value = result.sensitivity || 'medium';
        autoRunCheckbox.checked = result.autoRun !== false;
        pagesAnalyzed.textContent = result.pagesAnalyzed || '0';
        lastAnalysis.textContent = result.lastAnalysis || 'Never';
        
    } catch (error) {
        console.error('Error loading settings:', error);
    }
}

/**
 * Update extension status
 */
async function updateStatus() {
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        const isEnabled = enableToggle.checked;
        
        if (isEnabled) {
            statusIndicator.className = 'status-indicator active';
            statusIndicator.querySelector('.status-text').textContent = 'Active';
        } else {
            statusIndicator.className = 'status-indicator inactive';
            statusIndicator.querySelector('.status-text').textContent = 'Inactive';
        }
    } catch (error) {
        console.error('Error updating status:', error);
    }
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Enable/disable toggle
    enableToggle.addEventListener('change', async (e) => {
        await saveSettings();
        await updateStatus();
        
        // Send message to content script
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            await chrome.tabs.sendMessage(tab.id, {
                action: 'toggleExtension',
                enabled: e.target.checked
            });
        } catch (error) {
            console.error('Error sending message to content script:', error);
        }
    });
    
    // Sensitivity change
    sensitivitySelect.addEventListener('change', async () => {
        await saveSettings();
        
        // Send message to content script
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            await chrome.tabs.sendMessage(tab.id, {
                action: 'updateSensitivity',
                sensitivity: sensitivitySelect.value
            });
        } catch (error) {
            console.error('Error sending sensitivity update:', error);
        }
    });
    
    // Auto-run toggle
    autoRunCheckbox.addEventListener('change', async () => {
        await saveSettings();
    });
    
    // Analyze button
    analyzeBtn.addEventListener('click', async () => {
        await analyzeCurrentPage();
    });
    
    // Settings button
    settingsBtn.addEventListener('click', () => {
        chrome.tabs.create({ url: 'chrome://extensions/' });
    });
}

/**
 * Save settings to storage
 */
async function saveSettings() {
    try {
        await chrome.storage.sync.set({
            extensionEnabled: enableToggle.checked,
            sensitivity: sensitivitySelect.value,
            autoRun: autoRunCheckbox.checked
        });
    } catch (error) {
        console.error('Error saving settings:', error);
    }
}

/**
 * Analyze current page
 */
async function analyzeCurrentPage() {
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        
        // Send message to content script to analyze
        const response = await chrome.tabs.sendMessage(tab.id, {
            action: 'analyzePage'
        });
        
        if (response && response.success) {
            // Update statistics
            const currentCount = parseInt(pagesAnalyzed.textContent) + 1;
            pagesAnalyzed.textContent = currentCount;
            lastAnalysis.textContent = new Date().toLocaleTimeString();
            
            // Save updated statistics
            await chrome.storage.sync.set({
                pagesAnalyzed: currentCount,
                lastAnalysis: new Date().toLocaleTimeString()
            });
            
            // Show success message
            showNotification('Page analyzed successfully!', 'success');
        } else {
            showNotification('Analysis failed. Please try again.', 'error');
        }
    } catch (error) {
        console.error('Error analyzing page:', error);
        showNotification('Error analyzing page. Please refresh and try again.', 'error');
    }
}

/**
 * Show notification to user
 * @param {string} message - Notification message
 * @param {string} type - Notification type (success, error, info)
 */
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    // Add to popup
    document.body.appendChild(notification);
    
    // Remove after 3 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    }, 3000);
}

/**
 * Handle messages from content script
 */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === 'updateStatus') {
        updateStatus();
    } else if (message.action === 'updateStats') {
        pagesAnalyzed.textContent = message.pagesAnalyzed || '0';
        lastAnalysis.textContent = message.lastAnalysis || 'Never';
    }
});
