/**
 * Background Script for Browser Extension
 * 
 * Handles extension lifecycle, storage, and communication between components.
 */

// Extension state
let extensionActive = true;
let analysisCount = 0;

/**
 * Initialize background script
 */
function initializeBackground() {
    console.log('Background script initialized');
    
    // Setup event listeners
    setupEventListeners();
    
    // Load initial state
    loadExtensionState();
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Extension installation/update
    chrome.runtime.onInstalled.addListener(handleInstallation);
    
    // Tab updates
    chrome.tabs.onUpdated.addListener(handleTabUpdate);
    
    // Storage changes
    chrome.storage.onChanged.addListener(handleStorageChange);
    
    // Messages from content scripts and popup
    chrome.runtime.onMessage.addListener(handleMessage);
    
    // Extension startup
    chrome.runtime.onStartup.addListener(handleStartup);
}

/**
 * Handle extension installation/update
 */
function handleInstallation(details) {
    console.log('Extension installed/updated:', details);
    
    if (details.reason === 'install') {
        // First time installation
        initializeDefaultSettings();
        showWelcomeNotification();
    } else if (details.reason === 'update') {
        // Extension updated
        handleUpdate(details.previousVersion);
    }
}

/**
 * Initialize default settings
 */
async function initializeDefaultSettings() {
    try {
        await chrome.storage.sync.set({
            extensionEnabled: true,
            sensitivity: 'medium',
            autoRun: true,
            pagesAnalyzed: 0,
            lastAnalysis: null,
            firstInstall: new Date().toISOString()
        });
        console.log('Default settings initialized');
    } catch (error) {
        console.error('Error initializing default settings:', error);
    }
}

/**
 * Show welcome notification
 */
function showWelcomeNotification() {
    chrome.notifications.create({
        type: 'basic',
        iconUrl: 'icons/icon48.png',
        title: 'My Extension Installed',
        message: 'Welcome! The extension is now active and ready to use.'
    });
}

/**
 * Handle extension update
 */
function handleUpdate(previousVersion) {
    console.log('Extension updated from version:', previousVersion);
    
    // Perform any necessary migrations
    migrateSettings(previousVersion);
    
    // Show update notification
    chrome.notifications.create({
        type: 'basic',
        iconUrl: 'icons/icon48.png',
        title: 'Extension Updated',
        message: 'My Extension has been updated to the latest version.'
    });
}

/**
 * Migrate settings from previous version
 */
async function migrateSettings(previousVersion) {
    try {
        const currentSettings = await chrome.storage.sync.get();
        
        // Add any new settings that might be missing
        const defaultSettings = {
            extensionEnabled: true,
            sensitivity: 'medium',
            autoRun: true
        };
        
        let needsUpdate = false;
        for (const [key, value] of Object.entries(defaultSettings)) {
            if (!(key in currentSettings)) {
                currentSettings[key] = value;
                needsUpdate = true;
            }
        }
        
        if (needsUpdate) {
            await chrome.storage.sync.set(currentSettings);
            console.log('Settings migrated successfully');
        }
    } catch (error) {
        console.error('Error migrating settings:', error);
    }
}

/**
 * Handle tab updates
 */
function handleTabUpdate(tabId, changeInfo, tab) {
    if (changeInfo.status === 'complete' && tab.url) {
        console.log('Tab updated:', tab.url);
        
        // Check if extension should run on this tab
        if (shouldRunOnTab(tab.url)) {
            // Send message to content script
            chrome.tabs.sendMessage(tabId, {
                action: 'tabUpdated',
                url: tab.url
            }).catch(error => {
                // Content script might not be ready yet
                console.log('Content script not ready for tab:', tabId);
            });
        }
    }
}

/**
 * Check if extension should run on this tab
 */
function shouldRunOnTab(url) {
    // Define patterns for URLs where extension should run
    const allowedPatterns = [
        /^https?:\/\/.*/,  // All HTTP/HTTPS URLs
    ];
    
    return allowedPatterns.some(pattern => pattern.test(url));
}

/**
 * Handle storage changes
 */
function handleStorageChange(changes, areaName) {
    console.log('Storage changed:', changes, areaName);
    
    // Handle specific setting changes
    if (changes.extensionEnabled) {
        extensionActive = changes.extensionEnabled.newValue;
        console.log('Extension active state changed:', extensionActive);
    }
    
    if (changes.sensitivity) {
        console.log('Sensitivity changed to:', changes.sensitivity.newValue);
    }
}

/**
 * Handle messages from content scripts and popup
 */
function handleMessage(message, sender, sendResponse) {
    console.log('Background received message:', message);
    
    switch (message.action) {
        case 'getExtensionState':
            sendResponse({
                active: extensionActive,
                analysisCount: analysisCount
            });
            break;
            
        case 'updateAnalysisCount':
            analysisCount++;
            sendResponse({ success: true, count: analysisCount });
            break;
            
        case 'getSettings':
            chrome.storage.sync.get().then(settings => {
                sendResponse({ success: true, settings });
            });
            return true; // Keep message channel open for async response
            
        case 'updateSettings':
            chrome.storage.sync.set(message.settings).then(() => {
                sendResponse({ success: true });
            });
            return true; // Keep message channel open for async response
            
        default:
            sendResponse({ success: false, error: 'Unknown action' });
    }
}

/**
 * Handle extension startup
 */
function handleStartup() {
    console.log('Extension started');
    loadExtensionState();
}

/**
 * Load extension state from storage
 */
async function loadExtensionState() {
    try {
        const settings = await chrome.storage.sync.get([
            'extensionEnabled',
            'sensitivity',
            'autoRun',
            'pagesAnalyzed'
        ]);
        
        extensionActive = settings.extensionEnabled !== false;
        analysisCount = settings.pagesAnalyzed || 0;
        
        console.log('Extension state loaded:', {
            active: extensionActive,
            sensitivity: settings.sensitivity,
            autoRun: settings.autoRun,
            analysisCount
        });
    } catch (error) {
        console.error('Error loading extension state:', error);
    }
}

/**
 * Get current tab information
 */
async function getCurrentTab() {
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        return tab;
    } catch (error) {
        console.error('Error getting current tab:', error);
        return null;
    }
}

/**
 * Execute content script on tab
 */
async function executeContentScript(tabId) {
    try {
        await chrome.scripting.executeScript({
            target: { tabId },
            files: ['contentScript.js']
        });
        console.log('Content script executed on tab:', tabId);
    } catch (error) {
        console.error('Error executing content script:', error);
    }
}

/**
 * Send message to all tabs
 */
async function broadcastMessage(message) {
    try {
        const tabs = await chrome.tabs.query({});
        const promises = tabs.map(tab => 
            chrome.tabs.sendMessage(tab.id, message).catch(() => {
                // Tab might not have content script
            })
        );
        await Promise.all(promises);
        console.log('Message broadcasted to all tabs');
    } catch (error) {
        console.error('Error broadcasting message:', error);
    }
}

// Initialize background script
initializeBackground();
