/**
 * Content Script for Browser Extension
 * 
 * Runs in the context of web pages to analyze content and interact with the page.
 */

// Extension state
let extensionEnabled = true;
let sensitivity = 'medium';
let isAnalyzing = false;

/**
 * Initialize content script
 */
function initializeContentScript() {
    console.log('Content script initialized');
    
    // Load settings from storage
    loadSettings();
    
    // Setup message listeners
    setupMessageListeners();
    
    // Start page analysis if auto-run is enabled
    if (extensionEnabled) {
        analyzePageContent();
    }
}

/**
 * Load settings from storage
 */
async function loadSettings() {
    try {
        const result = await chrome.storage.sync.get([
            'extensionEnabled',
            'sensitivity',
            'autoRun'
        ]);
        
        extensionEnabled = result.extensionEnabled !== false;
        sensitivity = result.sensitivity || 'medium';
        
        console.log('Settings loaded:', { extensionEnabled, sensitivity });
    } catch (error) {
        console.error('Error loading settings:', error);
    }
}

/**
 * Setup message listeners for communication with popup and background
 */
function setupMessageListeners() {
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
        console.log('Content script received message:', message);
        
        switch (message.action) {
            case 'toggleExtension':
                extensionEnabled = message.enabled;
                console.log('Extension toggled:', extensionEnabled);
                sendResponse({ success: true });
                break;
                
            case 'updateSensitivity':
                sensitivity = message.sensitivity;
                console.log('Sensitivity updated:', sensitivity);
                sendResponse({ success: true });
                break;
                
            case 'analyzePage':
                analyzePageContent().then(result => {
                    sendResponse({ success: true, result });
                });
                return true; // Keep message channel open for async response
                
            default:
                sendResponse({ success: false, error: 'Unknown action' });
        }
    });
}

/**
 * Analyze page content
 */
async function analyzePageContent() {
    if (isAnalyzing) {
        console.log('Analysis already in progress');
        return;
    }
    
    isAnalyzing = true;
    console.log('Starting page analysis...');
    
    try {
        // Get page information
        const pageInfo = {
            url: window.location.href,
            title: document.title,
            domain: window.location.hostname,
            timestamp: new Date().toISOString()
        };
        
        // Analyze text content
        const textContent = extractTextContent();
        const analysis = await analyzeText(textContent);
        
        // Analyze images
        const imageAnalysis = analyzeImages();
        
        // Analyze links
        const linkAnalysis = analyzeLinks();
        
        // Combine results
        const results = {
            pageInfo,
            textAnalysis: analysis,
            imageAnalysis,
            linkAnalysis,
            summary: generateSummary(analysis, imageAnalysis, linkAnalysis)
        };
        
        console.log('Analysis complete:', results);
        
        // Show results to user
        displayResults(results);
        
        return results;
        
    } catch (error) {
        console.error('Error during analysis:', error);
        showNotification('Analysis failed: ' + error.message, 'error');
    } finally {
        isAnalyzing = false;
    }
}

/**
 * Extract text content from the page
 */
function extractTextContent() {
    // Remove script and style elements
    const elementsToRemove = document.querySelectorAll('script, style, noscript');
    elementsToRemove.forEach(el => el.remove());
    
    // Get all text content
    const textContent = document.body.innerText || document.body.textContent || '';
    
    // Clean up text
    return textContent
        .replace(/\s+/g, ' ')
        .trim()
        .substring(0, 10000); // Limit to 10k characters
}

/**
 * Analyze text content
 */
async function analyzeText(text) {
    const words = text.toLowerCase().split(/\s+/);
    const wordCount = words.length;
    
    // Basic text analysis
    const analysis = {
        wordCount,
        characterCount: text.length,
        sentences: text.split(/[.!?]+/).length,
        paragraphs: text.split(/\n\s*\n/).length,
        keywords: extractKeywords(words),
        sentiment: analyzeSentiment(text),
        readability: calculateReadability(text)
    };
    
    return analysis;
}

/**
 * Extract keywords from text
 */
function extractKeywords(words) {
    // Simple keyword extraction (in a real extension, you'd use more sophisticated methods)
    const stopWords = new Set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']);
    const wordFreq = {};
    
    words.forEach(word => {
        if (word.length > 3 && !stopWords.has(word)) {
            wordFreq[word] = (wordFreq[word] || 0) + 1;
        }
    });
    
    return Object.entries(wordFreq)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 10)
        .map(([word]) => word);
}

/**
 * Analyze sentiment of text
 */
function analyzeSentiment(text) {
    // Simple sentiment analysis (in a real extension, you'd use a proper sentiment analysis library)
    const positiveWords = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like'];
    const negativeWords = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'horrible', 'worst', 'disgusting'];
    
    const words = text.toLowerCase().split(/\s+/);
    let positiveCount = 0;
    let negativeCount = 0;
    
    words.forEach(word => {
        if (positiveWords.includes(word)) positiveCount++;
        if (negativeWords.includes(word)) negativeCount++;
    });
    
    const total = positiveCount + negativeCount;
    if (total === 0) return 'neutral';
    
    const score = (positiveCount - negativeCount) / total;
    if (score > 0.1) return 'positive';
    if (score < -0.1) return 'negative';
    return 'neutral';
}

/**
 * Calculate readability score
 */
function calculateReadability(text) {
    const sentences = text.split(/[.!?]+/).length;
    const words = text.split(/\s+/).length;
    const syllables = text.split(/\s+/).reduce((total, word) => total + countSyllables(word), 0);
    
    if (sentences === 0 || words === 0) return 0;
    
    // Simple Flesch Reading Ease formula
    const score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words));
    return Math.max(0, Math.min(100, score));
}

/**
 * Count syllables in a word
 */
function countSyllables(word) {
    word = word.toLowerCase();
    if (word.length <= 3) return 1;
    return word.replace(/[^aeiouy]/g, '').length;
}

/**
 * Analyze images on the page
 */
function analyzeImages() {
    const images = document.querySelectorAll('img');
    return {
        count: images.length,
        withAlt: Array.from(images).filter(img => img.alt).length,
        withoutAlt: Array.from(images).filter(img => !img.alt).length,
        totalSize: Array.from(images).reduce((total, img) => total + (img.naturalWidth * img.naturalHeight), 0)
    };
}

/**
 * Analyze links on the page
 */
function analyzeLinks() {
    const links = document.querySelectorAll('a[href]');
    const internalLinks = Array.from(links).filter(link => {
        const href = link.href;
        return href.startsWith(window.location.origin) || href.startsWith('/');
    });
    
    return {
        total: links.length,
        internal: internalLinks.length,
        external: links.length - internalLinks.length
    };
}

/**
 * Generate analysis summary
 */
function generateSummary(textAnalysis, imageAnalysis, linkAnalysis) {
    const summary = [];
    
    if (textAnalysis.wordCount > 1000) {
        summary.push('Long-form content');
    }
    
    if (imageAnalysis.withoutAlt > 0) {
        summary.push(`${imageAnalysis.withoutAlt} images missing alt text`);
    }
    
    if (linkAnalysis.external > linkAnalysis.internal) {
        summary.push('More external than internal links');
    }
    
    return summary.length > 0 ? summary.join(', ') : 'Standard web page';
}

/**
 * Display analysis results
 */
function displayResults(results) {
    // Create results overlay
    const overlay = document.createElement('div');
    overlay.id = 'extension-analysis-overlay';
    overlay.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        width: 300px;
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 10000;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 14px;
    `;
    
    overlay.innerHTML = `
        <div style="padding: 16px; border-bottom: 1px solid #eee;">
            <h3 style="margin: 0 0 8px 0; color: #333;">Page Analysis</h3>
            <p style="margin: 0; color: #666; font-size: 12px;">${results.summary}</p>
        </div>
        <div style="padding: 16px;">
            <div style="margin-bottom: 8px;">
                <strong>Words:</strong> ${results.textAnalysis.wordCount}
            </div>
            <div style="margin-bottom: 8px;">
                <strong>Images:</strong> ${results.imageAnalysis.count} (${results.imageAnalysis.withAlt} with alt text)
            </div>
            <div style="margin-bottom: 8px;">
                <strong>Links:</strong> ${results.linkAnalysis.total} (${results.linkAnalysis.internal} internal)
            </div>
            <div style="margin-bottom: 8px;">
                <strong>Sentiment:</strong> ${results.textAnalysis.sentiment}
            </div>
            <button onclick="this.parentElement.parentElement.parentElement.remove()" 
                    style="background: #007bff; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-size: 12px;">
                Close
            </button>
        </div>
    `;
    
    // Remove existing overlay
    const existing = document.getElementById('extension-analysis-overlay');
    if (existing) existing.remove();
    
    // Add new overlay
    document.body.appendChild(overlay);
    
    // Auto-remove after 10 seconds
    setTimeout(() => {
        if (overlay.parentNode) {
            overlay.parentNode.removeChild(overlay);
        }
    }, 10000);
}

/**
 * Show notification to user
 */
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        left: 50%;
        transform: translateX(-50%);
        background: ${type === 'error' ? '#dc3545' : '#007bff'};
        color: white;
        padding: 12px 24px;
        border-radius: 6px;
        z-index: 10001;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 14px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    `;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // Remove after 3 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    }, 3000);
}

// Initialize content script when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeContentScript);
} else {
    initializeContentScript();
}
