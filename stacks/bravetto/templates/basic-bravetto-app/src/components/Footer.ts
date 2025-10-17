/**
 * Footer Component
 * 
 * Reusable footer component with links and information.
 */

import { Component } from '@bravetto/core';

export class Footer extends Component {
  /**
   * Component name for debugging
   */
  public readonly name = 'Footer';

  /**
   * Initialize the footer component
   */
  constructor() {
    super();
  }

  /**
   * Render the footer content
   * @returns HTML string for the footer
   */
  public render(): string {
    const currentYear = new Date().getFullYear();
    
    return `
      <footer class="app-footer">
        <div class="footer-container">
          <div class="footer-content">
            <div class="footer-section">
              <h3 class="footer-title">Bravetto</h3>
              <p class="footer-description">
                A modern, high-performance development framework
              </p>
            </div>
            
            <div class="footer-section">
              <h4 class="footer-subtitle">Resources</h4>
              <ul class="footer-links">
                <li><a href="https://docs.bravetto.dev" target="_blank">Documentation</a></li>
                <li><a href="https://api.bravetto.dev" target="_blank">API Reference</a></li>
                <li><a href="https://community.bravetto.dev" target="_blank">Community</a></li>
              </ul>
            </div>
            
            <div class="footer-section">
              <h4 class="footer-subtitle">Community</h4>
              <ul class="footer-links">
                <li><a href="https://github.com/bravetto/bravetto" target="_blank">GitHub</a></li>
                <li><a href="https://discord.gg/bravetto" target="_blank">Discord</a></li>
                <li><a href="https://twitter.com/bravetto" target="_blank">Twitter</a></li>
              </ul>
            </div>
            
            <div class="footer-section">
              <h4 class="footer-subtitle">Support</h4>
              <ul class="footer-links">
                <li><a href="mailto:support@bravetto.dev">Email Support</a></li>
                <li><a href="https://github.com/bravetto/bravetto/issues" target="_blank">Report Issue</a></li>
                <li><a href="https://bravetto.dev/status" target="_blank">Status</a></li>
              </ul>
            </div>
          </div>
          
          <div class="footer-bottom">
            <div class="footer-copyright">
              <p>&copy; ${currentYear} Bravetto Team. All rights reserved.</p>
            </div>
            <div class="footer-legal">
              <a href="https://bravetto.dev/privacy" target="_blank">Privacy</a>
              <a href="https://bravetto.dev/terms" target="_blank">Terms</a>
              <a href="https://bravetto.dev/license" target="_blank">License</a>
            </div>
          </div>
        </div>
      </footer>
    `;
  }

  /**
   * Component lifecycle method - called after render
   */
  public mounted(): void {
    console.log('Footer component mounted');
  }

  /**
   * Component lifecycle method - called before unmount
   */
  public unmounted(): void {
    console.log('Footer component unmounted');
  }
}
