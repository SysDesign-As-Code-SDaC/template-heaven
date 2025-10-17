/**
 * About Page Component
 * 
 * Information page about the Bravetto framework and this template.
 */

import { Component } from '@bravetto/core';
import { Header } from '../components/Header';
import { Footer } from '../components/Footer';

export class AboutPage extends Component {
  /**
   * Component name for debugging and routing
   */
  public readonly name = 'AboutPage';

  /**
   * Initialize the about page component
   */
  constructor() {
    super();
  }

  /**
   * Render the about page content
   * @returns HTML string for the about page
   */
  public render(): string {
    return `
      <div class="about-page">
        ${new Header().render()}
        
        <main class="main-content">
          <section class="about-hero">
            <h1>About Bravetto</h1>
            <p class="about-subtitle">
              A modern framework for building scalable applications
            </p>
          </section>
          
          <section class="about-content">
            <div class="content-grid">
              <div class="content-section">
                <h2>What is Bravetto?</h2>
                <p>
                  Bravetto is a high-performance development framework designed 
                  for building scalable applications with a focus on developer 
                  experience and performance optimization.
                </p>
              </div>
              
              <div class="content-section">
                <h2>Why Choose Bravetto?</h2>
                <ul class="feature-list">
                  <li>🚀 <strong>High Performance:</strong> Optimized for speed and efficiency</li>
                  <li>🛠️ <strong>Developer Experience:</strong> Intuitive APIs and comprehensive tooling</li>
                  <li>📈 <strong>Scalability:</strong> Built for applications that need to grow</li>
                  <li>🏗️ <strong>Modern Architecture:</strong> Clean, maintainable code patterns</li>
                  <li>🌐 <strong>Cross-Platform:</strong> Works across different environments</li>
                </ul>
              </div>
              
              <div class="content-section">
                <h2>Getting Started</h2>
                <p>
                  This template provides a minimal setup to get you started 
                  with Bravetto development. It includes:
                </p>
                <ul class="template-features">
                  <li>Basic routing setup</li>
                  <li>Component structure</li>
                  <li>Development server</li>
                  <li>TypeScript support</li>
                  <li>Hot reloading</li>
                </ul>
              </div>
            </div>
          </section>
          
          <section class="cta-section">
            <h2>Ready to Start Building?</h2>
            <p>Check out our documentation and start building amazing applications.</p>
            <div class="cta-actions">
              <a href="/" class="btn btn-primary">Back to Home</a>
              <a href="https://docs.bravetto.dev" class="btn btn-secondary" target="_blank">
                View Documentation
              </a>
            </div>
          </section>
        </main>
        
        ${new Footer().render()}
      </div>
    `;
  }

  /**
   * Component lifecycle method - called after render
   */
  public mounted(): void {
    console.log('AboutPage component mounted');
  }

  /**
   * Component lifecycle method - called before unmount
   */
  public unmounted(): void {
    console.log('AboutPage component unmounted');
  }
}
