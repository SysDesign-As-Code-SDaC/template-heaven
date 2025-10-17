/**
 * Home Page Component
 * 
 * The main landing page for the Basic Bravetto App.
 * Demonstrates basic component structure and functionality.
 */

import { Component } from '@bravetto/core';
import { Header } from '../components/Header';
import { Footer } from '../components/Footer';

export class HomePage extends Component {
  /**
   * Component name for debugging and routing
   */
  public readonly name = 'HomePage';

  /**
   * Initialize the home page component
   */
  constructor() {
    super();
  }

  /**
   * Render the home page content
   * @returns HTML string for the home page
   */
  public render(): string {
    return `
      <div class="home-page">
        ${new Header().render()}
        
        <main class="main-content">
          <section class="hero">
            <h1>Welcome to Bravetto</h1>
            <p class="hero-subtitle">
              A modern, high-performance development framework
            </p>
            <div class="hero-actions">
              <a href="/about" class="btn btn-primary">Learn More</a>
              <a href="https://docs.bravetto.dev" class="btn btn-secondary" target="_blank">
                Documentation
              </a>
            </div>
          </section>
          
          <section class="features">
            <h2>Key Features</h2>
            <div class="feature-grid">
              <div class="feature-card">
                <h3>🚀 High Performance</h3>
                <p>Optimized for speed and efficiency</p>
              </div>
              <div class="feature-card">
                <h3>🛠️ Developer Experience</h3>
                <p>Intuitive APIs and comprehensive tooling</p>
              </div>
              <div class="feature-card">
                <h3>📈 Scalability</h3>
                <p>Built for applications that need to grow</p>
              </div>
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
    console.log('HomePage component mounted');
  }

  /**
   * Component lifecycle method - called before unmount
   */
  public unmounted(): void {
    console.log('HomePage component unmounted');
  }
}
