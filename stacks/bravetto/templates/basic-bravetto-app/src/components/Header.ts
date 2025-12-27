/**
 * Header Component
 * 
 * Reusable header component with navigation and branding.
 */

import { Component } from '@bravetto/core';

export class Header extends Component {
  /**
   * Component name for debugging
   */
  public readonly name = 'Header';

  /**
   * Initialize the header component
   */
  constructor() {
    super();
  }

  /**
   * Render the header content
   * @returns HTML string for the header
   */
  public render(): string {
    return `
      <header class="app-header">
        <div class="header-container">
          <div class="header-brand">
            <a href="/" class="brand-link">
              <span class="brand-logo">🚀</span>
              <span class="brand-text">Bravetto</span>
            </a>
          </div>
          
          <nav class="header-nav">
            <ul class="nav-list">
              <li class="nav-item">
                <a href="/" class="nav-link">Home</a>
              </li>
              <li class="nav-item">
                <a href="/about" class="nav-link">About</a>
              </li>
              <li class="nav-item">
                <a href="https://docs.bravetto.dev" class="nav-link" target="_blank">
                  Docs
                </a>
              </li>
            </ul>
          </nav>
          
          <div class="header-actions">
            <a href="https://github.com/bravetto/bravetto" class="btn btn-ghost" target="_blank">
              GitHub
            </a>
          </div>
        </div>
      </header>
    `;
  }

  /**
   * Component lifecycle method - called after render
   */
  public mounted(): void {
    console.log('Header component mounted');
  }

  /**
   * Component lifecycle method - called before unmount
   */
  public unmounted(): void {
    console.log('Header component unmounted');
  }
}
