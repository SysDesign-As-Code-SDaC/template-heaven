/**
 * Main entry point for the Basic Bravetto App
 * 
 * This file initializes the Bravetto application with basic configuration
 * and starts the development server.
 */

import { BravettoApp } from '@bravetto/core';
import { createRouter } from '@bravetto/router';
import { HomePage } from './pages/HomePage';
import { AboutPage } from './pages/AboutPage';

/**
 * Application configuration
 */
const appConfig = {
  name: 'Basic Bravetto App',
  version: '1.0.0',
  debug: process.env.NODE_ENV === 'development',
  port: process.env.PORT || 3000,
};

/**
 * Create router with basic routes
 */
const router = createRouter({
  routes: [
    {
      path: '/',
      component: HomePage,
      name: 'home'
    },
    {
      path: '/about',
      component: AboutPage,
      name: 'about'
    }
  ]
});

/**
 * Initialize and start the Bravetto application
 */
async function main(): Promise<void> {
  try {
    const app = new BravettoApp(appConfig);
    
    // Configure router
    app.use(router);
    
    // Start the application
    await app.start();
    
    console.log(`🚀 Bravetto app running at http://localhost:${appConfig.port}`);
  } catch (error) {
    console.error('Failed to start Bravetto app:', error);
    process.exit(1);
  }
}

// Start the application
if (require.main === module) {
  main();
}

export { appConfig, router };
