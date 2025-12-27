# Generic Templates Catalog

A comprehensive collection of generic templates for building modern applications across different platforms and use cases.

## 📋 Available Templates

### 1. VS Code Extension Template
**TypeScript-based VS Code extension with modern tooling**

| Property | Value |
|----------|-------|
| **Template Name** | `vscode-extension-template` |
| **Technology Stack** | TypeScript, VS Code API, Webpack |
| **Complexity** | Intermediate |
| **Use Case** | Development tools and IDE extensions |
| **Target Users** | VS Code extension developers |

#### Features
- ✅ **TypeScript Support**: Full TypeScript configuration
- ✅ **Modern Tooling**: Webpack, ESLint, and testing setup
- ✅ **Command Registration**: Easy command and menu integration
- ✅ **Status Bar Integration**: Custom status bar items
- ✅ **Document Analysis**: Real-time document processing
- ✅ **Configuration**: User-configurable settings

#### Quick Start
```bash
cp -r stacks/bravetto/templates/generic-templates/vscode-extension-template my-extension
cd my-extension
npm install
npm run compile
# Load in VS Code for testing
```

---

### 2. Browser Extension Template
**Chrome extension with modern web technologies**

| Property | Value |
|----------|-------|
| **Template Name** | `browser-extension-template` |
| **Technology Stack** | JavaScript, Chrome Extension API, Webpack |
| **Complexity** | Intermediate |
| **Use Case** | Web tools and browser integrations |
| **Target Users** | Browser extension developers |

#### Features
- ✅ **Real-time Detection**: Scans web pages as they load
- ✅ **Universal Support**: Works on all major websites
- ✅ **Privacy First**: All analysis happens locally
- ✅ **Smart Controls**: Enable/disable on specific websites
- ✅ **User Dashboard**: Track usage and manage settings
- ✅ **Content Analysis**: Advanced page content processing

#### Quick Start
```bash
cp -r stacks/bravetto/templates/generic-templates/browser-extension-template my-extension
cd my-extension
npm install
npm run build
# Load in Chrome for testing
```

---

### 3. Frontend Template
**Next.js-based web application with modern UI**

| Property | Value |
|----------|-------|
| **Template Name** | `frontend-template` |
| **Technology Stack** | Next.js 15, React 19, TypeScript, Tailwind CSS |
| **Complexity** | Advanced |
| **Use Case** | Web applications and landing pages |
| **Target Users** | Web developers, Full-stack developers |

#### Features
- ✅ **Next.js 15**: Latest Next.js with App Router
- ✅ **React 19**: Latest React features
- ✅ **TypeScript**: Full type safety
- ✅ **Tailwind CSS**: Utility-first styling
- ✅ **Responsive Design**: Mobile-first approach
- ✅ **Modern UI**: Clean and professional design

#### Quick Start
```bash
cp -r stacks/bravetto/templates/generic-templates/frontend-template my-app
cd my-app
npm install
npm run dev
```

## 🎯 Template Selection Guide

### Choose Based on Your Use Case

#### VS Code Extension Development
- **Use**: `vscode-extension-template`
- **When**: Creating development tools for VS Code
- **Requirements**: VS Code extension development knowledge
- **Features**: Real-time analysis, command integration

#### Browser Extension Development
- **Use**: `browser-extension-template`
- **When**: Building browser-based tools and integrations
- **Requirements**: Chrome extension development knowledge
- **Features**: Cross-platform detection, privacy protection

#### Web Application Development
- **Use**: `frontend-template`
- **When**: Building web applications and landing pages
- **Requirements**: React/Next.js development knowledge
- **Features**: Modern UI, responsive design, performance optimization

## 🚀 Getting Started

### Prerequisites
- Node.js 18+
- npm or yarn
- Git

### Installation Process
1. **Choose Template**: Select the appropriate template for your use case
2. **Copy Template**: Use the copy command to create your project
3. **Install Dependencies**: Run `npm install` in your project directory
4. **Configure Environment**: Set up environment variables and configuration
5. **Start Development**: Run the development server or build process
6. **Test & Deploy**: Test your application and deploy as needed

### Common Setup Steps
```bash
# 1. Copy template
cp -r stacks/bravetto/templates/generic-templates/[template-name] my-project

# 2. Navigate to project
cd my-project

# 3. Install dependencies
npm install

# 4. Configure environment (if needed)
cp .env.example .env.local
# Edit .env.local with your configuration

# 5. Start development
npm run dev
```

## 🔧 Customization

### Template Customization
All templates are designed to be easily customizable:

#### Styling & UI
- **CSS/Tailwind**: Modify styles and themes
- **Components**: Customize React components
- **Layouts**: Adjust page layouts and structure
- **Branding**: Update logos, colors, and branding

#### Functionality
- **Core Logic**: Modify main functionality
- **User Interface**: Customize user experience
- **Integration**: Add new integrations and APIs
- **Configuration**: Adjust settings and preferences

#### Platform-Specific
- **VS Code Extension**: Modify commands, settings, and UI
- **Browser Extension**: Adjust content scripts and popup interface
- **Frontend**: Customize pages, components, and styling

## 📊 Template Comparison

| Feature | VS Code Ext | Browser Ext | Frontend |
|---------|-------------|-------------|----------|
| **Complexity** | Intermediate | Intermediate | Advanced |
| **Setup Time** | 15-30 min | 15-30 min | 30-60 min |
| **Dependencies** | Medium | Low | High |
| **Customization** | Medium | Medium | High |
| **Deployment** | VS Code Marketplace | Chrome Web Store | Vercel/Netlify |
| **Maintenance** | Medium | Medium | High |

## 🛠️ Development Workflow

### 1. Template Selection
- Identify your use case and requirements
- Choose the appropriate template
- Review template documentation
- Plan your customization approach

### 2. Setup & Configuration
- Copy the template to your workspace
- Install dependencies and configure environment
- Set up development tools and IDE
- Configure version control and automation

### 3. Customization
- Modify styling and branding
- Customize functionality and features
- Add new integrations and APIs
- Configure settings and preferences

### 4. Testing & Deployment
- Test functionality and user experience
- Deploy to appropriate platforms
- Monitor performance and usage
- Gather feedback and iterate

## 📚 Documentation

### Template-Specific Documentation
- [VS Code Extension Template](./vscode-extension-template/README.md)
- [Browser Extension Template](./browser-extension-template/README.md)
- [Frontend Template](./frontend-template/README.md)

### General Resources
- [Next.js Documentation](https://nextjs.org/docs)
- [VS Code Extension API](https://code.visualstudio.com/api)
- [Chrome Extension API](https://developer.chrome.com/docs/extensions/)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)

## 🤝 Contributing

### How to Contribute
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Contribution Guidelines
- Follow coding standards and best practices
- Include comprehensive documentation
- Test all functionality thoroughly
- Provide clear commit messages
- Update documentation as needed

## 📄 License

MIT License - see [LICENSE](./LICENSE) for details.

## 🆘 Support

### Getting Help
- 📧 **Email Support**: support@bravetto.com
- 💬 **Discord Community**: [Bravetto Discord](https://discord.gg/bravetto)
- 🐛 **Issue Tracker**: [GitHub Issues](https://github.com/braveto/templates/issues)
- 📚 **Documentation**: [Bravetto Docs](https://docs.bravetto.dev)

### Common Issues
- **Setup Problems**: Check Node.js version and dependencies
- **Build Errors**: Verify configuration and environment variables
- **Runtime Issues**: Check browser compatibility and permissions
- **Performance**: Optimize code and UI components

---

**Generic Templates - Building the future of applications.**
