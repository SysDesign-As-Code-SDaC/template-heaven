# VS Code Extension Template

A comprehensive template for building VS Code extensions with TypeScript, modern tooling, and best practices.

## 🚀 Quick Start

```bash
# Copy this template
cp -r stacks/bravetto/templates/generic-templates/vscode-extension-template my-extension
cd my-extension

# Install dependencies
npm install

# Compile the extension
npm run compile

# Load in VS Code for testing
# 1. Open VS Code
# 2. Press F5 to run the extension
# 3. Or use "Extension: Install from VSIX" command
```

## 📋 Features

### Core Features
- ✅ **TypeScript Support**: Full TypeScript configuration
- ✅ **Modern Tooling**: Webpack, ESLint, and testing setup
- ✅ **Command Registration**: Easy command and menu integration
- ✅ **Status Bar Integration**: Custom status bar items
- ✅ **Document Analysis**: Real-time document processing
- ✅ **Configuration**: User-configurable settings

### Extension Capabilities
- 🔧 **Commands**: Custom commands and keyboard shortcuts
- ⚙️ **Settings**: User-configurable preferences
- 📊 **Status Bar**: Real-time status information
- 🎨 **Themes**: Consistent with VS Code themes
- 🔄 **Auto-updates**: Automatic extension updates

## 🏗️ Project Structure

```
vscode-extension-template/
├── src/                          # Source code
│   └── extension.ts             # Main extension file
├── package.json                 # Extension manifest
├── tsconfig.json               # TypeScript configuration
├── webpack.config.js           # Webpack configuration
└── README.md                   # Documentation
```

## 🛠️ Development

### Prerequisites
- Node.js 18+
- VS Code 1.80+
- TypeScript 5+

### Development Setup
```bash
# Install dependencies
npm install

# Compile TypeScript
npm run compile

# Watch for changes
npm run watch

# Run tests
npm test
```

### Available Scripts
```bash
npm run compile         # Compile TypeScript to JavaScript
npm run watch          # Watch for changes and recompile
npm run package        # Create VSIX package
npm run lint           # Run ESLint
npm test               # Run tests
```

## 🔧 Configuration

### Extension Settings
```json
{
  "myExtension.enable": true,
  "myExtension.sensitivity": "medium"
}
```

### Commands
- `My Extension: Hello World` - Display hello world message
- `My Extension: Analyze Current File` - Analyze the current file

### Status Bar
- Shows extension status and provides quick access to commands

## 🎯 Usage

### Basic Usage
1. **Install Extension**: Load the extension in VS Code
2. **Enable Features**: Configure extension settings
3. **Use Commands**: Access commands via Command Palette
4. **Monitor Status**: Check status bar for real-time information

### Commands
- **Command Palette**: `Ctrl+Shift+P` → "My Extension: Hello World"
- **Context Menu**: Right-click in editor → "Analyze Current File"
- **Status Bar**: Click status bar item for quick access

## 🎨 Customization

### Adding New Commands
```typescript
// Register a new command
const newCommand = vscode.commands.registerCommand('my-extension.newCommand', () => {
    vscode.window.showInformationMessage('New command executed!');
});
context.subscriptions.push(newCommand);
```

### Adding Configuration
```json
// In package.json
"contributes": {
  "configuration": {
    "properties": {
      "myExtension.newSetting": {
        "type": "string",
        "default": "default-value",
        "description": "Description of the setting"
      }
    }
  }
}
```

### Custom Status Bar Items
```typescript
// Create custom status bar item
const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
statusBarItem.text = "$(icon) Custom Status";
statusBarItem.command = 'my-extension.command';
statusBarItem.show();
```

## 🧪 Testing

### Unit Tests
```bash
# Run unit tests
npm test

# Run tests in watch mode
npm run test:watch
```

### Integration Tests
```bash
# Test extension in VS Code
npm run test:integration
```

## 📦 Packaging

### Create VSIX Package
```bash
# Build and package extension
npm run package

# Install locally
code --install-extension my-extension.vsix
```

### Publishing
```bash
# Publish to VS Code Marketplace
vsce publish

# Publish with specific version
vsce publish 1.0.0
```

## 🔧 Advanced Configuration

### Webpack Configuration
```javascript
// webpack.config.js
module.exports = {
  entry: './src/extension.ts',
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'extension.js',
    libraryTarget: 'commonjs2'
  },
  // ... additional webpack configuration
}
```

### TypeScript Configuration
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "lib": ["ES2020"],
    "outDir": "./dist",
    "strict": true,
    "esModuleInterop": true
  }
}
```

## 📚 Documentation

- [VS Code Extension API](https://code.visualstudio.com/api)
- [TypeScript Documentation](https://www.typescriptlang.org/docs)
- [Webpack Documentation](https://webpack.js.org/docs)
- [Extension Guidelines](https://code.visualstudio.com/api/references/extension-guidelines)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](./LICENSE) for details.

---

**Built with ❤️ by the Bravetto Team**
