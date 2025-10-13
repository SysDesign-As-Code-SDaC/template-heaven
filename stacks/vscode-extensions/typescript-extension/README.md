# VSCode TypeScript Extension Template

A production-ready VSCode extension template built with TypeScript, featuring modern development practices and comprehensive tooling.

## 🚀 Features

- **TypeScript** for type-safe development
- **Webpack** for bundling
- **ESLint & Prettier** for code quality
- **Jest** for testing
- **Husky** for git hooks
- **Automated Publishing** with vsce
- **CI/CD** with GitHub Actions
- **Documentation** with JSDoc
- **Debugging** support
- **Hot Reload** for development

## 📋 Prerequisites

- Node.js 18+
- VSCode
- Git

## 🛠️ Quick Start

### 1. Create New Extension

```bash
# Install Yeoman and VSCode extension generator
npm install -g yo generator-code

# Generate new extension
yo code
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Development

```bash
# Start development
npm run watch

# Open VSCode in Extension Development Host
# Press F5 to launch extension development host
```

## 📁 Project Structure

```
├── src/
│   ├── extension.ts       # Main extension entry point
│   ├── commands/          # Command implementations
│   ├── providers/         # Language features providers
│   ├── utils/             # Utility functions
│   └── types/             # TypeScript type definitions
├── test/
│   ├── runTest.ts         # Test runner
│   └── suite/             # Test suites
├── media/                 # Extension icons and images
├── .vscode/               # VSCode configuration
├── webpack.config.js      # Webpack configuration
└── package.json           # Extension manifest
```

## 🔧 Available Scripts

```bash
# Development
npm run watch              # Watch for changes and compile
npm run compile            # Compile TypeScript
npm run lint               # Run ESLint
npm run lint:fix           # Fix ESLint issues
npm run format             # Format code with Prettier

# Testing
npm run test               # Run tests
npm run test:watch         # Run tests in watch mode
npm run test:coverage      # Run tests with coverage

# Building
npm run build              # Build extension
npm run package            # Package extension for publishing
npm run publish            # Publish to VSCode Marketplace

# Development
npm run dev                # Start development mode
npm run dev:watch          # Start development with watch mode
```

## 📦 Extension Manifest

```json
{
  "name": "my-extension",
  "displayName": "My Extension",
  "description": "A sample VSCode extension",
  "version": "0.0.1",
  "engines": {
    "vscode": "^1.74.0"
  },
  "categories": [
    "Other"
  ],
  "activationEvents": [
    "onCommand:myExtension.helloWorld"
  ],
  "main": "./out/extension.js",
  "contributes": {
    "commands": [
      {
        "command": "myExtension.helloWorld",
        "title": "Hello World"
      }
    ],
    "menus": {
      "commandPalette": [
        {
          "command": "myExtension.helloWorld"
        }
      ]
    }
  },
  "scripts": {
    "vscode:prepublish": "npm run compile",
    "compile": "tsc -p ./",
    "watch": "tsc -watch -p ./"
  }
}
```

## 🎯 Command Implementation

```typescript
// src/commands/helloWorld.ts
import * as vscode from 'vscode';

export function registerHelloWorldCommand(context: vscode.ExtensionContext) {
  const disposable = vscode.commands.registerCommand('myExtension.helloWorld', () => {
    vscode.window.showInformationMessage('Hello World from My Extension!');
  });

  context.subscriptions.push(disposable);
}
```

## 🔍 Language Features Provider

```typescript
// src/providers/completionProvider.ts
import * as vscode from 'vscode';

export class MyCompletionProvider implements vscode.CompletionItemProvider {
  provideCompletionItems(
    document: vscode.TextDocument,
    position: vscode.Position,
    token: vscode.CancellationToken,
    context: vscode.CompletionContext
  ): vscode.ProviderResult<vscode.CompletionItem[] | vscode.CompletionList> {
    
    const completionItems: vscode.CompletionItem[] = [];
    
    // Add custom completion items
    const item = new vscode.CompletionItem('myKeyword', vscode.CompletionItemKind.Keyword);
    item.detail = 'My custom keyword';
    item.documentation = new vscode.MarkdownString('This is a custom keyword for my extension');
    
    completionItems.push(item);
    
    return completionItems;
  }
}
```

## 🎨 Webview Provider

```typescript
// src/providers/webviewProvider.ts
import * as vscode from 'vscode';

export class MyWebviewProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = 'myExtension.webview';

  constructor(private readonly _extensionUri: vscode.Uri) {}

  public resolveWebviewView(
    webviewView: vscode.WebviewView,
    context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken,
  ) {
    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [this._extensionUri]
    };

    webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);
  }

  private _getHtmlForWebview(webview: vscode.Webview) {
    return `<!DOCTYPE html>
      <html lang="en">
      <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>My Extension Webview</title>
      </head>
      <body>
        <h1>Hello from Webview!</h1>
        <p>This is a custom webview for my extension.</p>
      </body>
      </html>`;
  }
}
```

## 🧪 Testing

```typescript
// test/suite/extension.test.ts
import * as assert from 'assert';
import * as vscode from 'vscode';

suite('Extension Test Suite', () => {
  vscode.window.showInformationMessage('Start all tests.');

  test('Sample test', () => {
    assert.strictEqual(-1, [1, 2, 3].indexOf(5));
    assert.strictEqual(-1, [1, 2, 3].indexOf(0));
  });

  test('Extension should be present', () => {
    assert.ok(vscode.extensions.getExtension('my-extension'));
  });

  test('Should activate', async () => {
    const extension = vscode.extensions.getExtension('my-extension');
    if (extension) {
      await extension.activate();
      assert.ok(extension.isActive);
    }
  });
});
```

## 🚀 Publishing

### Package Extension

```bash
# Install vsce
npm install -g vsce

# Package extension
vsce package

# This creates a .vsix file ready for installation
```

### Publish to Marketplace

```bash
# Login to Azure DevOps
vsce login <publisher-name>

# Publish extension
vsce publish
```

### Automated Publishing with GitHub Actions

```yaml
# .github/workflows/publish.yml
name: Publish Extension

on:
  push:
    tags:
      - 'v*'

jobs:
  publish:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '18'
        cache: 'npm'
    
    - name: Install dependencies
      run: npm ci
    
    - name: Build extension
      run: npm run compile
    
    - name: Package extension
      run: npx vsce package
    
    - name: Publish to Marketplace
      run: npx vsce publish
      env:
        VSCE_PAT: ${{ secrets.VSCE_PAT }}
```

## 🔧 Configuration

### Webpack Configuration

```javascript
// webpack.config.js
const path = require('path');

module.exports = {
  mode: 'production',
  entry: './src/extension.ts',
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'extension.js',
    libraryTarget: 'commonjs2'
  },
  externals: {
    vscode: 'commonjs vscode'
  },
  resolve: {
    extensions: ['.ts', '.js']
  },
  module: {
    rules: [
      {
        test: /\.ts$/,
        exclude: /node_modules/,
        use: [
          {
            loader: 'ts-loader'
          }
        ]
      }
    ]
  }
};
```

### TypeScript Configuration

```json
{
  "compilerOptions": {
    "module": "commonjs",
    "target": "ES2020",
    "outDir": "out",
    "lib": ["ES2020"],
    "sourceMap": true,
    "rootDir": "src",
    "strict": true
  },
  "exclude": ["node_modules", ".vscode-test"]
}
```

## 📚 Learning Resources

- [VSCode Extension API](https://code.visualstudio.com/api)
- [Extension Development Guide](https://code.visualstudio.com/api/get-started/your-first-extension)
- [VSCode Extension Samples](https://github.com/microsoft/vscode-extension-samples)
- [VSCE Documentation](https://code.visualstudio.com/api/working-with-extensions/publishing-extension)

## 🔗 Upstream Source

- **Repository**: [microsoft/vscode-extension-samples](https://github.com/microsoft/vscode-extension-samples)
- **Generator**: [microsoft/vscode-generator-code](https://github.com/microsoft/vscode-generator-code)
- **Documentation**: [code.visualstudio.com/api](https://code.visualstudio.com/api)
- **License**: MIT
