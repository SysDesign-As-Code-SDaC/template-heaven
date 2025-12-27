/**
 * VS Code Extension Template
 * 
 * This is the main entry point for your VS Code extension.
 * It demonstrates basic extension functionality and structure.
 */

import * as vscode from 'vscode';

/**
 * Extension activation function
 * Called when the extension is activated
 */
export function activate(context: vscode.ExtensionContext) {
    console.log('Extension "my-vscode-extension" is now active!');

    // Register commands
    const helloWorldCommand = vscode.commands.registerCommand('my-extension.helloWorld', () => {
        vscode.window.showInformationMessage('Hello World from My Extension!');
    });

    const analyzeCommand = vscode.commands.registerCommand('my-extension.analyze', () => {
        analyzeCurrentFile();
    });

    // Add commands to context
    context.subscriptions.push(helloWorldCommand);
    context.subscriptions.push(analyzeCommand);

    // Register status bar item
    const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
    statusBarItem.text = "$(check) My Extension";
    statusBarItem.command = 'my-extension.helloWorld';
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    // Register text document change listener
    const documentChangeListener = vscode.workspace.onDidChangeTextDocument((event) => {
        if (event.document.languageId === 'typescript' || event.document.languageId === 'javascript') {
            // Perform analysis on document changes
            analyzeDocument(event.document);
        }
    });
    context.subscriptions.push(documentChangeListener);
}

/**
 * Analyze the current active file
 */
async function analyzeCurrentFile(): Promise<void> {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active editor found');
        return;
    }

    const document = editor.document;
    const analysis = await analyzeDocument(document);
    
    if (analysis) {
        vscode.window.showInformationMessage(`Analysis complete: ${analysis.summary}`);
    }
}

/**
 * Analyze a document for various patterns
 * @param document The document to analyze
 * @returns Analysis results
 */
async function analyzeDocument(document: vscode.TextDocument): Promise<any> {
    const text = document.getText();
    const lines = text.split('\n');
    
    // Basic analysis logic
    const analysis = {
        totalLines: lines.length,
        totalCharacters: text.length,
        summary: `Analyzed ${lines.length} lines`,
        findings: [] as string[]
    };

    // Example analysis patterns
    lines.forEach((line, index) => {
        // Check for TODO comments
        if (line.includes('TODO') || line.includes('FIXME')) {
            analysis.findings.push(`Line ${index + 1}: ${line.trim()}`);
        }
        
        // Check for console.log statements
        if (line.includes('console.log')) {
            analysis.findings.push(`Line ${index + 1}: Console.log found`);
        }
    });

    // Show findings in output channel
    if (analysis.findings.length > 0) {
        const outputChannel = vscode.window.createOutputChannel('My Extension');
        outputChannel.clear();
        outputChannel.appendLine('Document Analysis Results:');
        analysis.findings.forEach(finding => {
            outputChannel.appendLine(finding);
        });
        outputChannel.show();
    }

    return analysis;
}

/**
 * Extension deactivation function
 * Called when the extension is deactivated
 */
export function deactivate() {
    console.log('Extension "my-vscode-extension" is now deactivated');
}
