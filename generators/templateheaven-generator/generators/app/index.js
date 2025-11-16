const Generator = require('yeoman-generator');
const path = require('path');
const fs = require('fs-extra');
const yaml = require('js-yaml');

module.exports = class TemplateHeavenGenerator extends Generator {
  async prompting() {
    // Load stacks.yaml from repo
    const repoRoot = path.resolve(__dirname, '../../../../');
    const stacksYamlPath = path.resolve(repoRoot, 'templateheaven', 'data', 'stacks.yaml');

    if (!fs.existsSync(stacksYamlPath)) {
      this.log('Could not find stacks.yaml at ' + stacksYamlPath);
      this.log('Please run this generator from within the template-heaven repo.');
      process.exit(1);
    }

    const fileContents = fs.readFileSync(stacksYamlPath, 'utf8');
    const stacksData = yaml.load(fileContents);

    // Build choices
    const stacks = stacksData && stacksData.stacks ? Object.keys(stacksData.stacks) : [];

    if (!stacks.length) {
      this.log('No stacks found in stacks.yaml');
      process.exit(1);
    }

    const stackChoices = stacks.map((k) => ({ name: `${stacksData.stacks[k].name} (${k})`, value: k }));

    const answers = await this.prompt([
      {
        type: 'list',
        name: 'stack',
        message: 'Which stack category do you want to browse?',
        choices: [...stackChoices, { name: 'Search all templates', value: 'search' }],
      }
    ]);

    this.selectedStack = answers.stack;
    // If search selected, prompt for query
    if (this.selectedStack === 'search') {
      const searchAns = await this.prompt([
        { type: 'input', name: 'query', message: 'Search query (name, tag, or description):' }
      ]);
      this.query = searchAns.query || '';
    }

    // Load templates for the selected stack or search across stacks
    let templates = [];
    if (this.selectedStack === 'search') {
      const q = this.query.toLowerCase();
      for (const [stackName, stackData] of Object.entries(stacksData.stacks)) {
        const t = stackData.templates || [];
        for (const template of t) {
          const text = `${template.name} ${template.description} ${(template.tags || []).join(' ')}`.toLowerCase();
          if (text.indexOf(q) !== -1) templates.push({ stack: stackName, template });
        }
      }
    } else {
      const t = stacksData.stacks[this.selectedStack].templates || [];
      templates = t.map((template) => ({ stack: this.selectedStack, template }));
    }

    if (templates.length === 0) {
      this.log('No templates found with your filters.');
      process.exit(0);
    }

    const templateChoices = templates.map((t) => ({
      name: `${t.template.name} - ${t.template.description}`,
      value: `${t.stack}:::${t.template.name}`
    }));

    const tAnswers = await this.prompt([
      {
        type: 'list',
        name: 'templateSelect',
        message: 'Select a template to scaffold',
        choices: templateChoices
      }
    ]);

    const [stackChoice, templateChoice] = tAnswers.templateSelect.split(':::');
    this.selectedTemplate = templateChoice;
    this.selectedTemplateStack = stackChoice;
    // Ask for a project name
    const pname = await this.prompt([
      { type: 'input', name: 'projectName', message: 'Project name to scaffold into:', default: 'my-app' }
    ]);
    this.projectName = pname.projectName;
    // Ask whether to run the Python CLI if available
    const usePythonCli = await this.prompt([
      { type: 'confirm', name: 'usePythonCli', message: 'Use Python `templateheaven` CLI to perform Jinja2 rendering and final scaffolding when available?', default: true }
    ]);
    this.usePythonCli = usePythonCli.usePythonCli;
  }

  async writing() {
    // Source template directory
    const repoRoot = path.resolve(__dirname, '../../../../');
    const templateDir = path.resolve(repoRoot, 'templates', this.selectedTemplate);

    if (!fs.existsSync(templateDir)) {
      this.log(`Template directory not found: ${templateDir}`);
      this.log('If the template is remote, consider adding it locally under `templates/<name>`');
      return;
    }

  const destination = this.destinationRoot();
    this.log(`Scaffolding ${this.selectedTemplate} into: ${destination}`);

    // If user selected to use python CLI, prefer that approach so Jinja templating is processed
    if (this.usePythonCli) {
      // Try to execute `templateheaven` (installed in PATH) as a child process
      try {
        const { execSync } = require('child_process');
        // Use --no-wizard and pass template and project name
        execSync(`templateheaven init --template ${this.selectedTemplate} --name ${this.projectName} --no-wizard --directory ${destination}`, {
          stdio: 'inherit'
        });
        this.log('Python CLI scaffolding executed');
        return;
      } catch (e) {
        // Fall back to copy if CLI not available or fails
        this.log('Warning: Python CLI not available or failed. Falling back to raw copy.');
      }
    }

    // Copy directory recursively
    await fs.copy(templateDir, destination, {
      filter: (src) => {
        // Exclude .git and node_modules
        const name = path.basename(src);
        if (name === '.git' || name === 'node_modules') return false;
        return true;
      }
    });

  this.log('Scaffolded template files.');
  }

  end() {
    this.log('\nDone!');
    this.log('Next steps:');
    this.log('  1. Inspect and update the generated files.');
    this.log('  2. Run local install commands for the project (npm/pip/go/etc).');
  }
};
