# Template Heaven Yeoman Generator

This is a simple Yeoman generator for Template Heaven. It provides a quick way to browse bundled templates (the same metadata used by Template Heaven) and scaffold template files into the current directory.

Usage
1. Install dependencies for the generator (from the generator directory):

```bash
cd generators/templateheaven-generator
npm install
```

2. Link the generator locally (optional):

```bash
npm link
```

3. Run the generator (from where you want to scaffold the project):

```bash
yo templateheaven
```

This generator reads `templateheaven/data/stacks.yaml` from the repository root. The generator also includes a convenience option to call the Python `templateheaven` CLI (if installed and available in PATH) and pass the selected template and project name to it so the Python customizer can do Jinja2 rendering and final scaffolding.

Why call the Python CLI?
- The Python CLI processes Jinja2 variables, templating, and customizations in templates accurately. If you rely on Jinja2 templates shipped with Template Heaven, enabling the Python CLI in the Yeoman generator ensures the final scaffold has correctly substituted values.

If the `templateheaven` CLI is not installed, the generator will fall back to copying the template files without Jinja2 substitution (useful for quick scaffolds).

Notes
- This is a starting point. It copies template files as-is and does not yet process Jinja2 templating. For advanced customization, consider extending this generator to use templating engines such as Handlebars or to call the Template Heaven Python API to perform the Jinja2 rendering.
- The Yeoman generator can be extended to fetch templates from GitHub or other remote sources.

Contributing
- Add unit tests or manual test instructions.
- Add support for remote templates or template previews.
