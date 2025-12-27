# How the Template Heaven Wizard Works

## 🎯 Overview

The wizard is an **interactive CLI tool** that guides users through creating projects from templates. It uses:
- **Rich** - Beautiful terminal formatting
- **Questionary** - Interactive prompts
- **Jinja2** - Template variable substitution
- **Customizer** - File copying and processing

---

## 🔄 Complete Flow

```
User: templateheaven init
    ↓
1. Welcome Screen (Rich Panel)
    ↓
2. Stack Selection (Questionary dropdown)
    ↓
3. Template Selection (Rich Table + Questionary)
    ↓
4. Project Configuration (Multiple prompts)
    ↓
5. Confirmation Preview (Rich Table)
    ↓
6. Project Creation (Customizer.scaffold)
    ↓
7. Success + Next Steps
```

---

## 📋 Step-by-Step Breakdown

### Step 1: Welcome (`_display_welcome`)
- Shows Rich Panel with welcome message
- Introduces the wizard

### Step 2: Stack Selection (`_select_stack`)
- Gets stacks from `template_manager.get_stacks()`
- Shows each stack with template count
- Uses Questionary select dropdown
- Includes "Search all templates" option

**Special: Search Flow**
- If user selects search:
  - Prompts for query
  - Asks: Local or GitHub?
  - Searches and displays results
  - Can scaffold from GitHub repos directly

### Step 3: Template Selection (`_select_template`)
- Gets templates for selected stack
- Displays in Rich Table (Name, Description, Tags, Version)
- Uses Questionary for selection
- Includes "Back" option

### Step 4: Project Configuration (`_configure_project`)
Prompts for:
- **Project Name** (`_get_project_name`)
  - Validates name
  - Suggests sanitized version if invalid
- **Author** (`_get_author`)
  - Uses config default or prompts
- **License** (`_get_license`)
  - Dropdown: MIT, Apache-2.0, GPL-3.0, etc.
- **Package Manager** (`_get_package_manager`)
  - Auto-detects based on template tags
  - Python → pip/poetry
  - Node → npm/yarn/pnpm
  - Rust → cargo
  - Go → go
- **Description** (`_get_description`)
  - Defaults to template-based description

### Step 5: Confirmation (`_confirm_creation`)
- Displays project preview in Rich Table
- Shows all configured values
- Asks: "Create this project?"

### Step 6: Project Creation (`_create_project`)
Calls `customizer.customize()` which:

1. **Validates inputs**
   - Checks project directory doesn't exist
   - Validates template and config

2. **Creates project directory**
   - `output_dir / project_name`

3. **Copies template files** (`_copy_template_files`)
   - Finds template in `templates/` directory
   - Recursively copies all files
   - Processes Jinja2 templates

4. **Processes files with Jinja2** (`_copy_directory_recursive`)
   - Scans for `.j2` files or Jinja2 syntax
   - Replaces variables: `{{ project_name }}`, `{{ author }}`, etc.
   - Supports filters: `snake_case`, `kebab_case`, `pascal_case`

5. **Updates package files** (`_update_package_files`)
   - Updates `package.json` (Node projects)
   - Updates `pyproject.toml` (Python projects)
   - Updates `Cargo.toml` (Rust projects)

6. **Creates standard files**
   - README.md (if missing)
   - LICENSE (if missing)
   - CONTRIBUTING.md (if missing)
   - .gitignore (if missing)

### Step 7: Success (`_display_next_steps`)
- Shows success message
- Displays next steps:
  - `cd project-name`
  - `npm install` (or appropriate)
  - `npm run dev` (or appropriate)

---

## 🛠️ Key Components

### Wizard Class
- **`self.console`** - Rich Console for output
- **`self.customizer`** - Customizer instance
- **`self.template_manager`** - Template discovery
- **`self.config`** - User configuration

### Customizer Class
- **`self.file_ops`** - File operations utility
- **`self.jinja_env`** - Jinja2 environment
- **Custom filters**: snake_case, kebab_case, pascal_case, camel_case

### ProjectConfig
Contains:
- `name` - Project name
- `directory` - Output directory
- `template` - Selected template
- `author` - Author name
- `license` - License type
- `package_manager` - Package manager
- `description` - Project description

---

## 🎨 Jinja2 Templating

Templates can use variables:
```jinja2
# In template files
{{ project_name }}
{{ author }}
{{ license }}
{{ package_manager }}

# With filters
{{ project_name|snake_case }}
{{ project_name|kebab_case }}
{{ project_name|pascal_case }}
```

---

## 🔍 GitHub Integration

If user searches GitHub:
1. Clones repo to temp directory
2. Uses `customize_from_repo_dir()`
3. Processes files with Jinja2
4. Creates project structure

---

## ✅ Error Handling

- **KeyboardInterrupt** - Graceful cancellation
- **Validation errors** - Shows error, suggests fix
- **File errors** - Cleans up partial project
- **Template errors** - Shows error message

---

## 📝 Example Flow

```python
# User runs: templateheaven init

wizard = Wizard(template_manager, config)
wizard.run()

# Internally:
# 1. _display_welcome()
# 2. stack = _select_stack()  # User selects "frontend"
# 3. template = _select_template(stack)  # User selects "react-vite"
# 4. config = _configure_project(template)  # User enters: "my-app", "John Doe", "MIT", "npm"
# 5. if _confirm_creation(config):  # User confirms
# 6.     _create_project(config)  # Calls customizer.customize()
# 7.     # Customizer copies files, processes Jinja2, updates package.json
# 8.     _display_next_steps()  # Shows: cd my-app, npm install, npm run dev
```

---

## 🎯 Summary

The wizard provides a **guided, interactive experience** for:
1. **Discovering** templates (local or GitHub)
2. **Selecting** appropriate template
3. **Configuring** project settings
4. **Scaffolding** project with customization
5. **Providing** next steps

It's like Yeoman but built with Python, Rich, and Jinja2!
