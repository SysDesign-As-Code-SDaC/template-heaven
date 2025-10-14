#!/usr/bin/env python3
"""
Test script for documentation generation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

try:
    from templateheaven.core.stack_documentation import StackDocumentationGenerator

    print("Testing documentation generation...")

    generator = StackDocumentationGenerator()
    result = generator.generate_stack_documentation('frontend')

    print('Result:', result)

    if result['success']:
        print('✅ Documentation generation successful!')
        print('Generated files:')
        for file_path in result['files_generated']:
            print(f'  - {file_path}')
    else:
        print('❌ Documentation generation failed!')
        print('Errors:')
        for error in result['errors']:
            print(f'  - {error}')

except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
