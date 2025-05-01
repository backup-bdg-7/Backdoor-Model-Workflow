#!/usr/bin/env python3
"""
Script to add license headers to all Python files in the codebase.
"""

import os
import re
import sys

# License header to add
LICENSE_HEADER = '''"""
/**
 * Copyright (c) [2025] Backdoor Software Inc.
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of Backdoor Software Inc.
 * You may not disclose, reproduce, or distribute this software without the express written
 * permission of Backdoor Software Inc.
 *
 * Created by: Backdoor Software Inc.
 * Purpose: Activity Maintenance
 */
"""

'''

# For Jupyter notebooks, we'll use a different format
NOTEBOOK_LICENSE_HEADER = '''# /**
# * Copyright (c) [2025] Backdoor Software Inc.
# *
# * All rights reserved.
# *
# * This software is the confidential and proprietary information of Backdoor Software Inc.
# * You may not disclose, reproduce, or distribute this software without the express written
# * permission of Backdoor Software Inc.
# *
# * Created by: Backdoor Software Inc.
# * Purpose: Activity Maintenance
# */
'''

def add_license_to_file(file_path):
    """Add license header to a Python file if it doesn't already have it."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if the exact license header format exists
    if "/**" in content and "Copyright (c)" in content and "Backdoor Software Inc." in content and "*/" in content:
        print(f"Correct license header already exists in {file_path}")
        return False
    
    # Check if an old license header exists and replace it
    if "Copyright (c)" in content and "Backdoor Software Inc." in content:
        # Find the old license header and replace it
        lines = content.split('\n')
        start_idx = -1
        end_idx = -1
        
        for i, line in enumerate(lines):
            if "Copyright (c)" in line and "Backdoor Software Inc." in line:
                # Look backward to find the start of the header
                for j in range(i, -1, -1):
                    if lines[j].strip() == '"""':
                        start_idx = j
                        break
                
                # Look forward to find the end of the header
                for j in range(i, len(lines)):
                    if lines[j].strip() == '"""':
                        end_idx = j
                        break
                
                break
        
        if start_idx != -1 and end_idx != -1:
            # Remove the old header
            del lines[start_idx:end_idx+1]
            
            # Insert the new header at the beginning
            new_lines = LICENSE_HEADER.strip().split('\n')
            for i in range(len(new_lines)-1, -1, -1):
                lines.insert(0, new_lines[i])
            
            new_content = '\n'.join(lines)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"Replaced license header in {file_path}")
            return True
    
    # For Python files
    if file_path.endswith('.py'):
        # Check if file starts with a shebang or docstring
        if content.startswith('#!'):
            # Find the end of the shebang line
            shebang_end = content.find('\n') + 1
            new_content = content[:shebang_end] + '\n' + LICENSE_HEADER + content[shebang_end:]
        elif content.startswith('"""') or content.startswith("'''"):
            # Find the end of the docstring
            if content.startswith('"""'):
                docstring_end = content.find('"""', 3) + 3
            else:
                docstring_end = content.find("'''", 3) + 3
            
            # Add a newline after the docstring if there isn't one
            if docstring_end < len(content) and content[docstring_end] != '\n':
                docstring_end += 1
            
            new_content = content[:docstring_end] + '\n' + LICENSE_HEADER + content[docstring_end:]
        else:
            new_content = LICENSE_HEADER + content
    
        # Write the modified content back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"Added license header to {file_path}")
        return True
    
    return False

def add_license_to_notebook(file_path):
    """Add license header to a Jupyter notebook."""
    import json
    
    with open(file_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Check if license header already exists
    first_cell = notebook['cells'][0] if notebook['cells'] else None
    if first_cell and first_cell['cell_type'] == 'markdown':
        if "Copyright (c)" in ''.join(first_cell['source']) and "Backdoor Software Inc." in ''.join(first_cell['source']):
            print(f"License header already exists in {file_path}")
            return False
    
    # Create a new markdown cell with the license header
    license_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [NOTEBOOK_LICENSE_HEADER]
    }
    
    # Insert the license cell at the beginning
    notebook['cells'].insert(0, license_cell)
    
    # Write the modified notebook back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Added license header to {file_path}")
    return True

def process_directory(directory):
    """Process all Python files in the given directory and its subdirectories."""
    modified_files = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            
            if file.endswith('.py'):
                if add_license_to_file(file_path):
                    modified_files += 1
            # Skip Jupyter notebooks as per user request
    
    return modified_files

if __name__ == "__main__":
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = os.getcwd()
    
    print(f"Adding license headers to Python files in {directory}...")
    modified_files = process_directory(directory)
    print(f"Added license headers to {modified_files} files.")