#!/usr/bin/env python3

"""
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


import os
import re
import sys
import json

# Python license header
PY_LICENSE_HEADER = '''"""
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
"""

'''

# Markdown/YAML/TXT license header
MD_LICENSE_HEADER = '''<!--
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
-->

'''

# Shell script license header
SH_LICENSE_HEADER = '''#!/bin/bash
#
# Copyright (c) [2025] Backdoor Software Inc.
#
# All rights reserved.
#
# This software is the confidential and proprietary information of Backdoor Software Inc.
# You may not disclose, reproduce, or distribute this software without the express written
# permission of Backdoor Software Inc.
#
# Created by: Backdoor Software Inc.
# Purpose: Activity Maintenance
#

'''

def has_license(content):
    """Check if the file already has a license header."""
    return "Copyright (c) [2025] Backdoor Software Inc." in content

def get_license_header(file_path):
    """Get the appropriate license header based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.py':
        return PY_LICENSE_HEADER
    else:
        return None

def add_license_to_file(file_path):
    """Add license header to a file if it doesn't already have one."""
    license_header = get_license_header(file_path)
    if license_header is None:
        print(f"Skipping unsupported file type: {file_path}")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        print(f"Skipping binary file: {file_path}")
        return False
    
    if has_license(content):
        print(f"License already exists in {file_path}")
        return False
    
    # Check if the file starts with a shebang or coding declaration
    lines = content.split('\n')
    prefix = ""
    
    if lines and (lines[0].startswith('#!') or lines[0].startswith('# -*- coding')):
        prefix = lines[0] + '\n\n'
        content = '\n'.join(lines[1:])
    
    # Add license header
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(prefix + license_header + content)
    
    print(f"Added license to {file_path}")
    return True

def process_directory(directory):
    """Process all Python files in the directory and its subdirectories."""
    modified_count = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                # Skip hidden files and directories
                if os.path.basename(file_path).startswith('.'):
                    continue
                    
                # Skip __pycache__ directories
                if '__pycache__' in file_path:
                    continue
                
                if add_license_to_file(file_path):
                    modified_count += 1
    
    return modified_count

if __name__ == "__main__":
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "."
    
    modified_count = process_directory(directory)
    print(f"Added license header to {modified_count} files.")