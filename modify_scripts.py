#!/usr/bin/env python3
"""
Modified Analysis Script Helper
------------------------------
This script modifies all analysis scripts to drop the Bubble column
"""

import os
import re
import glob

def modify_script(script_path):
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Find the species_columns definition
    species_pattern = r"species_columns\s*=\s*\[(.*?)\]"
    species_match = re.search(species_pattern, content, re.DOTALL)
    
    if species_match:
        species_list = species_match.group(1)
        # Remove 'Bubble' from the list
        modified_species = re.sub(r"'Bubble',?\s*", "", species_list)
        # Replace the original species list with the modified one
        modified_content = content.replace(species_match.group(0), f"species_columns = [{modified_species}]")
        
        # Write the modified content back to the file
        with open(script_path, 'w') as f:
            f.write(modified_content)
        
        return True
    else:
        return False

# Get all analysis scripts
script_files = glob.glob('/home/ubuntu/analysis_scripts/*.py')

# Modify each script
for script in script_files:
    print(f"Modifying {script}...")
    success = modify_script(script)
    if success:
        print(f"Successfully modified {script}")
    else:
        print(f"Failed to modify {script}")

print("\nAll scripts have been modified to drop the Bubble column.")
