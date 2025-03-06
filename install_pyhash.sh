#!/bin/bash
# save as install_pyhash.sh

# Save current setuptools version
CURRENT_SETUPTOOLS=$(pip freeze | grep setuptools)

# Downgrade setuptools
pip install setuptools==57.5.0

# Install pyhash
pip install pyhash

# Restore original setuptools
pip install $CURRENT_SETUPTOOLS

echo "pyhash installed successfully"
