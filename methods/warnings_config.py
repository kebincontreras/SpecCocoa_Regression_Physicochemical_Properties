"""
Global warnings configuration for the SpecCocoa Regression project.
This file suppresses specific warnings that do not affect functionality.
"""

import warnings
import os

# Global variable to avoid configuring warnings multiple times
_warnings_configured = False

def configure_warnings():
    """Configure warning filters to suppress non-critical messages."""
    global _warnings_configured
    
    # Only configure once
    if _warnings_configured:
        return
    
    # Very aggressive suppression
    warnings.filterwarnings('ignore')  # Suppress ALL warnings
    
    # Also suppress specifically
    warnings.filterwarnings('ignore', message='.*pkg_resources.*')
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    # Configure at system level
    import sys
    if hasattr(sys, 'filterwarnings'):
        sys.filterwarnings = lambda *args, **kwargs: None
    
    # Mark as configured
    _warnings_configured = True
    # WITHOUT PRINTING ANYTHING

# Configure warnings automatically when importing this module (only once)
configure_warnings()

# Environment variable to configure warnings in subprocesses
os.environ['PYTHONWARNINGS'] = 'ignore'
