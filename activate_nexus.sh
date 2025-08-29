#!/bin/bash
# Nexus WSL2 Activation Script

# Set X11 display if not set
if [ -z "$DISPLAY" ]; then
    export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0.0
    echo "X11 Display set to: $DISPLAY"
fi

# Activate virtual environment
source venv/bin/activate

# Set WSL2 flag
export NEXUS_PLATFORM=WSL2

echo "Nexus environment activated (WSL2 mode)"
echo "Run 'nexus --help' to get started"
