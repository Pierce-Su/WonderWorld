#!/bin/bash
# Helper script to set OpenAI API key

if [ -z "$1" ]; then
    echo "Usage: source setup_openai_key.sh YOUR_API_KEY"
    echo "Or: ./setup_openai_key.sh YOUR_API_KEY"
    echo ""
    echo "This will set OPENAI_API_KEY for the current session."
    echo "To make it permanent, add this line to your ~/.bashrc or ~/.zshrc:"
    echo "  export OPENAI_API_KEY='your_api_key_here'"
    exit 1
fi

export OPENAI_API_KEY="$1"
echo "✅ OPENAI_API_KEY has been set for this session"
echo "   To verify: echo \$OPENAI_API_KEY"

