#!/bin/bash

echo "🔍 Starting code quality checks..."

echo "📏 Linting with flake8..."
flake8 . || { echo '❌ flake8 failed'; exit 1; }

echo "📐 Type checking with mypy..."
mypy . || { echo '❌ mypy failed'; exit 1; }

echo "✅ All checks passed successfully!"
