#!/bin/bash

set -e

# ----------------------------------
# ERROR HANDLER
# ----------------------------------

function handle_error {
    echo "ERROR: $1"
    exit 1
}

echo "🚀 Installing CRAFT..."
echo ""
echo "🔐 This installer may ask for your Mac password."
echo "This is required for installing system tools (Homebrew)."
echo ""

# ----------------------------------
# 1. Install Homebrew (if needed)
# ----------------------------------

if ! command -v brew &> /dev/null; then
    echo "🍺 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" \
        || handle_error "Failed to install Homebrew."

    echo "🔧 Setting up Homebrew PATH..."
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> "$HOME/.zprofile"
fi

# Ensure brew works in THIS script
eval "$(/opt/homebrew/bin/brew shellenv)"

echo "✅ Homebrew ready"

# ----------------------------------
# 2. Install system dependencies
# ----------------------------------

echo "📦 Installing system dependencies..."

brew unpin python@3.11 2>/dev/null || true
brew install git poppler cairo pango gdk-pixbuf libffi fontconfig python@3.11 python-tk@3.11
brew pin python@3.11

# ----------------------------------
# 3. Locate Python
# ----------------------------------

echo "🐍 Detecting Python..."

if command -v python3.11 &> /dev/null; then
    PYTHON_BIN=$(command -v python3.11)
elif command -v python3 &> /dev/null; then
    PYTHON_BIN=$(command -v python3)
else
    handle_error "Python not found after installation."
fi

echo "Using Python: $PYTHON_BIN"

# ----------------------------------
# 4. Ensure pip exists
# ----------------------------------

echo "📦 Ensuring pip is available..."

$PYTHON_BIN -m ensurepip --upgrade || true
$PYTHON_BIN -m pip install --upgrade pip

# ----------------------------------
# 5. Set target directory
# ----------------------------------

TARGET_DIR="$HOME/CRAFTtests"

# ----------------------------------
# 6. Download repo (no nesting)
# ----------------------------------

echo "⬇️ Setting up project in $TARGET_DIR"

if [ -d "$TARGET_DIR/.git" ]; then
    echo "🔄 Updating existing repository..."
    cd "$TARGET_DIR"
    git pull || echo "⚠️ Could not update repo"
else
    echo "📥 Cloning repository..."
    rm -rf "$TARGET_DIR"
    git clone https://github.com/cburst/CRAFT-Content-Restoration-Authorship-Familiarity-Test.git "$TARGET_DIR" \
        || handle_error "Failed to clone repository."
fi

# ----------------------------------
# 7. Install Python dependencies
# ----------------------------------

echo "📦 Installing Python dependencies..."

$PYTHON_BIN -m pip install -r "$TARGET_DIR/requirements.txt" \
    || handle_error "Failed to install Python dependencies."

# ----------------------------------
# 8. Install app to Applications
# ----------------------------------

echo "📦 Installing CRAFT.app..."

APP_SOURCE="$TARGET_DIR/dist/CRAFT.app"
APP_TARGET="/Applications/CRAFT.app"

if [ -d "$APP_SOURCE" ]; then
    cp -R "$APP_SOURCE" "$APP_TARGET" \
        || handle_error "Failed to copy CRAFT.app"

    echo "✅ App installed to /Applications"
else
    echo "⚠️ CRAFT.app not found in dist/ — skipping app install"
fi

# ----------------------------------
# 9. Done
# ----------------------------------

echo ""
echo "✅ Installation complete!"
echo ""
echo "👉 Launch the app from Applications (CRAFT.app)"
echo "👉 On first run, you will be prompted for your API key"
echo ""
echo "Press any key to close..."
read -n 1