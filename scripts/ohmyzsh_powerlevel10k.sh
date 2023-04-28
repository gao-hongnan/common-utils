#!/bin/bash

# Install Zsh (if not already installed)
if ! command -v zsh &> /dev/null; then
    echo "Zsh not found. Installing Zsh..."
    brew install zsh
else
    echo "Zsh found."
fi

# Install Oh My Zsh
if [ ! -d "$HOME/.oh-my-zsh" ]; then
    echo "Installing Oh My Zsh..."
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
else
    echo "Oh My Zsh already installed."
fi

# Install Powerlevel10k theme
if [ ! -d "${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k" ]; then
    echo "Installing Powerlevel10k theme..."
    git clone --depth=1 https://github.com/romkatv/powerlevel10k.git "${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k"
else
    echo "Powerlevel10k theme already installed."
fi

# Set the theme in .zshrc
echo "Setting Powerlevel10k as the theme in .zshrc..."
sed -i.bak 's/^ZSH_THEME=.*/ZSH_THEME="powerlevel10k\/powerlevel10k"/' ~/.zshrc
rm ~/.zshrc.bak

# Apply the changes
echo "Reloading Zsh to apply the changes..."
exec zsh
