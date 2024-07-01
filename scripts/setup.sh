#!/bin/bash

# Print welcome message
echo "Welcome to the setup script!"

# Make data directory
echo "Creating data directory..."
mkdir -p data
mkdir -p data/tuh_eeg

# Ask user for their name to set git config user.name
echo "Asking for your name to set git configuration..."
read -p "Please enter your name: " name
git config --global user.name "$name"

# Ask user for their email to set git config user.email
echo "Asking for your email to set git configuration..."
read -p "Please enter your email: " email
git config --global user.email "$email"

# Install required Python packages from requirements.txt
echo "Installing required Python packages..."
pip install -r requirements.txt

# Update package lists to get the latest version of available packages
echo "Updating package lists..."
apt-get update

# Install tmux and htop for terminal multiplexing and process monitoring
echo "Installing tmux and htop..."
apt-get install -y tmux htop rsync

# Generate a new SSH key using the provided email
echo "Generating a new SSH key..."
ssh-keygen -t ed25519 -C "$email" -f ~/.ssh/id_ed25519 

# Start the ssh-agent in the background
echo "Starting the ssh-agent..."
eval "$(ssh-agent -s)" 

# Add the SSH private key to the ssh-agent
echo "Adding the SSH private key to the ssh-agent..."
ssh-add ~/.ssh/id_ed25519 

# Display the newly created SSH public key
echo "SSH key created, here is your public key:"
cat ~/.ssh/id_ed25519.pub

# Print done message
echo "Setup is done!"
