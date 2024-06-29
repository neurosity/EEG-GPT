#!/bin/bash

# Print welcome message
echo "Welcome to the setup script!"

# Ask user for their name and email to set git config information
read -p "Please enter your name: " name
git config --global user.name "$name"

read -p "Please enter your email: " email
git config --global user.email "$email"

# Install required Python packages
pip install -r requirements.txt

# Update package lists and install tmux and htop
apt-get update
apt-get install -y tmux htop

# Print done message
echo "Setup is done!"
