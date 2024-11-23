#!/bin/bash

# Variables from secrets
API_TOKEN="ODZShc8cTLhlSgePOai5PzaEk6QBdrdd6GvlAXMWY2"
SSH_KEY_ID="f3591573-bb1e-4832-934c-9b1faf84257e"
SERVER_IP="65.108.33.89"
USER="root"

# First, let's get the SSH key details from DataCrunch API
echo "Retrieving SSH key information..."
SSH_KEY_INFO=$(curl -s -H "Authorization: Bearer $API_TOKEN" \
    -H "Accept: application/json" \
    "https://api.datacrunch.io/v1/ssh-keys/$SSH_KEY_ID")

# Save the SSH key to a file
echo "Setting up SSH key..."
SSH_DIR="$HOME/.ssh"
mkdir -p "$SSH_DIR"
chmod 700 "$SSH_DIR"

# Create SSH config
cat > "$SSH_DIR/config" << EOF
Host datacrunch
    HostName $SERVER_IP
    User $USER
    IdentityFile $SSH_DIR/datacrunch_key
    StrictHostKeyChecking no
EOF

chmod 600 "$SSH_DIR/config"

# Test the connection
echo "Testing SSH connection..."
ssh datacrunch "echo 'Connection successful!'"

if [ $? -eq 0 ]; then
    echo "✅ Successfully connected to the server!"
    echo "You can now connect to your server using:"
    echo "ssh datacrunch"
else
    echo "❌ Connection failed. Please check your API token and SSH key ID."
    echo "You may need to contact DataCrunch support for assistance."
fi
