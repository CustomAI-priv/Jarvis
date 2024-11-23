import os
from datacrunch import DataCrunchClient

# Get client secret from environment variable
CLIENT_SECRET = "ODZShc8cTLhlSgePOai5PzaEk6QBdrdd6GvlAXMWY2"
CLIENT_ID = '4rRxN1EitH0Flk1pkdrL2'  # Replace with your client ID

# Create datcrunch client
datacrunch = DataCrunchClient(CLIENT_ID, CLIENT_SECRET)

# Create new SSH key
public_key = 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAK1C5uTUgXDxfuxEUjqnaY1/oS2MZIMOwWLj4gOYRpR escobar@escobar'
ssh_key = datacrunch.ssh_keys.create('escobar', public_key)

# Print new key id, name, public key
print(ssh_key.id)
print(ssh_key.name)
print(ssh_key.public_key)

# Get all keys
all_ssh_keys = datacrunch.ssh_keys.get()
scripts = datacrunch.startup_scripts.get()
print(scripts)
# Get single key by id
some_ssh_key = datacrunch.ssh_keys.get_by_id(ssh_key.id)