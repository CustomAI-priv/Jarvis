import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import tempfile
import tarfile

def create_tar_archive(files_to_upload, backend_files):
    """Create a temporary tar archive of all files"""
    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
        with tarfile.open(fileobj=tmp, mode='w:gz') as tar:
            # Add backend files
            for file_name in backend_files:
                local_path = os.path.join('backend', file_name)
                if os.path.exists(local_path):
                    tar.add(local_path, f'backend/{file_name}')
            
            # Add api.py if it exists
            if os.path.exists('api.py'):
                tar.add('api.py', 'api.py')
        
        return tmp.name

def upload_and_extract(archive_path):
    """Upload and extract the archive on the remote server"""
    remote_base = '/root/BizEval_Upload_Auto'
    remote_archive = f'{remote_base}/upload.tar.gz'
    
    # Create remote directory
    subprocess.run(['ssh', 'root@5.78.113.143', f'mkdir -p {remote_base}/backend'])
    
    # Upload archive
    subprocess.run([
        'scp', '-C',  # Enable compression
        archive_path,
        f'root@5.78.113.143:{remote_archive}'
    ])
    
    # Extract on remote server and cleanup
    subprocess.run([
        'ssh', 'root@5.78.113.143',
        f'cd {remote_base} && tar xzf upload.tar.gz && rm upload.tar.gz'
    ])

def upload_files():
    # List of specific files to upload from backend directory
    backend_files = [
        'agent_prompts.py',
        'analytical_model.py',
        'api_keys.py',
        'chat_history_management.py',
        'custom_llms.py',
        'sector_settings.py',
        'setup.py',
        'text_model.py',
        'utils.py',
        'requirements.txt',
        '__init__.py'
    ]

    files_to_upload = []
    
    # Check which files exist
    for file_name in backend_files:
        local_path = os.path.join('backend', file_name)
        if os.path.exists(local_path):
            files_to_upload.append(local_path)
        else:
            print(f'Skipping (not found): {local_path}')
    
    if os.path.exists('api.py'):
        files_to_upload.append('api.py')
    
    print(f"Preparing to upload {len(files_to_upload)} files...")
    
    # Create archive
    archive_path = create_tar_archive(files_to_upload, backend_files)
    
    try:
        # Upload and extract
        upload_and_extract(archive_path)
        print("Upload completed successfully!")
    finally:
        # Cleanup temporary archive
        os.unlink(archive_path)

if __name__ == '__main__':
    upload_files()
