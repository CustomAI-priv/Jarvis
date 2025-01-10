import os
import zipfile
import dropbox
import tarfile

class DropboxUploader:
    def __init__(self, app_key, app_secret, dbx_refresh_token):
        self.dbx = dropbox.Dropbox(
            app_key=app_key,
            app_secret=app_secret,
            oauth2_refresh_token=dbx_refresh_token
        )

    def zip_folder(self, folder_path, zip_name):
        """Compresses the specified folder into a zip file."""
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=folder_path)
                    zipf.write(file_path, arcname)
        print(f'Folder "{folder_path}" zipped as "{zip_name}".')

    def tar_folder(self, folder_path, tar_name):
        """
        Compresses the specified folder into a tar.gz file.
        Only includes .py, .json, .txt, .yaml, .csv, and .db files.
        Includes empty directories.
        Excludes .venv directory and its contents.
        """
        # Validate inputs
        if not folder_path or not os.path.exists(folder_path):
            raise ValueError(f"Invalid or non-existent folder path: {folder_path}")
            
        # Define allowed file extensions and excluded directories
        ALLOWED_EXTENSIONS = ('.py', '.json', '.txt', '.yaml', '.csv', '.db', '.toml')
        EXCLUDED_DIRS = {'.venv', 'venv', '__pycache__', '.git', 'site-packages'}
            
        # Ensure the tar_name ends with .tar.gz
        if not tar_name.endswith('.tar.gz'):
            tar_name += '.tar.gz'
            
        # Convert to absolute paths
        folder_path = os.path.abspath(folder_path)
        tar_name = os.path.abspath(tar_name)
            
        with tarfile.open(tar_name, 'w:gz') as tar:
            # Get the parent directory of the folder
            parent_dir = os.path.dirname(folder_path)
            # Get the base name of the folder
            folder_base = os.path.basename(folder_path)
            
            if not folder_base:
                raise ValueError("Invalid folder path: cannot get base name")
                
            # Change to parent directory to preserve folder structure
            original_dir = os.getcwd()
            os.chdir(parent_dir)
            
            try:
                # First add all empty directories
                for root, dirs, files in os.walk(folder_base):
                    # Skip excluded directories
                    dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
                    
                    # Add empty directories
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        if not any(os.scandir(dir_path)):  # Check if directory is empty
                            tar.add(dir_path)
                            print(f'Added empty directory: {dir_path}')
                
                # Then add files with allowed extensions
                for root, dirs, files in os.walk(folder_base):
                    # Skip excluded directories
                    dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
                    
                    for file in files:
                        if file.endswith(ALLOWED_EXTENSIONS):
                            file_path = os.path.join(root, file)
                            # Double check we're not in an excluded directory
                            if not any(excluded in file_path.split(os.sep) for excluded in EXCLUDED_DIRS):
                                tar.add(file_path)
                                print(f'Added file: {file_path}')
            finally:
                # Change back to original directory
                os.chdir(original_dir)
                
        print(f'Folder "{folder_path}" compressed as "{tar_name}"')
        print(f'Only {", ".join(ALLOWED_EXTENSIONS)} files were included.')
        print(f'Excluded directories: {", ".join(EXCLUDED_DIRS)}')

    def upload_to_dropbox(self, file_path, dropbox_path):
        """Uploads a file to Dropbox."""
        with open(file_path, 'rb') as f:
            file_size = os.path.getsize(file_path)
            CHUNK_SIZE = 4 * 1024 * 1024  # 4MB

            if file_size <= CHUNK_SIZE:
                self.dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode('overwrite'))
            else:
                upload_session_start_result = self.dbx.files_upload_session_start(f.read(CHUNK_SIZE))
                cursor = dropbox.files.UploadSessionCursor(
                    session_id=upload_session_start_result.session_id,
                    offset=f.tell()
                )
                commit = dropbox.files.CommitInfo(path=dropbox_path)

                while f.tell() < file_size:
                    if (file_size - f.tell()) <= CHUNK_SIZE:
                        self.dbx.files_upload_session_finish(f.read(CHUNK_SIZE), cursor, commit)
                    else:
                        self.dbx.files_upload_session_append_v2(f.read(CHUNK_SIZE), cursor)
                        cursor.offset = f.tell()

        print(f'File "{file_path}" uploaded to Dropbox at "{dropbox_path}".')

if __name__ == "__main__":
    # Define Dropbox credentials
    app_key = 'mjzyoqx7m3it8gb'
    app_secret = '01itl53fwmnnh1z'
    dbx_refresh_token = 'x5hflpVIknYAAAAAAAAAAdbvnEVbpPi713c-jUDoCJx3sheHytSDpf8hqcxA5kSj'

    # Initialize the uploader
    uploader = DropboxUploader(app_key, app_secret, dbx_refresh_token)

    # Define the folder to zip and upload
    folder_to_zip = '/Users/martinmashalov/Documents/RAG_Demo/BizEval/'  # Replace with your folder path
    zip_file_name = 'production_run.tar.gz'       # Desired name for the zip file
    dropbox_destination = '/production_run.tar.gz'  # Dropbox destination path

    # Zip the folder
    uploader.tar_folder(folder_to_zip, zip_file_name)

    # Upload the zipped folder to Dropbox
    uploader.upload_to_dropbox(zip_file_name, dropbox_destination)
