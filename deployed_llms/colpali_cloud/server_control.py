from pydantic import BaseModel
import requests
import time
import paramiko

# Global variable to track the last time the token was refreshed
last_refresh_time = 0
refresh_token_interval = 3600  # 3600 seconds


class SecuritySettings():
    """Class for security settings"""

    # define the api key for the datacrunch api
    CLIENT_SECRET: str = '3nmlP5a1Wr1gDtpAfMllBYuXA7rlxPybnkGDBEFYM6'
    CLIENT_ID: str = '6OVOPvMNtXmVWAZ7ERCpD'


class CloudSettings(BaseModel):
    """Settings for Vertex AI server control"""

    # define the base url for the datacrunch api
    BASE_URL: str = "https://api.datacrunch.io/v1"

    # define the instance parameters
    INSTANCE_TYPE: str = "1V100.6V"
    IMAGE: str = "ubuntu-22.04"
    SSH_KEY_IDS: str = "f3591573-bb1e-4832-934c-9b1faf84257e"
    HOSTNAME: str = "jv-colpali-01"
    DESCRIPTION: str = "server for running colpali model"
    LOCATION_CODE: str = "FIN-01"

    # define the size of the storage volume
    STORAGE_VOLUME_SIZE: int = 50
    VOLUMES: list[dict] = [{
        "name": "instance_volume",
        "size": STORAGE_VOLUME_SIZE,
        "type": "NVMe"
    }]


class SecurityOperations():
    """Class for security operations"""

    def __init__(self):
        self.security_settings = SecuritySettings()

    def get_access_token(self):
        """Get the refresh token for the datacrunch api"""

        # define the url for the datacrunch api
        url = f"{self.cloud_settings.BASE_URL}/oauth2/token"

        # define the payload for the datacrunch api
        payload = {
            "grant_type": "client_credentials",
            "client_id": self.security_settings.CLIENT_ID,
            "client_secret": self.security_settings.CLIENT_SECRET
        }

        # define the headers for the datacrunch api
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        # send the request to the datacrunch api
        response = requests.post(url, data=payload, headers=headers)

        # print the response from the datacrunch api
        return response.json()['access_token']


class LaunchInstance(SecurityOperations):
    """Class for launching a new instance in the cloud"""

    def __init__(self):
        super().__init__()
        self.cloud_settings = CloudSettings()

        # get the access token
        self.access_token = self.access_token_management()

    def access_token_management(self):
        """Get the refresh token for the datacrunch api, ensuring it's called only once every 3600 seconds."""

        # global variable to track the last time the token was refreshed
        global last_refresh_time

        # get the current time
        current_time = time.time()

        # check if the token was refreshed in the last 3600 seconds
        if current_time - last_refresh_time >= refresh_token_interval:
            # refresh the token
            last_refresh_time = current_time
            # get the access token
            # fix the access token getting issue cause there is no function name get_refresh_token
            # implement that function in the class SecurityOperations
            return self.get_access_token(
            )  # Call the parent class method to get the token
        else:
            raise Exception(
                "Refresh token can only be retrieved once every 3600 seconds.")

    def deploy_new_instance(self, **kwargs):
        """Deploy a new instance to the cloud of datacrunch io"""

        # define the url for the datacrunch api
        url = self.cloud_settings.BASE_URL + "/instances"

        # get the volume id from kwargs 
        volume_id: str = kwargs.get("volume_id", None)

        # define the payload for the datacrunch api
        payload = {
            "instance_type": self.cloud_settings.INSTANCE_TYPE,
            "image": self.cloud_settings.IMAGE,
            "ssh_key_ids": self.cloud_settings.SSH_KEY_IDS,
            "hostname": self.cloud_settings.HOSTNAME,
            "description": self.cloud_settings.DESCRIPTION,
            "location_code": self.cloud_settings.LOCATION_CODE,
            #"volumes": self.cloud_settings.VOLUMES
        }

        if volume_id:
            payload["existing_volumes"] = [volume_id]

        # define the headers for the datacrunch api
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.access_token}"
        }

        # send the request to the datacrunch api
        response = requests.post(url, json=payload, headers=headers)
        # print the response from the datacrunch api
        print(response.text)
        print(response.status_code)
        return response.text

    def start_instance(self, instance_id: str):
        """Start a specific instance in the cloud"""

        # define the url for the datacrunch api
        url = f"{self.cloud_settings.BASE_URL}/instances"

        # define the payload for the datacrunch api
        payload = {"id": instance_id, "action": "start"}

        # define the headers for the datacrunch api
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}"
        }

        # run a loop to retry starting the instance every 30 seconds until it is ready
        while True:
            # wait for 30 seconds before trying again
            time.sleep(30)

            # send the request to the datacrunch api
            response = requests.put(url, json=payload, headers=headers)

            # check the status code
            if response.status_code < 400:
                break

        print(response.status_code)
        # print the response from the datacrunch api
        return response.text

    def delete_instance(self, instance_id: str):
        """Hibernate a specific instance in the cloud"""

        # define the url for the datacrunch api
        url = f"{self.cloud_settings.BASE_URL}/instances"

        # define the payload for the datacrunch api
        payload = {"id": instance_id, "action": "delete"}

        # define the headers for the datacrunch api
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}"
        }

        # send the request to the datacrunch api
        response = requests.put(url, json=payload, headers=headers)

        # print the response from the datacrunch api
        return response.text

    def get_instance_status(self, instance_id: str):
        """Get the status of the instance"""

        # define the url for the datacrunch api
        url = f"{self.cloud_settings.BASE_URL}/instances/{instance_id}"

        # define the headers for the datacrunch api
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.access_token}"
        }

        # send the request to the datacrunch api
        response = requests.get(url, headers=headers)

        # print the response from the datacrunch api
        return response.json()

    def create_storage_volume(self, instance_id: str):
        """Create a new storage volume in the cloud"""

        # define the url for the datacrunch api
        url = f"{self.cloud_settings.BASE_URL}/volumes"

        # define the payload for the datacrunch api
        payload = {
            "name": "volume1",
            "size": self.cloud_settings.STORAGE_VOLUME_SIZE,
            "type": "NVMe",
            "instance_id": instance_id,
            "location_code": self.cloud_settings.LOCATION_CODE
        }

        # define the headers for the datacrunch api
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}"
        }

        # send the request to the datacrunch api
        response = requests.post(url, json=payload, headers=headers)

        # print the response from the datacrunch api
        return response.json()

    def attach_volume(self, volume_id: str, instance_id: str):
        """Attach a storage volume to the instance"""

        # define the url for the datacrunch api
        url = f"{self.cloud_settings.BASE_URL}/volumes"

        # define the payload for the datacrunch api
        payload = {"id": volume_id, "action": "attach", "instance_id": instance_id}

        # define the headers for the datacrunch api
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}"
        }

        # send the request to the datacrunch api
        response = requests.put(url, json=payload, headers=headers)

        # print the response from the datacrunch api
        return response.json().get("VolumeId")

    def delete_volume(self, volume_id: str): 
        """delete the volume for a particular instance given the volume id"""

        # define the url for the datacrunch api
        url = f"{self.cloud_settings.BASE_URL}/volumes/{volume_id}"

        # define the request body
        body = {
            'is_permanent': True
        }

        # define the headers
        headers = {"Authorization": f"Bearer {self.access_token}"}
    
        response = requests.delete(url, headers=headers, json=body)
    
        print(response.json())

    def shutdown_instance(self, instance_id: str):
        """Shutdown a specific instance in the cloud"""

        # define the url for the datacrunch api
        url = f"{self.cloud_settings.BASE_URL}/instances"

        # define the headers for the datacrunch api
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}"
        }

        # define the payload for the datacrunch api
        payload = {"id": instance_id, "action": "shutdown"}

        # Check if the instance is running before attempting to shut it down
        while True:
            # wait for 10 seconds before checking again
            time.sleep(10)
            # get the instance status
            instance_status = self.get_instance_status(instance_id)
            print('Current instance status:', instance_status.get('status'))

            # check if the instance is running
            if instance_status.get('status') == 'running':
                # send the request to shut down the instance
                response = requests.put(url, headers=headers, json=payload)
                print('Shutdown response:', response.status_code)
                print('Shutdown response text:', response.text)
                break

        # print the response from the datacrunch api
        return response.text

    def volume_management(self, existing_volume_id: str, instance_id: str): 
        """manage the storage volume of the instance by deleting the volumes 
        that are not the existing volume id and attaching the existing volume id"""
        
        # get a list of all volumes
        url = f"{self.cloud_settings.BASE_URL}/volumes"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.access_token}"
        }
        response = requests.get(url, headers=headers)
        volumes = response.json()
        print('this is the volumes:', volumes)

        # get the volumes that are not the existing volume id
        volumes_to_delete = [volume for volume in volumes if volume['id'] != existing_volume_id]
        print('this is the volumes to delete:', volumes_to_delete)

        # delete the volumes that are not the existing volume id
        for volume in volumes_to_delete:
            self.delete_volume(volume['id'])
        
        # attach the existing volume id to the instance
        self.attach_volume(existing_volume_id, instance_id)


def scp_files_to_server(ip_address: str, local_file_path: str, remote_file_path: str, username: str, password: str):
    """Securely copy files to a server using SCP and run a script."""
    try:
        # Create an SSH client
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Connect to the server
        ssh.connect(ip_address, username=username, password=password)
        
        # Use SCP to transfer the file
        scp = paramiko.SFTPClient.from_transport(ssh.get_transport())
        scp.put(local_file_path, remote_file_path)
        
        # Close the SCP connection
        scp.close()
        
        print(f"Successfully copied {local_file_path} to {ip_address}:{remote_file_path}")
        
        # Run the vlm_api.py file on the server
        stdin, stdout, stderr = ssh.exec_command(f"python {remote_file_path}")
        print("Running vlm_api.py on the server...")
        print("Output:", stdout.read().decode())
        print("Errors:", stderr.read().decode())
        
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Close the SSH connection
        ssh.close()


# THIS IS THE TESTING SCRIPT


def test1():
    instance_manager = LaunchInstance(
    )  # Create an object of the LaunchInstance class
    instance_id = instance_manager.deploy_new_instance()
    # instance_id = "f91305d8-4151-4977-9ab8-d009c5dd9391"
    time.sleep(30)
    print(instance_id)
    print("Starting the instance...")
    start_response = instance_manager.start_instance(
        instance_id)  # Use the instance object to call the method
    print("Instance started:", start_response)

    time.sleep(10)  # Wait for 10 seconds

    print("Hibernating the instance...")
    delete_response = instance_manager.delete_instance(
        instance_id)  # Use the instance object to call the method
    print("Instance hibernated:", delete_response)


def test2(): 
    existing_volume_id: str = '1947784c-9083-499f-a7b8-e6deda35d222'
    
    # create a new instance
    instance_manager = LaunchInstance(
        )  # Create an object of the LaunchInstance class
    instance_id = instance_manager.deploy_new_instance(volume_id=existing_volume_id)
    # instance_id = "f91305d8-4151-4977-9ab8-d009c5dd9391"

    # wait for 30 seconds
    time.sleep(30)
    print('this is the instance id:', instance_id)

    # shutdown the instance
    print("Shutting down the instance...")
    shutdown_response = instance_manager.shutdown_instance(
        instance_id)  # Use the instance object to call the method
    print("Instance shutdown:", shutdown_response)

    # do the volume management
    #instance_manager.volume_management(existing_volume_id, instance_id)
    
    # start the instance
    #print("Starting the instance...")
    #start_response = instance_manager.start_instance(
    #    instance_id)  # Use the instance object to call the method
    #print("Instance started:", start_response)

def test3(): 

    # deploy a new instance
    instance_manager = LaunchInstance()  # Create an object of the LaunchInstance class
    instance_id = instance_manager.deploy_new_instance()
    
    # get the server status
    server_status = instance_manager.get_instance_status(instance_id)
    print('this is the server status:', server_status)

    # wait for 30 seconds
    time.sleep(30)
    
    # run a while loop to check if the server is running
    while True: 
        time.sleep(10)
        # get the server status
        server_status = instance_manager.get_instance_status(instance_id)
        print('this is the server status:', server_status)

        # copy the files to the server
        if server_status.get('status') == 'running':
            ip_address = server_status.get('ip')
            remote_file_path = '/home/ubuntu/server_control.py'
            username = 'ubuntu'
            password = 'password'

            for file_path in ['vlm_model.py', 'vlm_api.py']:
                scp_files_to_server(ip_address, file_path, remote_file_path, username, password)
 
test3()