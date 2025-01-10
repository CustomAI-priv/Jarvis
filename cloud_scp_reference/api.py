import os
from fastapi import FastAPI, HTTPException, Query, Depends, File, UploadFile, Form
import uvicorn
from typing import Dict, Any, List, Optional
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from backend.text_model import TextModel, AudioTranscriptionModel
from backend.analytical_model import AgentConfig
import base64
import random 
from pydantic import BaseModel, Field
import asyncio
# email imports
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from backend.chat_history_management import ChatHistoryManagement, UserManagement

from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from fastapi import WebSocket, Body
import time

# define the FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
text_model = TextModel()
audio_transcription_model = AudioTranscriptionModel()
analytical_model = AgentConfig()

# Add HTTP Basic Auth
security = HTTPBasic()

# define the chat history management
chat_history_mgmt = ChatHistoryManagement()
user_mgmt = UserManagement()


# define the email settings 
class EmailSettings(BaseModel):
    email_sender: str = 'bigbridgeai@gmail.com'
    app_password: str = 'cosl seed mcml rxqx'


def encode_image_to_base64(image_path: str) -> str:
    """Convert image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_html_files(directory: str) -> List[str]:
    html_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.html'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    html_content = f.read()
                html_files.append({"filename": file, "content": html_content})
    return html_files


def define_email_body(verification_code: str) -> str:
    """Define the email body"""
    return f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333333;
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }}
            .container {{
                background-color: #ffffff;
                border-radius: 8px;
                padding: 30px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
            }}
            .logo {{
                font-size: 24px;
                font-weight: bold;
                color: #2c5282;
            }}
            .code {{
                background-color: #f7fafc;
                border-radius: 4px;
                padding: 15px;
                font-size: 24px;
                font-weight: bold;
                text-align: center;
                letter-spacing: 5px;
                color: #2d3748;
                margin: 20px 0;
            }}
            .footer {{
                text-align: center;
                font-size: 14px;
                color: #718096;
                margin-top: 30px;
            }}
            .copy {{
                text-align: center;
                margin-top: 20px;
            }}
            .copy-button {{
                padding: 10px 15px;
                background-color: #2c5282;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }}
        </style>
        <script>
            function copyToClipboard() {{
                const code = document.getElementById('verification-code').innerText;
                navigator.clipboard.writeText(code).then(() => {{
                    alert('Verification code copied to clipboard!');
                }});
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">CustomAI</div>
            </div>
            <p>Hello,</p>
            <p>Your verification code is:</p>
            <div class="code" id="verification-code">{verification_code}</div>
            <button class="copy-button" onclick="copyToClipboard()">Copy Code</button>
            <p>Please use this code to complete your verification process.</p>
            <div class="footer">
                <p>This is an automated message, please do not reply.</p>
            </div>
        </div>
    </body>
    </html>
    """


def send_verification_email(email: str, verification_code: str) -> None:
    """Send a verification email to the user"""
    
    # define the email settings
    email_settings = EmailSettings()

    # define the message object
    message = MIMEMultipart("alternative")
    message["Subject"] = "CustomAI Verification Code"
    message["From"] = email_settings.email_sender
    message["To"] = email

    # define the email part object and attach it 
    part = MIMEText(define_email_body(verification_code), "html")
    message.attach(part)
    print('defined the email body')

    # Create secure connection with server and send email
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.ehlo()  # Can be omitted
            server.starttls(context=context)
            server.ehlo()  # Can be omitted
            server.login(email_settings.email_sender, email_settings.app_password)
            print('Logged in to the email server')
            server.sendmail(
                email_settings.email_sender, email, message.as_string()
            )
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False
    return True


# Store streams for each user and chat combination
user_streams: Dict[str, str] = {}

# Condition variable for notifying changes
stream_condition = asyncio.Condition()


class UpdateStreamRequest(BaseModel):
    output: str


@app.post("/update_stream/{user_id}/{chat_id}")
async def update_stream(user_id: str, chat_id: str, request: UpdateStreamRequest):
    user_key = f"{user_id}_{chat_id}"

    # Synchronize access to shared data
    async with stream_condition:
        if user_key in user_streams:
            user_streams[user_key] += request.output
        else:
            user_streams[user_key] = request.output
        stream_condition.notify_all()  # Notify all waiting tasks that new data is available

    return {"message": "Stream updated successfully"}


@app.get("/stream/{user_id}/{chat_id}", tags=["Real-Time Monitoring"])
async def stream_user_response(user_id: str, chat_id: str):
    async def event_generator():
        user_key = f"{user_id}_{chat_id}"
        if(user_key in user_streams):
            previous_data=user_streams[user_key]
        else:
            previous_data = ""

        while True:
            # Wait for new data to be added
            async with stream_condition:
                await stream_condition.wait()

                # Only send new data if it has changed
                if user_key in user_streams and user_streams[user_key] != previous_data:
                    current_data = user_streams[user_key]
#                    new_data_yield = current_data.replace(previous_data,"")
#                    yield new_data_yield
                    new_data = current_data[len(previous_data):]  # Extract new data that hasn't been sent yet
                    previous_data = current_data
                    yield new_data

    return StreamingResponse(event_generator(), media_type="text/plain", headers={"Cache-Control": "no-cache"})


# Define the Pydantic model for the request body
class QuestionRequest(BaseModel):
    question: str

@app.post("/text_model/")
def text_model_endpoint(
    user_id: str,
    chat_id: str,
    body: QuestionRequest
    #files: UploadFile = File(...)
    ) -> Dict[str, Any]:
    files = []

    # convert user_id and chat_id to integers
    user_id = int(user_id)
    chat_id = int(chat_id)

    try:
        # Handle file uploads if present
        file_contents = []

        if files:
            for file in files:
                content = file.read()
                temp_path = f"temp_{file.filename}"
                try:
                    # Save the uploaded file temporarily
                    with open(temp_path, "wb") as temp_file:
                        temp_file.write(content)
    
                    # Encode the image to base64
                    file_content = encode_image_to_base64(temp_path)
                    file_contents.append(file_content)
                except Exception as e:
                    return {
                        "status": 400,      
                        "detail": f"Your file: {file.filename} failed to upload. Try uploading again.",
                        "result": ''
                    }
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    
        # Process the request
        if file_contents:
            print("running file_content invoke")
            # Process files and question if both are provided
            result = text_model.invoke(
                question, 
                image_input=file_contents, 
                user_id=user_id, 
                chat_id=chat_id
            )


        else:
            # Process just the question if no files are provided
            # if not question:
            #     return {"status": 500, 'result': '', "detail": "Please provide either a question or upload files to proceed."}
            
            # Extract the question from the request body
            question = body.question

            # Validate that question is not empty
            if not question.strip():
                return {"result":"error", "status":400,"detail":"The 'question' field cannot be empty."}

            # Process the question (invoke your model here)
            result = text_model.invoke(
                question, disable_tabular=False, user_id=user_id, chat_id=chat_id
            )

        return {"result": result, "status": 200, "detail": ""}
    except Exception as e:
        return {"status": 500, "detail": "Internal Server Error. Please contact support at contact@thecustom.ai", 'result': ''}


@app.post("/analytical_model/")
def analytical_model_endpoint(
    user_id: str,
    chat_id: str,
    question: Optional[str] = None,
    #files: List[UploadFile] = File(...)
) -> Dict[str, Any]:
    files = []
    file_contents = []
    try:
        # Handle file uploads if present
        if files:
            for file in files:
                content = file.read()
                temp_path = f"temp_{file.filename}"
                try:
                    # Save the uploaded file temporarily
                    with open(temp_path, "wb") as temp_file:
                        temp_file.write(content)

                    # Encode the image to base64
                    file_content = encode_image_to_base64(temp_path)
                    file_contents.append(file_content)
                except Exception as e:
                    return {
                        "status": 400,
                        "result": '',
                        "detail": f"Your file {file.filename} failed to upload - please try uploading again."
                    }
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

        # Process the request
        if file_contents:
            # Process files and question if both are provided
            result = analytical_model.invoke(
                question, 
                image_input=file_contents, 
                user_id=user_id, 
                chat_id=chat_id
            )
        else:
            # Process just the question if no files are provided
            if not question:
                return {
                    "status": 400,
                    "result": '',
                    "detail": "Please provide either a question or upload files to proceed."
                }
            result = analytical_model.conditional_invoke(
                question,
                user_id=user_id,
                chat_id=chat_id
            )

        return {"result": result, "status": 200, "detail": ''}
    except Exception as e:
        return {
            "status": 500, 
            'result': '',
            "detail": "Internal Server Error - please contact support at contact@thecustom.ai"
        }


@app.post("/login")
async def login(email: str, password: str):
    try:
        user_id = user_mgmt.login_user(email, password)
        
        if user_id is not None:
            return {
                "status": 200,
                "detail": f"User with email '{email}' logged in successfully",
                "user_id": user_id
            }
        else:
            return {
                "status": 401,
                "user_id": '',
                "detail": "The email or password you entered is incorrect"
            }
    except Exception as e:
        return {
            "status": 500,
            "user_id": '',
            "detail": "Internal Server Error - please contact support at contact@thecustom.ai"
        }


@app.post("/send_verification_code")
async def send_verification_code(email: str, password: str):
    """Send a verification code to the user's email"""
    try:
        # Generate a random 6-digit verification code
        verification_code = str(random.randint(100000, 999999))
        print(f"Generated verification code: {verification_code}")
        
        # Send the email using the existing send_email function
        success = send_verification_email(email, verification_code)
        print(f"Verification code sent to {email}: {verification_code}")

        # Update the verification code for the user
        success = user_mgmt.enter_verification_code(email, password, verification_code)
        print(f"Verification code entered for {email}: {verification_code}")
        
        if success:
            return {
                "status": 200,
                "result": verification_code,
                "detail": "Verification code sent successfully to your email"
            }
        else:
            return {
                "status": 500,
                "result": "",
                "detail": "Verification code could not be sent. Check if the email address you provided is correct."
            }
    except:
        return {
            "status": 500,
            "result": "",
            "detail": "Internal Server Error - please contact support at contact@thecustom.ai"
        }


@app.post("/verify_verification_code")
async def verify_verification_code(email: str, verification_code: str):
    """Verify the verification code"""
    try:
        success = user_mgmt.verify_verification_code(email, verification_code)
        
        if success:
            return {
                "status": 200,
                "result": True,
                "detail": "Verification code successfully validated"
            }
        else:
            return {
                "status": 401,
                "result": False,
                "detail": "The verification code you entered is invalid"
            }
    except Exception as e:
        return {
            "status": 500,
            "result": False,
            "detail": "Internal Server Error - please contact support at contact@thecustom.ai"
        }


@app.post("/logout")
async def logout(username: str, password: str):
    try:
        # close the chat history management database 
        chat_history_mgmt.close()
        return {
            "status": 200,
            "result": True,
            "detail": "Successfully logged out"
        }
    except Exception as e:
        return {
            "status": 500,
            "result": False,
            "detail": "Internal Server Error - please contact support at contact@thecustom.ai"
        }


@app.post("/get_chat_history")
async def get_chat_history(user_id: int, model_id: int, chat_id: int):
    try:
        chat_history = chat_history_mgmt.get_chat_history(user_id, model_id, chat_id)
        return {
            "status": 200,
            "result": chat_history,
            "detail": "Chat history retrieved successfully"
        }
    except:
        return {
            "status": 500,
            "result": None,
            "detail": "Internal Server Error - please contact support at contact@thecustom.ai"
        }


@app.get("/get_chats")
async def get_chats(user_id: int):
    """Retrieve sorted chat IDs and model IDs and user IDs for a specific user."""
    try:
        chat_ids = chat_history_mgmt.get_chat_ids_sorted_by_timestamp(user_id)
        return {
            "status": 200,
            "result": chat_ids,
            "detail": "Chat history retrieved successfully"
        }
    except:
        return {
            "status": 500,
            "result": None,
            "detail": "Internal Server Error - please contact support at contact@thecustom.ai"
        }


@app.post("/delete_chat")
async def delete_chat(chat_id: int):
    try:
        # delete the chat by id
        chat_history_mgmt.delete_chat_by_id(chat_id)
        return {
            "status": 200,
            "result": True,
            "detail": "Chat deleted successfully"
        }
    except:
        return {
            "status": 500,
            "result": False,
            "detail": "Internal Server Error - please contact support at contact@thecustom.ai"
        }


# TODO: connect this to the text_model or analytical_model
@app.post("/transcribe_audio")
async def transcribe_audio(audio_file_path: str, clean_transcription: bool = True):
    try:
        # transcribe the audio file and extract the text
        transcription = audio_transcription_model.process_audio_file(audio_file_path, clean_transcription)
        return {
            "status": 200,
            "result": transcription,
            "detail": "Audio transcribed successfully"
        }
    except:
        return {
            "status": 500,
            "result": None,
            "detail": "Internal Server Error - please contact support at contact@thecustom.ai"
        }
    

@app.post("/register_user")
async def register_user(username: str, email: str, password: str):
    try:
        # register the user
        success = user_mgmt.register_user(username, email, password)
        
        if success:
            return {
                "status": 200,
                "result": True,
                "detail": "User registered successfully"
            }
        else:
            return {
                "status": 400,
                "result": False,
                "detail": "Username or email already exists"
            }
    except:
        return {
            "status": 500,
            "result": None,
            "detail": "Internal Server Error - please contact support at contact@thecustom.ai"
        }


if __name__ == "__main__":
    uvicorn.run(app, host='5.78.113.143', port=8005)
