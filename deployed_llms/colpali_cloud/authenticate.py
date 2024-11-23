
import requests
import xml.etree.ElementTree as ET

# Define the SOAP request details
url = "https://affwsapi.ams360.com/v2/service.asmx"
headers = {
    "Content-Type": "text/xml; charset=utf-8",
    "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/Authenticate"
}

# Replace these values with actual credentials
agency_no = "1002683-1"
login_id = "Bigbridgeai"
password = "Welcome1234"
employee_code = "!(-"



# SOAP request body
soap_request = f"""<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.W3.org/2001/XMLSchema" xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
  <soap:Body>
    <AgencyNo xmlns="http://www.WSAPI.AMS360.com/v2.0">{agency_no}</AgencyNo>
    <LoginId xmlns="http://www.WSAPI.AMS360.com/v2.0">{login_id}</LoginId>
    <Password xmlns="http://www.WSAPI.AMS360.com/v2.0">{password}</Password>
    <EmployeeCode xmlns="http://www.WSAPI.AMS360.com/v2.0">{employee_code}</EmployeeCode>
  </soap:Body>
</soap:Envelope>"""

# Send the request
response = requests.post(url, data=soap_request, headers=headers)

# Check for a successful response
if response.status_code == 200:
    # Parse the XML response
    root = ET.fromstring(response.content)

    # Find the token in the SOAP Header
    namespace = {"ns": "http://www.WSAPI.AMS360.com/v2.0"}
    token = root.find(".//ns:WSAPIAuthToken/ns:Token", namespace)

    if token is not None:
        print("Token:", token.text)
    else:
        print("Token not found in the response.")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
