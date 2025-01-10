from xml.sax import saxutils
import xml.etree.ElementTree as ET
import requests
import base64
from datetime import datetime
import xmltodict
from pprint import pprint

# Define the URL for the SOAP service
url = 'https://affwsapi.ams360.com/v2/service.asmx'

# Your token obtained via authentication
token: str = ''

def extract_data(element: ET.Element) -> dict:
    data = {}
    for child in element:
        tag = child.tag.split('}')[-1]
        if len(child) == 0:
            data[tag] = child.text
        else:
            for i, subchild in enumerate(child):
                subtag = subchild.tag.split('}')[-1]
                for subsubchild in subchild:
                    subsubtag = subsubchild.tag.split('}')[-1]
                    data[f"{tag}_{i+1}_{subtag}_{subsubtag}"] = subsubchild.text
    return data


def authenticate():

    # Define your credentials
    agency_no = "1002683-1"
    login_id = "Bigbridgeai"
    password = "Welcome1234"
    employee_code = "!(-"

    # AMS360 SOAP API endpoint
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # SOAPAction header
    headers = {
        "Content-Type": "text/xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/Authenticate"
    }

    # SOAP envelope with authentication details
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
    <soap:Body>
        <AgencyNo xmlns="http://www.WSAPI.AMS360.com/v2.0">{agency_no}</AgencyNo>
        <LoginId xmlns="http://www.WSAPI.AMS360.com/v2.0">{login_id}</LoginId>
        <Password xmlns="http://www.WSAPI.AMS360.com/v2.0">{password}</Password>
        <EmployeeCode xmlns="http://www.WSAPI.AMS360.com/v2.0">{employee_code}</EmployeeCode>
    </soap:Body>
    </soap:Envelope>"""

    # Send the POST request
    response = requests.post(url, data=soap_body, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:

        # Parse the XML response
        root = ET.fromstring(response.text)

        # Define namespaces
        namespaces = {
            'soap': 'http://schemas.xmlsoap.org/soap/envelope/',
            'wsapi': 'http://www.WSAPI.AMS360.com/v2.0'
        }

        # Use XPath to find the token
        token = root.find('.//soap:Header/wsapi:WSAPIAuthToken/wsapi:Token', namespaces)

        if token is not None:
            ticket = token.text
            return ticket
        else:
            raise Exception("Token not found")
    else:
        raise Exception("Authentication failed")


escaped_token = authenticate()


def changePolicyPersonnel(escaped_token):
    # Define the URL for the SOAP service
    url = 'https://affwsapi.ams360.com/v2/service.asmx'
    # SOAP Request Body (as XML)
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{escaped_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <ChangePolicyPersonnel_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <PolicyId>123456</PolicyId>
          <ModifyList>
            <PolicyPersonnel>
              <EmployeeCode>EMP001</EmployeeCode>
              <EmployeeType>Sales</EmployeeType>
              <IsPrimary>true</IsPrimary>
            </PolicyPersonnel>
            <PolicyPersonnel>
              <EmployeeCode>EMP002</EmployeeCode>
              <EmployeeType>Service</EmployeeType>
              <IsPrimary>false</IsPrimary>
            </PolicyPersonnel>
          </ModifyList>
          <DeleteList>
            <PolicyPersonnel>
              <EmployeeCode>EMP003</EmployeeCode>
              <EmployeeType>Support</EmployeeType>
              <IsPrimary>false</IsPrimary>
            </PolicyPersonnel>
            <PolicyPersonnel>
              <EmployeeCode>EMP004</EmployeeCode>
              <EmployeeType>Admin</EmployeeType>
              <IsPrimary>true</IsPrimary>
            </PolicyPersonnel>
          </DeleteList>
        </ChangePolicyPersonnel_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Debugging: print the full SOAP request to check the structure
    print("SOAP Body:")
    print(soap_body)

    # Headers (Note that Token should be a string and passed as plain text)
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/ChangePolicyPersonnel',
        'Authorization': f"Bearer {escaped_token}"  # Use Authorization header for the token
    }

    # Send the SOAP request using requests.post
    response = requests.post(url, data=soap_body, headers=headers)

    # Check the response
    if response.status_code == 200:
        print("Response Data:", response.text)
    else:
        print(f"Error: {response.status_code}")
        print(f"Response Body: {response.text}")

    # Parse the XML response
    root = ET.fromstring(response.text)

    # Define the namespace
    namespace = {'soap': 'http://www.w3.org/2003/05/soap-envelope',
                 'wsapi': 'http://www.WSAPI.AMS360.com/v2.0'}

    # Extract Overall Result and Severity
    overall_result = root.find('.//wsapi:OverallResult', namespace).text
    overall_severity = root.find('.//wsapi:OverallSeverity', namespace).text

    # Extract WSAPIResults and their details
    wsapi_results = root.findall('.//wsapi:WSAPIResult', namespace)

    # Loop through each result and extract the relevant information
    for result in wsapi_results:
        message = result.find('wsapi:Message', namespace).text
        result_code = result.find('wsapi:Result', namespace).text
        data_name = result.find('wsapi:DataName', namespace).text
        data_value = result.find('wsapi:DataValue', namespace).text
        severity = result.find('wsapi:Severity', namespace).text

        # Print or store the extracted data
        print(f"Message: {message}")
        print(f"Result: {result_code}")
        print(f"DataName: {data_name}")
        print(f"DataValue: {data_value}")
        print(f"Severity: {severity}")
        print("-----")

    # Optionally, print overall result information
    print(f"Overall Result: {overall_result}")
    print(f"Overall Severity: {overall_severity}")


def getBroker(escaped_token, broker_code="BKR001", short_name="BrokerSmith"):
    # Define the URL for the SOAP service
    url = 'https://affwsapi.ams360.com/v2/service.asmx'
    # SOAP Request Body (as XML)
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{escaped_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetBroker_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <BrokerCode>{broker_code}</BrokerCode>
          <ShortName>{short_name}</ShortName>
        </GetBroker_Request>
      </soap12:Body>
    </soap12:Envelope>
    """
    # Debugging: print the full SOAP request to check the structure
    print("SOAP Body:")
    print(soap_body)

    # Headers (Note that Token should be a string and passed as plain text)
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/GetBroker',
        'Authorization': f"Bearer {escaped_token}"  # Use Authorization header for the token
    }

    # Send the SOAP request using requests.post
    response = requests.post(url, data=soap_body, headers=headers)

    # Check the response
    if response.status_code == 200:
        print("Response Data:", response.text)
    else:
        print(f"Error: {response.status_code}")
        print(f"Response Body: {response.text}")

    # Parse the XML
    root = ET.fromstring(response.text)

    # Extract elements with namespace
    namespace = {'ns': 'http://www.WSAPI.AMS360.com/v2.0'}

    # Extract OverallResult and OverallSeverity
    overall_result = root.find('.//ns:OverallResult', namespace).text
    overall_severity = root.find('.//ns:OverallSeverity', namespace).text

    # Extract details from WSAPIResult
    wsapi_result = root.find('.//ns:WSAPIResult', namespace)
    message = wsapi_result.find('ns:Message', namespace).text
    result = wsapi_result.find('ns:Result', namespace).text
    severity = wsapi_result.find('ns:Severity', namespace).text

    # Print extracted data
    print("OverallResult:", overall_result)
    print("OverallSeverity:", overall_severity)
    print("Message:", message)
    print("Result:", result)
    print("Severity:", severity)


def getBrokerList(escaped_token , last_name_prefix = "Jo", filter_active = "true" ):
    # Define the URL for the SOAP service
    url = 'https://affwsapi.ams360.com/v2/service.asmx'

    # SOAP request body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{escaped_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetBrokerList_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <LastNamePrefix>{last_name_prefix}</LastNamePrefix>
          <FilterActive>{filter_active}</FilterActive>
        </GetBrokerList_Request>
      </soap12:Body>
    </soap12:Envelope>
    """
    # Debugging: print the full SOAP request to check the structure
    print("SOAP Body:")
    print(soap_body)

    # Headers (Note that Token should be a string and passed as plain text)
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/GetBrokerList',
        'Authorization': f"Bearer {escaped_token}"  # Use Authorization header for the token
    }

    # Send the SOAP request using requests.post
    response = requests.post(url, data=soap_body, headers=headers)

    # Check the response
    if response.status_code == 200:
        print("Response Data:", response.text)
    else:
        print(f"Error: {response.status_code}")
        print(f"Response Body: {response.text}")

    # Parse the XML
    root = ET.fromstring(response.text)
    # Parse the response
    namespace = {'soap': 'http://www.w3.org/2003/05/soap-envelope', 'ns': 'http://www.WSAPI.AMS360.com/v2.0'}

    # Extract data
    overall_result = root.find('.//ns:OverallResult', namespace).text
    overall_severity = root.find('.//ns:OverallSeverity', namespace).text
    message = root.find('.//ns:WSAPIResult/ns:Message', namespace).text
    result = root.find('.//ns:WSAPIResult/ns:Result', namespace).text
    severity = root.find('.//ns:WSAPIResult/ns:Severity', namespace).text

    # Print extracted data
    print("Overall Result:", overall_result)
    print("Overall Severity:", overall_severity)
    print("Message:", message)
    print("Result:", result)
    print("Severity:", severity)


def getClaimList(escaped_token, claim_number = "12345",policy_number = "ABC-12345",get_related_data = 'true'):

    url = 'https://affwsapi.ams360.com/v2/service.asmx'
    # SOAP Request Body (as XML)
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{escaped_token}</Token>  # Replace with actual token
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetClaimList_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <ClaimNumber>{claim_number}</ClaimNumber>
          <PolicyNumber>{policy_number}</PolicyNumber>
          <GetRelatedData>{str(get_related_data).lower()}</GetRelatedData>
        </GetClaimList_Request>
      </soap12:Body>
    </soap12:Envelope>
    """

    # Headers (Note that Token should be a string and passed as plain text)
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/GetClaimList',
        'Authorization': f"Bearer {escaped_token}"  # Use Authorization header for the token
    }

    # Send the SOAP request using requests.post
    response = requests.post(url, data=soap_body, headers=headers)
    return response.text


def getCompany(escaped_token,company_code = "COMP123", short_name = "Tech Corp" ):

    url = 'https://affwsapi.ams360.com/v2/service.asmx'

    # SOAP request body as XML
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{escaped_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetCompany_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <CompanyCode>{company_code}</CompanyCode>
          <ShortName>{short_name}</ShortName>
        </GetCompany_Request>
      </soap12:Body>
    </soap12:Envelope>
    """
    # Debugging: print the full SOAP request to check the structure
    print("SOAP Body:")
    print(soap_body)

    # Headers (Note that Token should be a string and passed as plain text)
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/GetCompany',
        'Authorization': f"Bearer {escaped_token}"  # Use Authorization header for the token
    }

    # Send the SOAP request using requests.post
    response = requests.post(url, data=soap_body, headers=headers)

    # Check the response
    if response.status_code == 200:
        print("Response Data:", response.text)
    else:
        print(f"Error: {response.status_code}")
        print(f"Response Body: {response.text}")

    # Parse the XML
    root = ET.fromstring(response.text)
    # Parse the response
    # Define the namespace
    ns = {"soap": "http://www.w3.org/2003/05/soap-envelope", "wsapi": "http://www.WSAPI.AMS360.com/v2.0"}

    # Extract OverallResult, OverallSeverity, Message, Result, and Severity
    overall_result = root.find(".//wsapi:OverallResult", ns).text
    overall_severity = root.find(".//wsapi:OverallSeverity", ns).text
    message = root.find(".//wsapi:WSAPIResult/wsapi:Message", ns).text
    result_code = root.find(".//wsapi:WSAPIResult/wsapi:Result", ns).text
    severity = root.find(".//wsapi:WSAPIResult/wsapi:Severity", ns).text

    # Print extracted data
    print("Overall Result:", overall_result)
    print("Overall Severity:", overall_severity)
    print("Message:", message)
    print("Result Code:", result_code)
    print("Severity:", severity)


def CompanyList(escaped_token,name_prefix = "ABC",company_type_list = "Type1,Type2",parent_company_code = "P123",filter_active = "true"):

    url = 'https://affwsapi.ams360.com/v2/service.asmx'
    # SOAP Request Body (as XML)
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{escaped_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetCompanyList_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <NamePrefix>{name_prefix}</NamePrefix>
          <CompanyTypeList>{company_type_list}</CompanyTypeList>
          <ParentCompanyCode>{parent_company_code}</ParentCompanyCode>
          <FilterActive>{filter_active}</FilterActive>
        </GetCompanyList_Request>
      </soap12:Body>
    </soap12:Envelope>
    """

    # Debugging: print the full SOAP request to check the structure
    print("SOAP Body:")
    print(soap_body)

    # Headers (Note that Token should be a string and passed as plain text)
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/GetCompanyList',
        'Authorization': f"Bearer {escaped_token}"  # Use Authorization header for the token
    }

    # Send the SOAP request using requests.post
    response = requests.post(url, data=soap_body, headers=headers)

    # Check the response
    if response.status_code == 200:
        print("Response Data:", response.text)
    else:
        print(f"Error: {response.status_code}")
        print(f"Response Body: {response.text}")

    # Parse the XML
    root = ET.fromstring(response.text)
    # Parse the XML response
    namespace = {'soap': 'http://www.w3.org/2003/05/soap-envelope'}

    # Extract data
    overall_result = root.find('.//{http://www.WSAPI.AMS360.com/v2.0}OverallResult').text
    overall_severity = root.find('.//{http://www.WSAPI.AMS360.com/v2.0}OverallSeverity').text
    message = root.find('.//{http://www.WSAPI.AMS360.com/v2.0}Message').text
    result_code = root.find('.//{http://www.WSAPI.AMS360.com/v2.0}Result').text
    severity = root.find('.//{http://www.WSAPI.AMS360.com/v2.0}Severity').text

    # Print extracted data
    print("Overall Result:", overall_result)
    print("Overall Severity:", overall_severity)
    print("Message:", message)
    print("Result Code:", result_code)
    print("Severity:", severity)


def getCustomer(escaped_token,customer_number = 12345,customer_id = "CUST001",get_related_data = "true" ):

    url = 'https://affwsapi.ams360.com/v2/service.asmx'
    # SOAP Request Body (as XML)
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{escaped_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetCustomer_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <CustomerNumber>{customer_number}</CustomerNumber>
          <CustomerId>{customer_id}</CustomerId>
          <GetRelatedData>{get_related_data}</GetRelatedData>
        </GetCustomer_Request>
      </soap12:Body>
    </soap12:Envelope>
    """

    # Headers (Note that Token should be a string and passed as plain text)
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/GetCustomer',
        'Authorization': f"Bearer {escaped_token}"  # Use Authorization header for the token
    }

    # Send the SOAP request using requests.post
    response = requests.post(url, data=soap_body, headers=headers)
    return response.text


def getCustomerList(escaped_token,name_prefix = "John",type_of_customer = "Individual",is_brokers_customer = "true",get_related_data = "true",filter_active = "true"):

    url = 'https://affwsapi.ams360.com/v2/service.asmx'
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{escaped_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetCustomerList_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <NamePrefix>{name_prefix}</NamePrefix>
          <TypeOfCustomer>{type_of_customer}</TypeOfCustomer>
          <IsBrokersCustomer>{is_brokers_customer}</IsBrokersCustomer>
          <GetRelatedData>{get_related_data}</GetRelatedData>
          <FilterActive>{filter_active}</FilterActive>
        </GetCustomerList_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Headers (Note that Token should be a string and passed as plain text)
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/GetCustomerList',
        'Authorization': f"Bearer {escaped_token}"  # Use Authorization header for the token
    }

    # Send the SOAP request using requests.post
    response = requests.post(url, data=soap_body, headers=headers)
    return response.text


def getCustomerProfileAnswer(escaped_token,customer_id = "CUST001",question_id = "QST001" ):

    url = 'https://affwsapi.ams360.com/v2/service.asmx'

    # SOAP request body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{escaped_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetCustomerProfileAnswer_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <CustomerId>{customer_id}</CustomerId>
          <QuestionId>{question_id}</QuestionId>
        </GetCustomerProfileAnswer_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Headers (Note that Token should be a string and passed as plain text)
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/GetCustomerProfileAnswer',
        'Authorization': f"Bearer {escaped_token}"  # Use Authorization header for the token
    }

    # Send the SOAP request using requests.post
    response = requests.post(url, data=soap_body, headers=headers)
    return response.text


def getCustomerProfileAnswerList(escaped_token,customer_id = "CUST001"):

    url = 'https://affwsapi.ams360.com/v2/service.asmx'

    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
                     xmlns:xsd="http://www.w3.org/2001/XMLSchema" 
                     xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{escaped_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetCustomerProfileAnswerList_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <CustomerId>{customer_id}</CustomerId>
        </GetCustomerProfileAnswerList_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Headers (Note that Token should be a string and passed as plain text)
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/GetCustomerProfileAnswerList',
        'Authorization': f"Bearer {escaped_token}"  # Use Authorization header for the token
    }

    # Send the SOAP request using requests.post
    response = requests.post(url, data=soap_body, headers=headers)
    return response.text


def getEmployee(escaped_token,employee_code = "EMP001",short_name = "John Doe"):

    url = 'https://affwsapi.ams360.com/v2/service.asmx'

    # SOAP Request Body (as XML)
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{escaped_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetEmployee_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <EmployeeCode>{employee_code}</EmployeeCode>
          <ShortName>{short_name}</ShortName>
        </GetEmployee_Request>
      </soap12:Body>
    </soap12:Envelope>
    """

    # Headers (Note that Token should be a string and passed as plain text)
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/getEmployee',
        'Authorization': f"Bearer {escaped_token}"  # Use Authorization header for the token
    }

    # Send the SOAP request using requests.post
    response = requests.post(url, data=soap_body, headers=headers)
    return response.text


def getEmployeeList(escaped_token,last_name_prefix = "Jo",emp_type = "FullTime"):

    url = 'https://affwsapi.ams360.com/v2/service.asmx'
    # SOAP Request Body (as XML)
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{escaped_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetEmployeeList_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <LastNamePrefix>{last_name_prefix}</LastNamePrefix>
          <EmpType>{emp_type}</EmpType>
        </GetEmployeeList_Request>
      </soap12:Body>
    </soap12:Envelope>
    """
    # Debugging: print the full SOAP request to check the structure
    #print("SOAP Body:")
    #print(soap_body)

    # Headers (Note that Token should be a string and passed as plain text)
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/GetEmployeeList',
        'Authorization': f"Bearer {escaped_token}"  # Use Authorization header for the token
    }

    # Send the SOAP request using requests.post
    response = requests.post(url, data=soap_body, headers=headers)
    return response.text


def getFieldInfo(escaped_token, entity_name = "Policy",field_name = "EffectiveDate",return_list = "true"):

    url = 'https://affwsapi.ams360.com/v2/service.asmx'

    # SOAP request body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{escaped_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetFieldInfo_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <EntityName>{entity_name}</EntityName>
          <FieldName>{field_name}</FieldName>
          <ReturnList>{return_list}</ReturnList>
        </GetFieldInfo_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Headers (Note that Token should be a string and passed as plain text)
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/GetFieldInfo',
        'Authorization': f"Bearer {escaped_token}"  # Use Authorization header for the token
    }

    # Send the SOAP request using requests.post
    response = requests.post(url, data=soap_body, headers=headers)
    return response.text


def getGLBranch(escaped_token,gl_branch_code = "BR001",short_name = "MainBranch"):

    url = 'https://affwsapi.ams360.com/v2/service.asmx'
    # SOAP request body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
                     xmlns:xsd="http://www.w3.org/2001/XMLSchema" 
                     xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{escaped_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetGLBranch_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <GLBranchCode>{gl_branch_code}</GLBranchCode>
          <ShortName>{short_name}</ShortName>
        </GetGLBranch_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Headers (Note that Token should be a string and passed as plain text)
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/GetGLBranch',
        'Authorization': f"Bearer {escaped_token}"  # Use Authorization header for the token
    }

    # Send the SOAP request using requests.post
    response = requests.post(url, data=soap_body, headers=headers)

    return response.text


def getGLBranchList(escaped_token,name_prefix = "Branch",gl_division_code = "123",filter_active = "true"):

    url = 'https://affwsapi.ams360.com/v2/service.asmx'
    # SOAP request body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{escaped_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetGLBranchList_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <NamePrefix>{name_prefix}</NamePrefix>
          <GLDivisionCode>{gl_division_code}</GLDivisionCode>
          <FilterActive>{filter_active}</FilterActive>
        </GetGLBranchList_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Headers (Note that Token should be a string and passed as plain text)
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/GetGLBranchList',
        'Authorization': f"Bearer {escaped_token}"  # Use Authorization header for the token
    }

    # Send the SOAP request using requests.post
    response = requests.post(url, data=soap_body, headers=headers)

    return response.text


def getGLDepartment(escaped_token,gl_department_code = "FIN001",short_name = "Finance"):

    url = 'https://affwsapi.ams360.com/v2/service.asmx'

    # SOAP Request Body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{escaped_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetGLDepartment_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <GLDepartmentCode>{gl_department_code}</GLDepartmentCode>
          <ShortName>{short_name}</ShortName>
        </GetGLDepartment_Request>
      </soap12:Body>
    </soap12:Envelope>
    """

    # Headers (Note that Token should be a string and passed as plain text)
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/GetGLDepartment',
        'Authorization': f"Bearer {escaped_token}"  # Use Authorization header for the token
    }

    # Send the SOAP request using requests.post
    response = requests.post(url, data=soap_body, headers=headers)

    return response.text


def getGLDepartmentList(escaped_token,name_prefix = "HR",gl_branch_code = "BRANCH001",filter_active = "true"):

    url = 'https://affwsapi.ams360.com/v2/service.asmx'

    # SOAP request body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{escaped_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetGLDepartmentList_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <NamePrefix>{name_prefix}</NamePrefix>
          <GLBranchCode>{gl_branch_code}</GLBranchCode>
          <FilterActive>{filter_active}</FilterActive>
        </GetGLDepartmentList_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Headers (Note that Token should be a string and passed as plain text)
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/GetGLDepartmentList',
        'Authorization': f"Bearer {escaped_token}"  # Use Authorization header for the token
    }

    # Send the SOAP request using requests.post
    response = requests.post(url, data=soap_body, headers=headers)

    return response.text


def getGLDivision(escaped_token,gl_division_code = "DIV12345",short_name = "MainDivision"):

    url = 'https://affwsapi.ams360.com/v2/service.asmx'

    # SOAP Request Body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{escaped_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetGLDivision_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <GLDivisionCode>{gl_division_code}</GLDivisionCode>
          <ShortName>{short_name}</ShortName>
        </GetGLDivision_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Headers (Note that Token should be a string and passed as plain text)
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/GetGLDivision',
        'Authorization': f"Bearer {escaped_token}"  # Use Authorization header for the token
    }

    # Send the SOAP request using requests.post
    response = requests.post(url, data=soap_body, headers=headers)
    return response.text


def getGLDivisionList(escaped_token,name_prefix = "Finance",filter_active = "true"):

    url = 'https://affwsapi.ams360.com/v2/service.asmx'

    # SOAP Request Body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{escaped_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetGLDivisionList_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <NamePrefix>{name_prefix}</NamePrefix>
          <FilterActive>{filter_active}</FilterActive>
        </GetGLDivisionList_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Headers (Note that Token should be a string and passed as plain text)
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/GetGLDivisionList',
        'Authorization': f"Bearer {escaped_token}"  # Use Authorization header for the token
    }

    # Send the SOAP request using requests.post
    response = requests.post(url, data=soap_body, headers=headers)
    return response.text


def getGLGroup(escaped_token,gl_group_code = "GLGroup123",short_name = "FinanceGroup"):

    url = 'https://affwsapi.ams360.com/v2/service.asmx'

    # SOAP Request Body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{escaped_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetGLGroup_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <GLGroupCode>{gl_group_code}</GLGroupCode>
          <ShortName>{short_name}</ShortName>
        </GetGLGroup_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Debugging: print the full SOAP request to check the structure
    print("SOAP Body:")
    print(soap_body)

    # Headers (Note that Token should be a string and passed as plain text)
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/GetGLGroup',
        'Authorization': f"Bearer {escaped_token}"  # Use Authorization header for the token
    }

    # Send the SOAP request using requests.post
    response = requests.post(url, data=soap_body, headers=headers)

    # Check the response
    if response.status_code == 200:
        print("Response Data:", response.text)
    else:
        print(f"Error: {response.status_code}")
        print(f"Response Body: {response.text}")

    # Parse the XML
    root = ET.fromstring(response.text)

    # Parse the XML
    namespace = {'soap': 'http://www.w3.org/2003/05/soap-envelope',
                 'ns': 'http://www.WSAPI.AMS360.com/v2.0'}

    # Extract values
    overall_result = root.find('.//ns:OverallResult', namespace).text
    overall_severity = root.find('.//ns:OverallSeverity', namespace).text
    message = root.find('.//ns:WSAPIResult/ns:Message', namespace).text
    result = root.find('.//ns:WSAPIResult/ns:Result', namespace).text
    severity = root.find('.//ns:WSAPIResult/ns:Severity', namespace).text

    # Print extracted data
    print("OverallResult:", overall_result)
    print("OverallSeverity:", overall_severity)
    print("Message:", message)
    print("Result:", result)
    print("Severity:", severity)


def getGLGroupList(escaped_token,name_prefix = "SalesGroup",gl_department_code = "FinanceDept001",filter_active = "true"):

    url = 'https://affwsapi.ams360.com/v2/service.asmx'

    # SOAP Request Body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
                     xmlns:xsd="http://www.w3.org/2001/XMLSchema" 
                     xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{escaped_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetGLGroupList_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <NamePrefix>{name_prefix}</NamePrefix>
          <GLDepartmentCode>{gl_department_code}</GLDepartmentCode>
          <FilterActive>{filter_active}</FilterActive>
        </GetGLGroupList_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Debugging: print the full SOAP request to check the structure
    print("SOAP Body:")
    print(soap_body)

    # Headers (Note that Token should be a string and passed as plain text)
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/GetGLGroupList',
        'Authorization': f"Bearer {escaped_token}"  # Use Authorization header for the token
    }

    # Send the SOAP request using requests.post
    response = requests.post(url, data=soap_body, headers=headers)

    # Check the response
    if response.status_code == 200:
        print("Response Data:", response.text)
    else:
        print(f"Error: {response.status_code}")
        print(f"Response Body: {response.text}")

    # Parse the XML
    root = ET.fromstring(response.text)
    # Parse the XML
    ns = {"soap": "http://www.w3.org/2003/05/soap-envelope", "ns": "http://www.WSAPI.AMS360.com/v2.0"}

    # Extract required fields
    overall_result = root.find(".//ns:OverallResult", ns).text
    overall_severity = root.find(".//ns:OverallSeverity", ns).text
    wsapi_message = root.find(".//ns:WSAPIResult/ns:Message", ns).text
    wsapi_result = root.find(".//ns:WSAPIResult/ns:Result", ns).text
    wsapi_severity = root.find(".//ns:WSAPIResult/ns:Severity", ns).text

    # Output the extracted data
    print("Overall Result:", overall_result)
    print("Overall Severity:", overall_severity)
    print("WSAPI Message:", wsapi_message)
    print("WSAPI Result:", wsapi_result)
    print("WSAPI Severity:", wsapi_severity)


def getPolicy(escaped_token,policy_id = "1234",transaction_effective_date = "2024-01-01",get_related_data = "true" ):
    url = 'https://affwsapi.ams360.com/v2/service.asmx'

    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{escaped_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetPolicy_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <PolicyId>{policy_id}</PolicyId>
          <TransactionEffectiveDate>{transaction_effective_date}</TransactionEffectiveDate>
          <GetRelatedData>{get_related_data}</GetRelatedData>
        </GetPolicy_Request>
      </soap12:Body>
    </soap12:Envelope>
    """

    # Headers (Note that Token should be a string and passed as plain text)
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/GetPolicy',
        'Authorization': f"Bearer {escaped_token}"  # Use Authorization header for the token
    }

    # Send the SOAP request using requests.post
    response = requests.post(url, data=soap_body, headers=headers)
    return response.text


def getPolicyList(escaped_token,customer_number = "12345",policy_status = "Active",get_related_data = "true",filter_multi_entity = "false",filter_accounting_subtype = "false",
                  filter_submission_subtype = "false",filter_policy_subtype = "false"):
    BASE_URL = "https://wsapi.ams360.com/v3/WSAPIService.svc"

    def send_soap_request(session_token: str, action: str, body: str) -> ET.Element:
        soap_envelope = f"""<?xml version="1.0" encoding="utf-8"?>
        <s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/">
            <s:Header>
                <h:WSAPISession xmlns:h="http://www.WSAPI.AMS360.com/v3.0" xmlns:i="http://www.w3.org/2001/XMLSchema-instance">
                    <h:Ticket>{session_token}</h:Ticket>
                </h:WSAPISession>
            </s:Header>
            <s:Body>
                {body}
            </s:Body>
        </s:Envelope>
        """

        headers = {
            'Content-Type': 'text/xml; charset=utf-8',
            'SOAPAction': f'http://www.WSAPI.AMS360.com/v3.0/WSAPIServiceContract/{action}'
        }

        response = requests.post(BASE_URL, data=soap_envelope, headers=headers)
        return response.text
    
    body = f"""
    <PolicyGetListByCustomerNumber xmlns="http://www.WSAPI.AMS360.com/v3.0">
        <Request xmlns:d4p1="http://www.WSAPI.AMS360.com/v3.0/DataContract" xmlns:i="http://www.w3.org/2001/XMLSchema-instance">
            <d4p1:CustomerNumber>{customer_number}</d4p1:CustomerNumber>
        </Request>
    </PolicyGetListByCustomerNumber>
        """
    root = send_soap_request(escaped_token, "PolicyGetListByCustomerNumber", body)
    return root


def getLineOfBusiness(escaped_token,line_of_business_code = "CommercialProperty"):
    url = 'https://affwsapi.ams360.com/v2/service.asmx'

    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{escaped_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetLineOfBusiness_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <LineOfBusinessCode>{line_of_business_code}</LineOfBusinessCode>
        </GetLineOfBusiness_Request>
      </soap12:Body>
    </soap12:Envelope>
    """

    # Debugging: print the full SOAP request to check the structure
    print("SOAP Body:")
    print(soap_body)

    # Headers (Note that Token should be a string and passed as plain text)
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/GetLineOfBusiness',
        'Authorization': f"Bearer {escaped_token}"  # Use Authorization header for the token
    }

    # Send the SOAP request using requests.post
    response = requests.post(url, data=soap_body, headers=headers)

    # Check the response
    if response.status_code == 200:
        print("Response Data:", response.text)
        return response.text
    else:
        print(f"Error: {response.status_code}")
        print(f"Response Body: {response.text}")


def getLineOfBusinessList(escaped_token,line_of_business_code = "CommercialProperty" ):
    url = 'https://affwsapi.ams360.com/v2/service.asmx'

    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{escaped_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetLineOfBusiness_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <LineOfBusinessCode>{line_of_business_code}</LineOfBusinessCode>
        </GetLineOfBusiness_Request>
      </soap12:Body>
    </soap12:Envelope>
    """

    # Debugging: print the full SOAP request to check the structure
    print("SOAP Body:")
    print(soap_body)

    # Headers (Note that Token should be a string and passed as plain text)
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/GetLineOfBusinessList',
        'Authorization': f"Bearer {escaped_token}"  # Use Authorization header for the token
    }

    # Send the SOAP request using requests.post
    response = requests.post(url, data=soap_body, headers=headers)

    # Check the response
    if response.status_code == 200:
        print("Response Data:", response.text)
        return response.text
    else:
        print(f"Error: {response.status_code}")
        print(f"Response Body: {response.text}")


def getPlanType(escaped_token,plan_code = "HealthPlan001" ):

    url = 'https://affwsapi.ams360.com/v2/service.asmx'

    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{escaped_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetPlanType_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <PlanCode>{plan_code}</PlanCode>
        </GetPlanType_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Debugging: print the full SOAP request to check the structure
    print("SOAP Body:")
    print(soap_body)

    # Headers (Note that Token should be a string and passed as plain text)
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/GetPlanType',
        'Authorization': f"Bearer {escaped_token}"  # Use Authorization header for the token
    }

    # Send the SOAP request using requests.post
    response = requests.post(url, data=soap_body, headers=headers)

    # Check the response
    if response.status_code == 200:
        print("Response Data:", response.text)
        return response.text
    else:
        print(f"Error: {response.status_code}")
        print(f"Response Body: {response.text}")


def getPlanTypeList(escaped_token,company_code = "ABCInsuranceCo",filter_active = "true"):

    url = 'https://affwsapi.ams360.com/v2/service.asmx'

    # SOAP Request Body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{escaped_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetPlanTypeList_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <CompanyCode>{company_code}</CompanyCode>
          <FilterActive>{filter_active}</FilterActive>
        </GetPlanTypeList_Request>
      </soap12:Body>
    </soap12:Envelope>"""
    # Debugging: print the full SOAP request to check the structure
    print("SOAP Body:")
    print(soap_body)

    # Headers (Note that Token should be a string and passed as plain text)
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/GetPlanTypeList',
        'Authorization': f"Bearer {escaped_token}"  # Use Authorization header for the token
    }

    # Send the SOAP request using requests.post
    response = requests.post(url, data=soap_body, headers=headers)

    # Check the response
    if response.status_code == 200:
        print("Response Data:", response.text)
        return response.text
    else:
        print(f"Error: {response.status_code}")
        print(f"Response Body: {response.text}")


def getPPAPolicyDetail(escaped_token,policy_id = "123456",transaction_effective_date = "2024-05-15",get_related_data = "false"):
    # Construct the SOAP request body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{escaped_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetPPAPolicyDetail_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <PolicyId>{policy_id}</PolicyId>
          <TransactionEffectiveDate>{transaction_effective_date}</TransactionEffectiveDate>
          <GetRelatedData>{get_related_data}</GetRelatedData>
        </GetPPAPolicyDetail_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Headers (Note that Token should be a string and passed as plain text)
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/GetPPAPolicyDetail',
        'Authorization': f"Bearer {escaped_token}"  # Use Authorization header for the token
    }

    # Send the SOAP request using requests.post
    response = requests.post(url, data=soap_body, headers=headers)
    return response.text


def getPolicyTransactionPremium(auth_token,policy_transaction_premium_id = "Premium12345"):
    url = 'https://affwsapi.ams360.com/v2/service.asmx'
    # SOAP Request Body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetPolicyTransactionPremium_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <PolicyTransactionPremiumId>{policy_transaction_premium_id}</PolicyTransactionPremiumId>
        </GetPolicyTransactionPremium_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Debugging: print the full SOAP request to check the structure
    print("SOAP Body:")
    print(soap_body)

    # Headers (Note that Token should be a string and passed as plain text)
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/GetPolicyTransactionPremium',
        'Authorization': f"Bearer {auth_token}"  # Use Authorization header for the token
    }

    # Send the SOAP request using requests.post
    response = requests.post(url, data=soap_body, headers=headers)

    # Check the response
    if response.status_code == 200:
        print("Response Data:", response.text)
        return response.text
    else:
        print(f"Error: {response.status_code}")
        print(f"Response Body: {response.text}")


def getPolicyTransactionPremiumsForPolicy(auth_token,policy_id = "Policy12345",transaction_effective_date = "2024-05-15",filter_include_corrected = "false"):

    url = 'https://affwsapi.ams360.com/v2/service.asmx'

    # SOAP Request Body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetPolicyTransactionPremiumsForPolicy_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <PolicyId>{policy_id}</PolicyId>
          <TransactionEffectiveDate>{transaction_effective_date}</TransactionEffectiveDate>
          <FilterIncludeCorrected>{filter_include_corrected}</FilterIncludeCorrected>
        </GetPolicyTransactionPremiumsForPolicy_Request>
      </soap12:Body>
    </soap12:Envelope>"""
    # Debugging: print the full SOAP request to check the structure
    print("SOAP Body:")
    print(soap_body)

    # Headers (Note that Token should be a string and passed as plain text)
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/GetPolicyTransactionPremiumsForPolicy',
        'Authorization': f"Bearer {auth_token}"  # Use Authorization header for the token
    }

    # Send the SOAP request using requests.post
    response = requests.post(url, data=soap_body, headers=headers)

    # Check the response
    if response.status_code == 200:
        print("Response Data:", response.text)
        return response.text
    else:
        print(f"Error: {response.status_code}")
        print(f"Response Body: {response.text}")


def getRemark(auth_token,policy_id = "Policy12345",line_of_business_id = "LOB12345",remark_type = 1,transaction_effective_date = "2024-05-15T00:00:00",parent_id = "Parent12345"):
    url = 'https://affwsapi.ams360.com/v2/service.asmx'
    # SOAP Request Body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetRemark_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <PolicyId>{policy_id}</PolicyId>
          <LineOfBusinessId>{line_of_business_id}</LineOfBusinessId>
          <RemarkType>{remark_type}</RemarkType>
          <TransactionEffectiveDate>{transaction_effective_date}</TransactionEffectiveDate>
          <ParentId>{parent_id}</ParentId>
        </GetRemark_Request>
      </soap12:Body>
    </soap12:Envelope>"""
    # Debugging: print the full SOAP request to check the structure
    print("SOAP Body:")
    print(soap_body)

    # Headers (Note that Token should be a string and passed as plain text)
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/GetRemark',
        'Authorization': f"Bearer {auth_token}"  # Use Authorization header for the token
    }

    # Send the SOAP request using requests.post
    response = requests.post(url, data=soap_body, headers=headers)

    # Check the response
    if response.status_code == 200:
        print("Response Data:", response.text)
        return response.text
    else:
        print(f"Error: {response.status_code}")
        print(f"Response Body: {response.text}")

def getSuspense(auth_token,suspense_id = "Suspense12345"):

    # SOAP Request Body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetSuspense_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <SuspenseId>{suspense_id}</SuspenseId>
        </GetSuspense_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Request headers
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/GetSuspense',
        'Authorization': f"Bearer {auth_token}"  # Use Authorization header for the token
    }

    # Endpoint URL
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Send the SOAP request
    response = requests.post(url, data=soap_body, headers=headers)

    # Output the response
    if response.status_code == 200:
        print("Response Data:", response.text)
        return response.text
    else:
        print(f"Error: {response.status_code}")
        print(f"Response Body: {response.text}")

def getSuspenseList(auth_token,attached_to = "Attachment123",attached_to_type = 1,assigned_to_emp_code = "EmpCode456"):

    # SOAP Request Body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetSuspenseList_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <AttachedTo>{attached_to}</AttachedTo>
          <AttachedToType>{attached_to_type}</AttachedToType>
          <AssignedToEmpCode>{assigned_to_emp_code}</AssignedToEmpCode>
        </GetSuspenseList_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Request headers
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/GetSuspenseList',
        'Authorization': f"Bearer {auth_token}"  # Use Authorization header for the token
    }

    # Endpoint URL
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Send the SOAP request
    response = requests.post(url, data=soap_body, headers=headers)

    # Output the response
    if response.status_code == 200:
        print("Response Data:", response.text)
        return response.text
    else:
        print(f"Error: {response.status_code}")
        print(f"Response Body: {response.text}")

def getValueList(auth_token,list_name = "YourListName"):

    # SOAP Request Body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetValueList_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <ListName>{list_name}</ListName>
        </GetValueList_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Request headers
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/GetValueList',
        'Authorization': f"Bearer {auth_token}"  # Use Authorization header for the token
    }

    # Endpoint URL
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Send the SOAP request
    response = requests.post(url, data=soap_body, headers=headers)

    # Output the response
    if response.status_code == 200:
        print("Response Data:", response.text)
        return response.text
    else:
        print(f"Error: {response.status_code}")
        print(f"Response Body: {response.text}")

def getVersion(auth_token):
    # SOAP Request Body with an empty <soap12:Body>
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body />
    </soap12:Envelope>"""

    # Request headers
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': '"http://www.WSAPI.AMS360.com/v2.0/GetVersion"',
        'Authorization': f"Bearer {auth_token}"  # Use Authorization header for the token
    }

    # Endpoint URL
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Send the SOAP request
    response = requests.post(url, data=soap_body, headers=headers)

    # Output the response
    if response.status_code == 200:
        print("Response Data:", response.text)
        return response.text
    else:
        print(f"Error: {response.status_code}")
        print(f"Response Body: {response.text}")


def insertAL3Policy(auth_token,policy_number = "",customer_id = "",bill_method = "",
                    policy_effective_date = "",policy_expiration_date = "",transaction_effective_date = "",company_code = "",transaction_type = "",transaction_description = "",file_contents = ""
):

    # SOAP Request Body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <InsertAL3Policy_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <PolicyNumber>{policy_number}</PolicyNumber>
          <CustomerId>{customer_id}</CustomerId>
          <BillMethod>{bill_method}</BillMethod>
          <PolicyEffectiveDate>{policy_effective_date}</PolicyEffectiveDate>
          <PolicyExpirationDate>{policy_expiration_date}</PolicyExpirationDate>
          <TransactionEffectiveDate>{transaction_effective_date}</TransactionEffectiveDate>
          <CompanyCode>{company_code}</CompanyCode>
          <TransactionType>{transaction_type}</TransactionType>
          <TransactionDescription>{transaction_description}</TransactionDescription>
          <FileContents>{file_contents}</FileContents>
          <AdditionalInformation>
            <CustomerNotation></CustomerNotation>
            <PolicyNotation></PolicyNotation>
            <PolicyInsured>
              <MobilePhoneAreaCode></MobilePhoneAreaCode>
              <MobilePhoneNumber></MobilePhoneNumber>
              <MobilePhoneExt></MobilePhoneExt>
              <Email></Email>
            </PolicyInsured>
            <PolicyCoInsured>
              <SSN></SSN>
              <DOB></DOB>
              <FirstName></FirstName>
              <LastName></LastName>
              <MiddleName></MiddleName>
              <ResidencePhoneAreaCode></ResidencePhoneAreaCode>
              <ResidencePhoneNumber></ResidencePhoneNumber>
              <ResidencePhoneExt></ResidencePhoneExt>
              <MobilePhoneAreaCode></MobilePhoneAreaCode>
              <MobilePhoneNumber></MobilePhoneNumber>
              <MobilePhoneExt></MobilePhoneExt>
              <Email></Email>
            </PolicyCoInsured>
            <PriorCarrierList />
            <DriverList />
            <VehicleList />
            <UsageList />
          </AdditionalInformation>
        </InsertAL3Policy_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Request headers
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': '"http://www.WSAPI.AMS360.com/v2.0/InsertAL3Policy"',
        'Authorization': f"Bearer {auth_token}"  # Use Authorization header for the token
    }

    # Endpoint URL
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Send the SOAP request
    response = requests.post(url, data=soap_body, headers=headers)

    # Output the response
    if response.status_code == 200:
        print("Response Data:", response.text)
        return response.text
    else:
        print(f"Error: {response.status_code}")
        print(f"Response Body: {response.text}")


def insertActivity(auth_token,activity_id="",assigned_to="",activity_type="",
    policy_id = "",company_code = "",activity_action = "",activity_date = "",
    activity_time = "",employee_code = "",claim_id = "",description = "",doc360_document_id = "",
    doc_description = "",storage_location = "",document_type = "",security_class = "",index1 = "",
    index2 = "",comments = "",source_file_name = "",received_date_time = "",processed_date_time = "",
    total_bytes = ""):

    # SOAP Request Body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <InsertActivity_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Activity>
            <ActivityId>{activity_id}</ActivityId>
            <AssignedTo>{assigned_to}</AssignedTo>
            <ActivityType>{activity_type}</ActivityType>
            <PolicyId>{policy_id}</PolicyId>
            <CompanyCode>{company_code}</CompanyCode>
            <ActivityAction>{activity_action}</ActivityAction>
            <ActivityDate>{activity_date}</ActivityDate>
            <ActivityTime>{activity_time}</ActivityTime>
            <EmployeeCode>{employee_code}</EmployeeCode>
            <ClaimId>{claim_id}</ClaimId>
            <Description>{description}</Description>
          </Activity>
          <Attachments>
            <Doc360Document>
              <Doc360DocumentId>{doc360_document_id}</Doc360DocumentId>
              <Description>{doc_description}</Description>
              <StorageLocation>{storage_location}</StorageLocation>
              <DocumentType>{document_type}</DocumentType>
              <SecurityClass>{security_class}</SecurityClass>
              <Index1>{index1}</Index1>
              <Index2>{index2}</Index2>
              <Comments>{comments}</Comments>
              <SourceFileName>{source_file_name}</SourceFileName>
              <ReceivedDateTime>{received_date_time}</ReceivedDateTime>
              <ProcessedDateTime>{processed_date_time}</ProcessedDateTime>
              <TotalBytes>{total_bytes}</TotalBytes>
            </Doc360Document>
          </Attachments>
        </InsertActivity_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Request headers
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': '"http://www.WSAPI.AMS360.com/v2.0/InsertActivity"',
        'Authorization': f"Bearer {auth_token}"  # Use Authorization header for the token
    }

    # Endpoint URL
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Send the SOAP request
    response = requests.post(url, data=soap_body, headers=headers)

    # Output the response
    if response.status_code == 200:
        print("Response Data:", response.text)
        return response.text
    else:
        print(f"Error: {response.status_code}")
        print(f"Response Body: {response.text}")


def insertCustomer(auth_token,customer_data):
    # SOAP Request Body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
        <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
            <Token>{auth_token}</Token>
        </WSAPIAuthToken>
          </soap12:Header>
          <soap12:Body>
            <InsertCustomer_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
              <Customer>
                {''.join([f'<{key}>{value}</{key}>' for key, value in customer_data.items()])}
              </Customer>
            </InsertCustomer_Request>
        </soap12:Body>
    </soap12:Envelope>"""


    # Request headers
    headers = {
            'Content-Type': 'application/soap+xml; charset=utf-8',
            'SOAPAction': '"http://www.WSAPI.AMS360.com/v2.0/InsertCustomer"',
            'Authorization': f"Bearer {auth_token}"  # Use Authorization header for the token
    }

    # Endpoint URL
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Send the SOAP request
    response = requests.post(url, data=soap_body, headers=headers)

    # Output the response
    if response.status_code == 200:
        print("Response Data:", response.text)
        return response.text
    else:
        print(f"Error: {response.status_code}")
        print(f"Response Body: {response.text}")


def insertInvoice(auth_token,policy_id = "POL123456",bill_method = "Direct Bill",pay_plan_id = "PAY001",
                    premium_to_bill = 500.75,installment_day = 15,transaction_effective_date = "2024-11-20",
                    invoice_date = "2024-11-21",due_date = "2024-12-01"):
    # Endpoint URL
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Generate SOAP Body dynamically
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
        <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
          <soap12:Header>
            <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
              <Token>{auth_token}</Token>
            </WSAPIAuthToken>
          </soap12:Header>
          <soap12:Body>
            <InsertInvoice_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
              <Invoice>
                <PolicyId>{policy_id}</PolicyId>
                <BillMethod>{bill_method}</BillMethod>
                <PayPlanId>{pay_plan_id}</PayPlanId>
                <PremiumToBillOnEffectiveDate>{premium_to_bill}</PremiumToBillOnEffectiveDate>
                <InstallmentDayOfMonth>{installment_day}</InstallmentDayOfMonth>
                <TransactionEffectiveDate>{transaction_effective_date}</TransactionEffectiveDate>
                <InvoiceDate>{invoice_date}</InvoiceDate>
                <DueDate>{due_date}</DueDate>
              </Invoice>
            </InsertInvoice_Request>
          </soap12:Body>
        </soap12:Envelope>"""

    # Debugging: print the full SOAP request to check the structure
    print("SOAP Body:")
    print(soap_body)

    # Headers (Note that Token should be a string and passed as plain text)
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://www.WSAPI.AMS360.com/v2.0/InsertInvoice',
        'Authorization': f"Bearer {auth_token}"  # Use Authorization header for the token
    }

    # Send the SOAP request using requests.post
    response = requests.post(url, data=soap_body, headers=headers)

    # Check the response
    if response.status_code == 200:
        print("Response Data:", response.text)
    else:
        print(f"Error: {response.status_code}")
        print(f"Response Body: {response.text}")


def insertNote(auth_token, note_data):
        """
        Sends an InsertNote_Request SOAP request.

        Parameters:
            auth_token (str): Bearer token for authorization.
            note_data (dict): Dictionary containing the note details.

        Returns:
            response: HTTP response from the SOAP service.
        """
        # Constructing SOAP body
        soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
        <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.W3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
          <soap12:Header>
            <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
              <Token>{auth_token}</Token>
            </WSAPIAuthToken>
          </soap12:Header>
          <soap12:Body>
            <InsertNote_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
              <Note>
                <NoteId>{note_data["NoteId"]}</NoteId>
                <NoteText>{note_data["NoteText"]}</NoteText>
                <IsSticky>{note_data["IsSticky"]}</IsSticky>
                <NoteDate>{note_data["NoteDate"]}</NoteDate>
                <PurgeDate>{note_data["PurgeDate"]}</PurgeDate>
                <NoteType>{note_data["NoteType"]}</NoteType>
                <Priority>{note_data["Priority"]}</Priority>
                <AttachedTo>{note_data["AttachedTo"]}</AttachedTo>
                <PolicyId>{note_data["PolicyId"]}</PolicyId>
                <ClaimId>{note_data["ClaimId"]}</ClaimId>
              </Note>
            </InsertNote_Request>
          </soap12:Body>
        </soap12:Envelope>
        """

        # Headers with Authorization and SOAPAction
        headers = {
            "Content-Type": "application/soap+xml; charset=utf-8",
            "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/InsertNote",
            "Authorization": f"Bearer {auth_token}"
        }

        # Sending the SOAP request
        url = "https://affwsapi.ams360.com/v2/service.asmx"
        response = requests.post(url, data=soap_body, headers=headers)

        # Output response
        if response.status_code == 200:
            print("SOAP request successful.")
            print("Response:", response.text)
            return response.text
        else:
            print(f"Error: {response.status_code}")
            print("Response:", response.text)


def insert_policy_transaction_fee(auth_token, transaction_fee_details):
    """
    Sends a SOAP request to insert a policy transaction fee.

    Parameters:
        auth_token (str): Bearer token for authorization.
        transaction_fee_details (dict): Dictionary containing transaction fee details.

    Returns:
        Response: Response object from the server.
    """
    # Extract details from input dictionary
    policy_id = transaction_fee_details.get("PolicyId", "")
    transaction_effective_date = transaction_fee_details.get("TransactionEffectiveDate", "")
    policy_transaction_fee_id = transaction_fee_details.get("PolicyTransactionFeeId", "")
    charge_type = transaction_fee_details.get("ChargeType", "")
    description = transaction_fee_details.get("Description", "")
    company_code = transaction_fee_details.get("CompanyCode", "")
    amount = transaction_fee_details.get("Amount", 0.0)
    how_billed = transaction_fee_details.get("HowBilled", "")
    include_premium = transaction_fee_details.get("IncludePremium", False)
    reconciled = transaction_fee_details.get("Reconciled", "")

    # Construct SOAP body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.W3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <InsertPolicyTransactionFee_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <PolicyTransactionFee>
            <PolicyId>{policy_id}</PolicyId>
            <TransactionEffectiveDate>{transaction_effective_date}</TransactionEffectiveDate>
            <PolicyTransactionFeeId>{policy_transaction_fee_id}</PolicyTransactionFeeId>
            <ChargeType>{charge_type}</ChargeType>
            <Description>{description}</Description>
            <CompanyCode>{company_code}</CompanyCode>
            <Amount>{amount}</Amount>
            <HowBilled>{how_billed}</HowBilled>
            <IncludePremium>{str(include_premium).lower()}</IncludePremium>
            <Reconciled>{reconciled}</Reconciled>
          </PolicyTransactionFee>
        </InsertPolicyTransactionFee_Request>
      </soap12:Body>
    </soap12:Envelope>
    """

    # Set headers
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/InsertPolicyTransactionFee",
        "Authorization": f"Bearer {auth_token}",
    }

    # URL for SOAP service
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Send POST request
    response = requests.post(url, data=soap_body, headers=headers)

    # Print the response
    if response.status_code == 200:
        print("SOAP request was successful.")
        print(response.text)
        return response.text
    else:
        print(f"Failed with status code: {response.status_code}")
        print(response.text)

# Example input format for InsertPolicyTransactionFee

transaction_fee_details = {
    "PolicyId": "POL12345",
    "TransactionEffectiveDate": "2024-01-01",
    "PolicyTransactionFeeId": "FEE12345",
    "ChargeType": "Fee",
    "Description": "Transaction fee description",
    "CompanyCode": "COMP123",
    "Amount": 100.50,
    "HowBilled": "Monthly",
    "IncludePremium": True,
    "Reconciled": "No",
}


def insertPolicyTransactionFees(auth_token, policy_id, transaction_fee_list):
    """
    Sends a SOAP request to insert multiple policy transaction fees.

    Parameters:
        auth_token (str): Bearer token for authorization.
        policy_id (str): ID of the policy.
        transaction_fee_list (list): List of dictionaries containing transaction fee details.

    Returns:
        Response: Response object from the server.
    """
    # Construct PolicyTransactionFee XML from the list of transaction fees
    policy_transaction_fees_xml = ""
    for fee in transaction_fee_list:
        policy_transaction_fees_xml += f"""
        <PolicyTransactionFee>
          <PolicyId>{fee.get("PolicyId", "")}</PolicyId>
          <TransactionEffectiveDate>{fee.get("TransactionEffectiveDate", "")}</TransactionEffectiveDate>
          <PolicyTransactionFeeId>{fee.get("PolicyTransactionFeeId", "")}</PolicyTransactionFeeId>
          <ChargeType>{fee.get("ChargeType", "")}</ChargeType>
          <Description>{fee.get("Description", "")}</Description>
          <CompanyCode>{fee.get("CompanyCode", "")}</CompanyCode>
          <Amount>{fee.get("Amount", 0.0)}</Amount>
          <HowBilled>{fee.get("HowBilled", "")}</HowBilled>
          <IncludePremium>{str(fee.get("IncludePremium", False)).lower()}</IncludePremium>
          <Reconciled>{fee.get("Reconciled", "")}</Reconciled>
        </PolicyTransactionFee>
        """

    # Construct SOAP body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <InsertPolicyTransactionFees_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <PolicyId>{policy_id}</PolicyId>
          <PolicyTransactionFeeList>
            {policy_transaction_fees_xml}
          </PolicyTransactionFeeList>
        </InsertPolicyTransactionFees_Request>
      </soap12:Body>
    </soap12:Envelope>
    """

    # Set headers
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/InsertPolicyTransactionFees",
        "Authorization": f"Bearer {auth_token}",
    }

    # URL for SOAP service
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Send POST request
    response = requests.post(url, data=soap_body, headers=headers)

    # Print the response
    if response.status_code == 200:
        print("SOAP request was successful.")
        print(response.text)
        return response.text
    else:
        print(f"Failed with status code: {response.status_code}")
        print(response.text)


# Example input for insertPolicyTransactionFees
policy_id = "POL12345"

transaction_fee_list = [
    {
        "PolicyId": "POL12345",
        "TransactionEffectiveDate": "2024-01-01",
        "PolicyTransactionFeeId": "FEE001",
        "ChargeType": "Fee",
        "Description": "Transaction fee 1",
        "CompanyCode": "COMP001",
        "Amount": 100.50,
        "HowBilled": "Monthly",
        "IncludePremium": True,
        "Reconciled": "No",
    },
    {
        "PolicyId": "POL12345",
        "TransactionEffectiveDate": "2024-02-01",
        "PolicyTransactionFeeId": "FEE002",
        "ChargeType": "Fee",
        "Description": "Transaction fee 2",
        "CompanyCode": "COMP002",
        "Amount": 200.75,
        "HowBilled": "Quarterly",
        "IncludePremium": False,
        "Reconciled": "Yes",
    },
]


def insert_policy_transaction_premium(auth_token, transaction_premium_details):
    """
    Sends a SOAP request to insert a policy transaction premium.

    Parameters:
        auth_token (str): Bearer token for authorization.
        transaction_premium_details (dict): Dictionary containing details of the transaction premium.

    Returns:
        Response: Response object from the server.
    """
    # Construct SOAP body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <InsertPolicyTransactionPremium_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <PolicyTransactionPremium>
            <PolicyId>{transaction_premium_details.get("PolicyId", "")}</PolicyId>
            <TransactionEffectiveDate>{transaction_premium_details.get("TransactionEffectiveDate", "")}</TransactionEffectiveDate>
            <PolicyTransactionPremiumId>{transaction_premium_details.get("PolicyTransactionPremiumId", "")}</PolicyTransactionPremiumId>
            <LineOfBusiness>{transaction_premium_details.get("LineOfBusiness", "")}</LineOfBusiness>
            <PlanType>{transaction_premium_details.get("PlanType", "")}</PlanType>
            <WritingCompanyCode>{transaction_premium_details.get("WritingCompanyCode", "")}</WritingCompanyCode>
            <Description>{transaction_premium_details.get("Description", "")}</Description>
            <Premium>{transaction_premium_details.get("Premium", 0.0)}</Premium>
            <BilledPremium>{transaction_premium_details.get("BilledPremium", 0.0)}</BilledPremium>
            <WrittenPremium>{transaction_premium_details.get("WrittenPremium", 0.0)}</WrittenPremium>
            <FullTermPremium>{transaction_premium_details.get("FullTermPremium", 0.0)}</FullTermPremium>
            <EstRevenue>{transaction_premium_details.get("EstRevenue", 0.0)}</EstRevenue>
            <IncludePremium>{str(transaction_premium_details.get("IncludePremium", False)).lower()}</IncludePremium>
            <HowBilled>{transaction_premium_details.get("HowBilled", "")}</HowBilled>
            <IsBilled>{str(transaction_premium_details.get("IsBilled", False)).lower()}</IsBilled>
            <IsSuspended>{str(transaction_premium_details.get("IsSuspended", False)).lower()}</IsSuspended>
            <IsCorrected>{str(transaction_premium_details.get("IsCorrected", False)).lower()}</IsCorrected>
            <Reconciled>{transaction_premium_details.get("Reconciled", "")}</Reconciled>
          </PolicyTransactionPremium>
        </InsertPolicyTransactionPremium_Request>
      </soap12:Body>
    </soap12:Envelope>
    """

    # Set headers
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/InsertPolicyTransactionPremium",
        "Authorization": f"Bearer {auth_token}",
    }

    # URL for SOAP service
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Send POST request
    response = requests.post(url, data=soap_body, headers=headers)

    # Print the response
    if response.status_code == 200:
        print("SOAP request was successful.")
        print(response.text)
        return response.text
    else:
        print(f"Failed with status code: {response.status_code}")
        print(response.text)

    return response


# Example input for InsertPolicyTransactionPremium

transaction_premium_details = {
    "PolicyId": "POL12345",
    "TransactionEffectiveDate": "2024-01-01",
    "PolicyTransactionPremiumId": "PREM001",
    "LineOfBusiness": "Auto",
    "PlanType": "Comprehensive",
    "WritingCompanyCode": "COMP001",
    "Description": "Transaction premium details",
    "Premium": 500.00,
    "BilledPremium": 500.00,
    "WrittenPremium": 500.00,
    "FullTermPremium": 1000.00,
    "EstRevenue": 50.00,
    "IncludePremium": True,
    "HowBilled": "Monthly",
    "IsBilled": True,
    "IsSuspended": False,
    "IsCorrected": False,
    "Reconciled": "No",
}


def insert_policy_transaction_premiums(auth_token, policy_id, premium_list):
    """
    Sends a SOAP request to insert multiple policy transaction premiums.

    Parameters:
        auth_token (str): Authorization token for the request.
        policy_id (str): Policy ID for the premiums.
        premium_list (list[dict]): List of premium details dictionaries.

    Returns:
        Response: The HTTP response from the SOAP request.
    """
    # Constructing the PolicyTransactionPremiumList XML
    premium_items = ""
    for premium in premium_list:
        premium_items += f"""
        <PolicyTransactionPremium>
          <PolicyId>{premium.get("PolicyId", "")}</PolicyId>
          <TransactionEffectiveDate>{premium.get("TransactionEffectiveDate", "")}</TransactionEffectiveDate>
          <PolicyTransactionPremiumId>{premium.get("PolicyTransactionPremiumId", "")}</PolicyTransactionPremiumId>
          <LineOfBusiness>{premium.get("LineOfBusiness", "")}</LineOfBusiness>
          <PlanType>{premium.get("PlanType", "")}</PlanType>
          <WritingCompanyCode>{premium.get("WritingCompanyCode", "")}</WritingCompanyCode>
          <Description>{premium.get("Description", "")}</Description>
          <Premium>{premium.get("Premium", 0.0)}</Premium>
          <BilledPremium>{premium.get("BilledPremium", 0.0)}</BilledPremium>
          <WrittenPremium>{premium.get("WrittenPremium", 0.0)}</WrittenPremium>
          <FullTermPremium>{premium.get("FullTermPremium", 0.0)}</FullTermPremium>
          <EstRevenue>{premium.get("EstRevenue", 0.0)}</EstRevenue>
          <IncludePremium>{str(premium.get("IncludePremium", False)).lower()}</IncludePremium>
          <HowBilled>{premium.get("HowBilled", "")}</HowBilled>
          <IsBilled>{str(premium.get("IsBilled", False)).lower()}</IsBilled>
          <IsSuspended>{str(premium.get("IsSuspended", False)).lower()}</IsSuspended>
          <IsCorrected>{str(premium.get("IsCorrected", False)).lower()}</IsCorrected>
          <Reconciled>{premium.get("Reconciled", "")}</Reconciled>
        </PolicyTransactionPremium>
        """

    # SOAP Body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <InsertPolicyTransactionPremiums_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <PolicyId>{policy_id}</PolicyId>
          <PolicyTransactionPremiumList>
            {premium_items}
          </PolicyTransactionPremiumList>
        </InsertPolicyTransactionPremiums_Request>
      </soap12:Body>
    </soap12:Envelope>
    """

    # Headers
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/InsertPolicyTransactionPremiums",
        "Authorization": f"Bearer {auth_token}",
    }

    # URL
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Sending POST Request
    response = requests.post(url, data=soap_body, headers=headers)

    # Print the response
    if response.status_code == 200:
        print("SOAP request was successful.")
        print(response.text)
        return response.text
    else:
        print(f"Failed with status code: {response.status_code}")
        print(response.text)

    return response

# Example Input for InsertPolicyTransactionPremiums
policy_id = "POL12345"
premium_list = [
    {
        "PolicyId": "POL12345",
        "TransactionEffectiveDate": "2024-01-01",
        "PolicyTransactionPremiumId": "PREM001",
        "LineOfBusiness": "Auto",
        "PlanType": "Comprehensive",
        "WritingCompanyCode": "COMP001",
        "Description": "First premium",
        "Premium": 500.00,
        "BilledPremium": 500.00,
        "WrittenPremium": 500.00,
        "FullTermPremium": 1000.00,
        "EstRevenue": 50.00,
        "IncludePremium": True,
        "HowBilled": "Monthly",
        "IsBilled": True,
        "IsSuspended": False,
        "IsCorrected": False,
        "Reconciled": "No",
    },
    {
        "PolicyId": "POL12345",
        "TransactionEffectiveDate": "2024-02-01",
        "PolicyTransactionPremiumId": "PREM002",
        "LineOfBusiness": "Home",
        "PlanType": "Basic",
        "WritingCompanyCode": "COMP002",
        "Description": "Second premium",
        "Premium": 300.00,
        "BilledPremium": 300.00,
        "WrittenPremium": 300.00,
        "FullTermPremium": 600.00,
        "EstRevenue": 30.00,
        "IncludePremium": False,
        "HowBilled": "Quarterly",
        "IsBilled": False,
        "IsSuspended": False,
        "IsCorrected": True,
        "Reconciled": "Yes",
    },
]


def insert_receipt(auth_token, receipt_data, insured_accounts, direct_bill_deposits):
    """
    Sends a SOAP request to insert a receipt with insured accounts receivables and direct bill deposits.

    Parameters:
        auth_token (str): Authorization token for the request.
        receipt_data (dict): Dictionary containing receipt information.
        insured_accounts (list[dict]): List of insured accounts receivable details.
        direct_bill_deposits (list[dict]): List of direct bill deposit details.

    Returns:
        Response: The HTTP response from the SOAP request.
    """
    # Constructing the InsuredAccountsReceivableList XML
    insured_accounts_xml = ""
    for account in insured_accounts:
        insured_accounts_xml += f"""
        <InsuredAccountsReceivable>
          <CustomerId>{account.get("CustomerId", "")}</CustomerId>
          <PolicyId>{account.get("PolicyId", "")}</PolicyId>
          <InvoiceId>{account.get("InvoiceId", "")}</InvoiceId>
          <Amount>{account.get("Amount", 0.0)}</Amount>
          <Description>{account.get("Description", "")}</Description>
          <RowType>{account.get("RowType", "")}</RowType>
          <GLDivisionCode>{account.get("GLDivisionCode", "")}</GLDivisionCode>
          <GLBranchCode>{account.get("GLBranchCode", "")}</GLBranchCode>
          <GLDepartmentCode>{account.get("GLDepartmentCode", "")}</GLDepartmentCode>
          <GLGroupCode>{account.get("GLGroupCode", "")}</GLGroupCode>
        </InsuredAccountsReceivable>
        """

    # Constructing the DirectBillDepositList XML
    direct_bill_deposits_xml = ""
    for deposit in direct_bill_deposits:
        direct_bill_deposits_xml += f"""
        <DirectBillDeposit>
          <Amount>{deposit.get("Amount", 0.0)}</Amount>
          <Description>{deposit.get("Description", "")}</Description>
          <InsuranceCompanyCode>{deposit.get("InsuranceCompanyCode", "")}</InsuranceCompanyCode>
          <FinanceCompanyCode>{deposit.get("FinanceCompanyCode", "")}</FinanceCompanyCode>
          <BrokerageCompanyCode>{deposit.get("BrokerageCompanyCode", "")}</BrokerageCompanyCode>
        </DirectBillDeposit>
        """

    # SOAP Body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <InsertReceipt_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Receipt>
            <CustomerId>{receipt_data.get("CustomerId", "")}</CustomerId>
            <ReceiptDate>{receipt_data.get("ReceiptDate", "")}</ReceiptDate>
            <Bank>{receipt_data.get("Bank", "")}</Bank>
            <BankGLDivision>{receipt_data.get("BankGLDivision", "")}</BankGLDivision>
            <ReceivedFrom>{receipt_data.get("ReceivedFrom", "")}</ReceivedFrom>
            <Name>{receipt_data.get("Name", "")}</Name>
            <Description>{receipt_data.get("Description", "")}</Description>
            <PaymentType>{receipt_data.get("PaymentType", "")}</PaymentType>
            <BankNumber>{receipt_data.get("BankNumber", "")}</BankNumber>
            <CheckNumber>{receipt_data.get("CheckNumber", "")}</CheckNumber>
            <Amount>{receipt_data.get("Amount", 0.0)}</Amount>
            <CreditCardAuthorizationNumber>{receipt_data.get("CreditCardAuthorizationNumber", "")}</CreditCardAuthorizationNumber>
          </Receipt>
          <InsuredAccountsReceivableList>
            {insured_accounts_xml}
          </InsuredAccountsReceivableList>
          <DirectBillDepositList>
            {direct_bill_deposits_xml}
          </DirectBillDepositList>
        </InsertReceipt_Request>
      </soap12:Body>
    </soap12:Envelope>
    """

    # Headers
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/InsertReceipt",
        "Authorization": f"Bearer {auth_token}",
    }

    # URL
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Sending POST Request
    response = requests.post(url, data=soap_body, headers=headers)

    # Print the response
    if response.status_code == 200:
        print("SOAP request was successful.")
        print(response.text)
        return response.text
    else:
        print(f"Failed with status code: {response.status_code}")
        print(response.text)

    return response


# Example Input for Insert Receipt
auth_token = "your_auth_token_here"
receipt_data = {
    "CustomerId": "CUST001",
    "ReceiptDate": "2024-11-20",
    "Bank": "Main Bank",
    "BankGLDivision": "DIV001",
    "ReceivedFrom": "John Doe",
    "Name": "Policy Payment",
    "Description": "Monthly policy premium payment",
    "PaymentType": "Check",
    "BankNumber": "123456",
    "CheckNumber": "987654",
    "Amount": 1500.00,
    "CreditCardAuthorizationNumber": "",
}

insured_accounts = [
    {
        "CustomerId": "CUST001",
        "PolicyId": "POL12345",
        "InvoiceId": "INV001",
        "Amount": 500.00,
        "Description": "First payment",
        "RowType": "AR",
        "GLDivisionCode": "DIV001",
        "GLBranchCode": "BR001",
        "GLDepartmentCode": "DEP001",
        "GLGroupCode": "GRP001",
    },
    {
        "CustomerId": "CUST001",
        "PolicyId": "POL12346",
        "InvoiceId": "INV002",
        "Amount": 1000.00,
        "Description": "Second payment",
        "RowType": "AR",
        "GLDivisionCode": "DIV001",
        "GLBranchCode": "BR002",
        "GLDepartmentCode": "DEP002",
        "GLGroupCode": "GRP002",
    },
]

direct_bill_deposits = [
    {
        "Amount": 300.00,
        "Description": "Direct bill from insurer",
        "InsuranceCompanyCode": "INS001",
        "FinanceCompanyCode": "FIN001",
        "BrokerageCompanyCode": "BRO001",
    },
    {
        "Amount": 200.00,
        "Description": "Additional bill",
        "InsuranceCompanyCode": "INS002",
        "FinanceCompanyCode": "FIN002",
        "BrokerageCompanyCode": "BRO002",
    },
]

def insert_receipt(auth_token, receipt_data, insured_accounts, direct_bill_deposits):
    """
    Sends a SOAP request to insert a receipt with insured accounts receivables and direct bill deposits.

    Parameters:
        auth_token (str): Authorization token for the request.
        receipt_data (dict): Dictionary containing receipt information.
        insured_accounts (list[dict]): List of insured accounts receivable details.
        direct_bill_deposits (list[dict]): List of direct bill deposit details.

    Returns:
        Response: The HTTP response from the SOAP request.
    """
    # Constructing the InsuredAccountsReceivableList XML
    insured_accounts_xml = ""
    for account in insured_accounts:
        insured_accounts_xml += f"""
        <InsuredAccountsReceivable>
          <CustomerId>{account.get("CustomerId", "")}</CustomerId>
          <PolicyId>{account.get("PolicyId", "")}</PolicyId>
          <InvoiceId>{account.get("InvoiceId", "")}</InvoiceId>
          <Amount>{account.get("Amount", 0.0)}</Amount>
          <Description>{account.get("Description", "")}</Description>
          <RowType>{account.get("RowType", "")}</RowType>
          <GLDivisionCode>{account.get("GLDivisionCode", "")}</GLDivisionCode>
          <GLBranchCode>{account.get("GLBranchCode", "")}</GLBranchCode>
          <GLDepartmentCode>{account.get("GLDepartmentCode", "")}</GLDepartmentCode>
          <GLGroupCode>{account.get("GLGroupCode", "")}</GLGroupCode>
        </InsuredAccountsReceivable>
        """

    # Constructing the DirectBillDepositList XML
    direct_bill_deposits_xml = ""
    for deposit in direct_bill_deposits:
        direct_bill_deposits_xml += f"""
        <DirectBillDeposit>
          <Amount>{deposit.get("Amount", 0.0)}</Amount>
          <Description>{deposit.get("Description", "")}</Description>
          <InsuranceCompanyCode>{deposit.get("InsuranceCompanyCode", "")}</InsuranceCompanyCode>
          <FinanceCompanyCode>{deposit.get("FinanceCompanyCode", "")}</FinanceCompanyCode>
          <BrokerageCompanyCode>{deposit.get("BrokerageCompanyCode", "")}</BrokerageCompanyCode>
        </DirectBillDeposit>
        """

    # SOAP Body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <InsertReceipt_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Receipt>
            <CustomerId>{receipt_data.get("CustomerId", "")}</CustomerId>
            <ReceiptDate>{receipt_data.get("ReceiptDate", "")}</ReceiptDate>
            <Bank>{receipt_data.get("Bank", "")}</Bank>
            <BankGLDivision>{receipt_data.get("BankGLDivision", "")}</BankGLDivision>
            <ReceivedFrom>{receipt_data.get("ReceivedFrom", "")}</ReceivedFrom>
            <Name>{receipt_data.get("Name", "")}</Name>
            <Description>{receipt_data.get("Description", "")}</Description>
            <PaymentType>{receipt_data.get("PaymentType", "")}</PaymentType>
            <BankNumber>{receipt_data.get("BankNumber", "")}</BankNumber>
            <CheckNumber>{receipt_data.get("CheckNumber", "")}</CheckNumber>
            <Amount>{receipt_data.get("Amount", 0.0)}</Amount>
            <CreditCardAuthorizationNumber>{receipt_data.get("CreditCardAuthorizationNumber", "")}</CreditCardAuthorizationNumber>
          </Receipt>
          <InsuredAccountsReceivableList>
            {insured_accounts_xml}
          </InsuredAccountsReceivableList>
          <DirectBillDepositList>
            {direct_bill_deposits_xml}
          </DirectBillDepositList>
        </InsertReceipt_Request>
      </soap12:Body>
    </soap12:Envelope>
    """

    # Headers
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/InsertReceipt",
        "Authorization": f"Bearer {auth_token}",
    }

    # URL
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Sending POST Request
    response = requests.post(url, data=soap_body, headers=headers)

    # Print the response
    if response.status_code == 200:
        print("SOAP request was successful.")
        print(response.text)
        return response.text
    else:
        print(f"Failed with status code: {response.status_code}")
        print(response.text)

# Example Input for Insert Receipt
auth_token = "your_auth_token_here"
receipt_data = {
    "CustomerId": "CUST001",
    "ReceiptDate": "2024-11-20",
    "Bank": "Main Bank",
    "BankGLDivision": "DIV001",
    "ReceivedFrom": "John Doe",
    "Name": "Policy Payment",
    "Description": "Monthly policy premium payment",
    "PaymentType": "Check",
    "BankNumber": "123456",
    "CheckNumber": "987654",
    "Amount": 1500.00,
    "CreditCardAuthorizationNumber": "",
}

insured_accounts = [
    {
        "CustomerId": "CUST001",
        "PolicyId": "POL12345",
        "InvoiceId": "INV001",
        "Amount": 500.00,
        "Description": "First payment",
        "RowType": "AR",
        "GLDivisionCode": "DIV001",
        "GLBranchCode": "BR001",
        "GLDepartmentCode": "DEP001",
        "GLGroupCode": "GRP001",
    },
    {
        "CustomerId": "CUST001",
        "PolicyId": "POL12346",
        "InvoiceId": "INV002",
        "Amount": 1000.00,
        "Description": "Second payment",
        "RowType": "AR",
        "GLDivisionCode": "DIV001",
        "GLBranchCode": "BR002",
        "GLDepartmentCode": "DEP002",
        "GLGroupCode": "GRP002",
    },
]

direct_bill_deposits = [
    {
        "Amount": 300.00,
        "Description": "Direct bill from insurer",
        "InsuranceCompanyCode": "INS001",
        "FinanceCompanyCode": "FIN001",
        "BrokerageCompanyCode": "BRO001",
    },
    {
        "Amount": 200.00,
        "Description": "Additional bill",
        "InsuranceCompanyCode": "INS002",
        "FinanceCompanyCode": "FIN002",
        "BrokerageCompanyCode": "BRO002",
    },
]


def insert_remark(auth_token, remark_details):
    """
    Sends a SOAP request to insert a remark.

    Parameters:
        auth_token (str): The authorization token for the API.
        remark_details (dict): Dictionary containing the remark details.

    Returns:
        Response: The response from the SOAP API.
    """
    # Construct the SOAP body with dynamic values
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <InsertRemark_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Remark>
            <RemarkId>{remark_details.get("RemarkId", "")}</RemarkId>
            <RemarkText>{remark_details.get("RemarkText", "")}</RemarkText>
            <IsSaved>{str(remark_details.get("IsSaved", False)).lower()}</IsSaved>
            <IsUploaded>{str(remark_details.get("IsUploaded", False)).lower()}</IsUploaded>
            <TransactionEffectiveDate>{remark_details.get("TransactionEffectiveDate", "")}</TransactionEffectiveDate>
            <RemarkType>{remark_details.get("RemarkType", 0)}</RemarkType>
            <ParentId>{remark_details.get("ParentId", "")}</ParentId>
            <PolicyId>{remark_details.get("PolicyId", "")}</PolicyId>
            <LineOfBusinessId>{remark_details.get("LineOfBusinessId", "")}</LineOfBusinessId>
          </Remark>
        </InsertRemark_Request>
      </soap12:Body>
    </soap12:Envelope>
    """

    # Set headers, including Authorization
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/InsertRemark",
        "Authorization": f"Bearer {auth_token}",
    }

    # Define the endpoint URL
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Make the POST request
    response = requests.post(url, data=soap_body, headers=headers)

    # Output the response
    if response.status_code == 200:
        print("Request Successful!")
        print(response.text)
        return response.text
    else:
        print(f"Request Failed with Status Code: {response.status_code}")
        print(response.text)


# Example usage for Insert Remark
remark_details = {
    "RemarkId": "1234",
    "RemarkText": "Sample remark text",
    "IsSaved": True,
    "IsUploaded": False,
    "TransactionEffectiveDate": "2024-11-20T12:00:00",
    "RemarkType": 1,
    "ParentId": "5678",
    "PolicyId": "91011",
    "LineOfBusinessId": "31415"
}


def insert_suspense(auth_token, suspense_details):
    """
    Sends a SOAP request to insert a suspense record.

    Parameters:
        auth_token (str): The authorization token for the API.
        suspense_details (dict): Dictionary containing the suspense details.

    Returns:
        Response: The response from the SOAP API.
    """
    # Construct the SOAP body with dynamic values
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <InsertSuspense_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Suspense>
            <SuspenseId>{suspense_details.get("SuspenseId", "")}</SuspenseId>
            <AttachedTo>{suspense_details.get("AttachedTo", "")}</AttachedTo>
            <SuspenseType>{suspense_details.get("SuspenseType", 0)}</SuspenseType>
            <PolicyId>{suspense_details.get("PolicyId", "")}</PolicyId>
            <PolicyEffectiveDate>{suspense_details.get("PolicyEffectiveDate", "")}</PolicyEffectiveDate>
            <CompanyCode>{suspense_details.get("CompanyCode", "")}</CompanyCode>
            <ClaimId>{suspense_details.get("ClaimId", "")}</ClaimId>
            <SuspenseAction>{suspense_details.get("SuspenseAction", "")}</SuspenseAction>
            <InitiatedDate>{suspense_details.get("InitiatedDate", "")}</InitiatedDate>
            <InitiatedByEmpCode>{suspense_details.get("InitiatedByEmpCode", "")}</InitiatedByEmpCode>
            <AssignedToEmpCode>{suspense_details.get("AssignedToEmpCode", "")}</AssignedToEmpCode>
            <DueDate>{suspense_details.get("DueDate", "")}</DueDate>
            <TimesRescheduled>{suspense_details.get("TimesRescheduled", 0)}</TimesRescheduled>
            <Description>{suspense_details.get("Description", "")}</Description>
            <Priority>{suspense_details.get("Priority", 0)}</Priority>
            <IsComplete>{str(suspense_details.get("IsComplete", False)).lower()}</IsComplete>
          </Suspense>
        </InsertSuspense_Request>
      </soap12:Body>
    </soap12:Envelope>
    """

    # Set headers, including Authorization
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/InsertSuspense",
        "Authorization": f"Bearer {auth_token}",
    }

    # Define the endpoint URL
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Make the POST request
    response = requests.post(url, data=soap_body, headers=headers)

    # Output the response
    if response.status_code == 200:
        print("Request Successful!")
        print(response.text)
        return response.text
    else:
        print(f"Request Failed with Status Code: {response.status_code}")
        print(response.text)

    return response

# Example input for Insert Suspense

suspense_details = {
    "SuspenseId": "1234",
    "AttachedTo": "Policy",
    "SuspenseType": 1,
    "PolicyId": "5678",
    "PolicyEffectiveDate": "2024-11-20T12:00:00",
    "CompanyCode": "001",
    "ClaimId": "91011",
    "SuspenseAction": "Action needed",
    "InitiatedDate": "2024-11-20T12:00:00",
    "InitiatedByEmpCode": "EMP001",
    "AssignedToEmpCode": "EMP002",
    "DueDate": "2024-12-01T12:00:00",
    "TimesRescheduled": 0,
    "Description": "Suspense description",
    "Priority": 1,
    "IsComplete": False
}

def logout_request(auth_token):
    """
    Sends a SOAP request to log out using the Logout action.

    Parameters:
        auth_token (str): The authorization token for the API.

    Returns:
        Response: The response from the SOAP API.
    """
    # SOAP Body with an empty Body section for Logout
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body />
    </soap12:Envelope>
    """

    # Set headers
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/Logout",
        "Authorization": f"Bearer {auth_token}",
    }

    # Define the endpoint URL
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Make the POST request
    response = requests.post(url, data=soap_body, headers=headers)

    # Output the response
    if response.status_code == 200:
        print("Logout Successful!")
        print(response.text)
        return response.text
    else:
        print(f"Request Failed with Status Code: {response.status_code}")
        print(response.text)


def post_invoice(auth_token, policy_id):
    """
    Sends a SOAP request to post an invoice for a specified policy ID.

    Parameters:
        auth_token (str): The authorization token for the API.
        policy_id (str): The ID of the policy for which to post the invoice.

    Returns:
        Response: The response from the SOAP API.
    """
    # SOAP Request Body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <PostInvoice_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <PolicyId>{policy_id}</PolicyId>
        </PostInvoice_Request>
      </soap12:Body>
    </soap12:Envelope>
    """

    # Set headers
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/PostInvoice",
        "Authorization": f"Bearer {auth_token}",
    }

    # Define the endpoint URL
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Make the POST request
    response = requests.post(url, data=soap_body, headers=headers)

    # Output the response
    if response.status_code == 200:
        print("PostInvoice Request Successful!")
        print(response.text)
        return response.text
    else:
        print(f"Request Failed with Status Code: {response.status_code}")
        print(response.text)

def renew_policy(auth_token, policy_data, additional_personnel, line_of_business, transaction_details):
    """
    Sends a SOAP request to renew a policy with the given details.

    Parameters:
        auth_token (str): The authorization token for the API.
        policy_data (dict): Dictionary containing policy details.
        additional_personnel (list): List of additional personnel dictionaries.
        line_of_business (list): List of line of business dictionaries.
        transaction_details (dict): Dictionary containing transaction details.

    Returns:
        Response: The response from the SOAP API.
    """
    # Construct SOAP Request Body
    policy_xml = f"""
        <CustomerId>{policy_data['CustomerId']}</CustomerId>
        <PolicyId>{policy_data['PolicyId']}</PolicyId>
        <PolicyNumber>{policy_data['PolicyNumber']}</PolicyNumber>
        <TypeOfBusiness>{policy_data['TypeOfBusiness']}</TypeOfBusiness>
        <PolicyType>{policy_data['PolicyType']}</PolicyType>
        <PolicyEffectiveDate>{policy_data['PolicyEffectiveDate']}</PolicyEffectiveDate>
        <PolicyExpirationDate>{policy_data['PolicyExpirationDate']}</PolicyExpirationDate>
        <IsContinuous>{str(policy_data['IsContinuous']).lower()}</IsContinuous>
        <PolicyStatus>{policy_data['PolicyStatus']}</PolicyStatus>
    """

    personnel_xml = "".join(
        f"""
        <PolicyPersonnel>
          <EmployeeCode>{personnel['EmployeeCode']}</EmployeeCode>
          <EmployeeType>{personnel['EmployeeType']}</EmployeeType>
          <IsPrimary>{str(personnel['IsPrimary']).lower()}</IsPrimary>
        </PolicyPersonnel>
        """
        for personnel in additional_personnel
    )

    line_of_business_xml = "".join(
        f"""
        <PolicyLineOfBusiness>
          <LineOfBusinessId>{lob['LineOfBusinessId']}</LineOfBusinessId>
          <LineOfBusiness>{lob['LineOfBusiness']}</LineOfBusiness>
          <SortOrderNumber>{lob['SortOrderNumber']}</SortOrderNumber>
          <Action>{lob['Action']}</Action>
        </PolicyLineOfBusiness>
        """
        for lob in line_of_business
    )

    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <RenewPolicy_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Policy>
            {policy_xml}
          </Policy>
          <AdditionalPersonnel>
            {personnel_xml}
          </AdditionalPersonnel>
          <LineOfBusiness>
            {line_of_business_xml}
          </LineOfBusiness>
          <TransactionType>{transaction_details['TransactionType']}</TransactionType>
          <TransactionDescription>{transaction_details['TransactionDescription']}</TransactionDescription>
          <PriorPolicyId>{transaction_details['PriorPolicyId']}</PriorPolicyId>
        </RenewPolicy_Request>
      </soap12:Body>
    </soap12:Envelope>
    """

    # Set headers
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/RenewPolicy",
        "Authorization": f"Bearer {auth_token}",
    }

    # Define the endpoint URL
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Make the POST request
    response = requests.post(url, data=soap_body, headers=headers)

    # Output the response
    if response.status_code == 200:
        print("RenewPolicy Request Successful!")
        print(response.text)
        return response.text
    else:
        print(f"Request Failed with Status Code: {response.status_code}")
        print(response.text)

# Example usage of Renew Policy

policy_data = {
    "CustomerId": "123",
    "PolicyId": "456",
    "PolicyNumber": "789",
    "TypeOfBusiness": "1",
    "PolicyType": "General",
    "PolicyEffectiveDate": "2024-11-01",
    "PolicyExpirationDate": "2025-11-01",
    "IsContinuous": True,
    "PolicyStatus": "Active",
}

additional_personnel = [
    {"EmployeeCode": "E001", "EmployeeType": "Manager", "IsPrimary": True},
    {"EmployeeCode": "E002", "EmployeeType": "Rep", "IsPrimary": False},
]

line_of_business = [
    {"LineOfBusinessId": "LOB001", "LineOfBusiness": "Auto", "SortOrderNumber": 1, "Action": "Add"},
    {"LineOfBusinessId": "LOB002", "LineOfBusiness": "Home", "SortOrderNumber": 2, "Action": "Update"},
]

transaction_details = {
    "TransactionType": "Renewal",
    "TransactionDescription": "Policy renewal for 2025",
    "PriorPolicyId": "123456",
}

def search_by_phone_number(auth_token, phone_number):
    """
    Sends a SOAP request to search for a record by phone number.

    Parameters:
        auth_token (str): The authorization token for the API.
        phone_number (str): The phone number to search.

    Returns:
        Response: The response from the SOAP API.
    """
    # SOAP Request Body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <SearchByPhoneNumber_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <PhoneNumber>{phone_number}</PhoneNumber>
        </SearchByPhoneNumber_Request>
      </soap12:Body>
    </soap12:Envelope>
    """

    # Set headers
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/SearchByPhoneNumber",
        "Authorization": f"Bearer {auth_token}",
    }

    # Define the endpoint URL
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Make the POST request
    response = requests.post(url, data=soap_body, headers=headers)

    # Output the response
    if response.status_code == 200:
        print("SearchByPhoneNumber Request Successful!")
        print(response.text)
        return response.text
    else:
        print(f"Request Failed with Status Code: {response.status_code}")
        print(response.text)


def send_file_chunk(auth_token, file_chunk_id, doc_stage_id, file_data, sequence_number):
    """
    Sends a SOAP request to upload a file chunk.

    Parameters:
        auth_token (str): The authorization token for the API.
        file_chunk_id (str): The ID of the file chunk.
        doc_stage_id (str): The ID of the document stage.
        file_data (bytes): The binary data of the file chunk.
        sequence_number (int): The sequence number of the chunk.

    Returns:
        Response: The response from the SOAP API.
    """
    # Encode the file data to Base64
    encoded_data = base64.b64encode(file_data).decode("utf-8")

    # SOAP Request Body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <SendFileChunk_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <FileChunk>
            <FileChunkId>{file_chunk_id}</FileChunkId>
            <DocStageId>{doc_stage_id}</DocStageId>
            <Data>{encoded_data}</Data>
            <SequenceNumber>{sequence_number}</SequenceNumber>
          </FileChunk>
        </SendFileChunk_Request>
      </soap12:Body>
    </soap12:Envelope>
    """

    # Set headers
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/SendFileChunk",
        "Authorization": f"Bearer {auth_token}",
    }

    # Define the endpoint URL
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Make the POST request
    response = requests.post(url, data=soap_body, headers=headers)

    # Output the response
    if response.status_code == 200:
        print("SendFileChunk Request Successful!")
        print(response.text)
        return response.text
    else:
        print(f"Request Failed with Status Code: {response.status_code}")
        print(response.text)

# Example usage of input  File Chunk
file_chunk_id = "chunk123"
doc_stage_id = "stage456"
file_data = b"This is a sample file data"  # Replace with actual file content
sequence_number = 1


def update_al3_policy(auth_token, policy_data):
    """
    Sends a SOAP request to update an AL3 policy.

    Parameters:
        auth_token (str): The authorization token for the API.
        policy_data (dict): The data for the policy update.

    Returns:
        Response: The response from the SOAP API.
    """
    # SOAP Request Body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <UpdateAL3Policy_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <PolicyId>{policy_data['PolicyId']}</PolicyId>
          <PolicyNumber>{policy_data['PolicyNumber']}</PolicyNumber>
          <CustomerId>{policy_data['CustomerId']}</CustomerId>
          <BillMethod>{policy_data['BillMethod']}</BillMethod>
          <PolicyEffectiveDate>{policy_data['PolicyEffectiveDate']}</PolicyEffectiveDate>
          <PolicyExpirationDate>{policy_data['PolicyExpirationDate']}</PolicyExpirationDate>
          <TransactionEffectiveDate>{policy_data['TransactionEffectiveDate']}</TransactionEffectiveDate>
          <CompanyCode>{policy_data['CompanyCode']}</CompanyCode>
          <TransactionType>{policy_data['TransactionType']}</TransactionType>
          <TransactionDescription>{policy_data['TransactionDescription']}</TransactionDescription>
          <IsRenewRewrite>{str(policy_data['IsRenewRewrite']).lower()}</IsRenewRewrite>
          <FileContents>{policy_data['FileContents']}</FileContents>
          <AdditionalInformation>
            <CustomerNotation>{policy_data['AdditionalInformation']['CustomerNotation']}</CustomerNotation>
            <PolicyNotation>{policy_data['AdditionalInformation']['PolicyNotation']}</PolicyNotation>
          </AdditionalInformation>
        </UpdateAL3Policy_Request>
      </soap12:Body>
    </soap12:Envelope>
    """

    # Set headers
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/UpdateAL3Policy",
        "Authorization": f"Bearer {auth_token}",
    }

    # Define the endpoint URL
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Make the POST request
    response = requests.post(url, data=soap_body, headers=headers)

    # Output the response
    if response.status_code == 200:
        print("UpdateAL3Policy Request Successful!")
        print(response.text)
        return response.text
    else:
        print(f"Request Failed with Status Code: {response.status_code}")
        print(response.text)

# Example of Input
policy_data = {
    "PolicyId": "12345",
    "PolicyNumber": "POLICY-67890",
    "CustomerId": "CUSTOMER-12345",
    "BillMethod": "Direct Bill",
    "PolicyEffectiveDate": "2024-01-01",
    "PolicyExpirationDate": "2025-01-01",
    "TransactionEffectiveDate": "2024-11-20",
    "CompanyCode": "COMPANY-123",
    "TransactionType": "Renewal",
    "TransactionDescription": "Renewed for the year 2024-2025",
    "IsRenewRewrite": True,
    "FileContents": "AL3_POLICY_FILE_CONTENTS",
    "AdditionalInformation": {
        "CustomerNotation": "Customer prefers electronic communication.",
        "PolicyNotation": "Policy renewed with updated terms.",
    },
}


def update_customer(auth_token, customer_data):
    """
    Sends a SOAP request to update a customer.

    Parameters:
        auth_token (str): The authorization token for the API.
        customer_data (dict): The customer data for the update request.

    Returns:
        Response: The response from the SOAP API.
    """
    # SOAP Request Body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <UpdateCustomer_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Customer>
            <CustomerNumber>{customer_data['CustomerNumber']}</CustomerNumber>
            <CustomerId>{customer_data['CustomerId']}</CustomerId>
            <LastName>{customer_data['LastName']}</LastName>
            <FirstName>{customer_data['FirstName']}</FirstName>
            <MiddleName>{customer_data['MiddleName']}</MiddleName>
            <FirmName>{customer_data['FirmName']}</FirmName>
            <AddressLine1>{customer_data['AddressLine1']}</AddressLine1>
            <AddressLine2>{customer_data['AddressLine2']}</AddressLine2>
            <City>{customer_data['City']}</City>
            <State>{customer_data['State']}</State>
            <County>{customer_data['County']}</County>
            <ZipCode>{customer_data['ZipCode']}</ZipCode>
            <HomeAreaCode>{customer_data['HomeAreaCode']}</HomeAreaCode>
            <HomePhone>{customer_data['HomePhone']}</HomePhone>
            <HomeExtension>{customer_data['HomeExtension']}</HomeExtension>
            <BusinessAreaCode>{customer_data['BusinessAreaCode']}</BusinessAreaCode>
            <BusinessPhone>{customer_data['BusinessPhone']}</BusinessPhone>
            <BusinessExtension>{customer_data['BusinessExtension']}</BusinessExtension>
            <FaxAreaCode>{customer_data['FaxAreaCode']}</FaxAreaCode>
            <FaxPhone>{customer_data['FaxPhone']}</FaxPhone>
            <FaxExtension>{customer_data['FaxExtension']}</FaxExtension>
            <CellAreaCode>{customer_data['CellAreaCode']}</CellAreaCode>
            <CellPhone>{customer_data['CellPhone']}</CellPhone>
            <CellExtension>{customer_data['CellExtension']}</CellExtension>
            <PagerAreaCode>{customer_data['PagerAreaCode']}</PagerAreaCode>
            <PagerPhone>{customer_data['PagerPhone']}</PagerPhone>
            <PagerExtension>{customer_data['PagerExtension']}</PagerExtension>
            <Email>{customer_data['Email']}</Email>
            <WebAddress>{customer_data['WebAddress']}</WebAddress>
            <FormalSalutation>{customer_data['FormalSalutation']}</FormalSalutation>
            <InformalSalutation>{customer_data['InformalSalutation']}</InformalSalutation>
            <DateOfBirth>{customer_data['DateOfBirth']}</DateOfBirth>
            <Occupation>{customer_data['Occupation']}</Occupation>
            <MaritalStatus>{customer_data['MaritalStatus']}</MaritalStatus>
            <InBusinessSince>{customer_data['InBusinessSince']}</InBusinessSince>
            <BusinessEntityType>{customer_data['BusinessEntityType']}</BusinessEntityType>
            <DoingBusinessAs>{customer_data['DoingBusinessAs']}</DoingBusinessAs>
            <SICCode>{customer_data['SICCode']}</SICCode>
            <FederalTaxIdNumber>{customer_data['FederalTaxIdNumber']}</FederalTaxIdNumber>
            <DUNSNumber>{customer_data['DUNSNumber']}</DUNSNumber>
            <CustomerType>{customer_data['CustomerType']}</CustomerType>
            <IsBrokersCustomer>{str(customer_data['IsBrokersCustomer']).lower()}</IsBrokersCustomer>
            <IsActive>{str(customer_data['IsActive']).lower()}</IsActive>
            <AccountExecCode>{customer_data['AccountExecCode']}</AccountExecCode>
            <AccountRepCode>{customer_data['AccountRepCode']}</AccountRepCode>
            <BrokerCode>{customer_data['BrokerCode']}</BrokerCode>
            <GLDivisionCode>{customer_data['GLDivisionCode']}</GLDivisionCode>
            <GLDepartmentCode>{customer_data['GLDepartmentCode']}</GLDepartmentCode>
            <GLBranchCode>{customer_data['GLBranchCode']}</GLBranchCode>
            <GLGroupCode>{customer_data['GLGroupCode']}</GLGroupCode>
            <DateCustomerAdded>{customer_data['DateCustomerAdded']}</DateCustomerAdded>
            <IsPersonal>{str(customer_data['IsPersonal']).lower()}</IsPersonal>
            <IsCommercial>{str(customer_data['IsCommercial']).lower()}</IsCommercial>
            <IsLife>{str(customer_data['IsLife']).lower()}</IsLife>
            <IsHealth>{str(customer_data['IsHealth']).lower()}</IsHealth>
            <IsNonPropertyAndCasualty>{str(customer_data['IsNonPropertyAndCasualty']).lower()}</IsNonPropertyAndCasualty>
            <IsFinancial>{str(customer_data['IsFinancial']).lower()}</IsFinancial>
            <IsBenefits>{str(customer_data['IsBenefits']).lower()}</IsBenefits>
            <NAICSCode>{customer_data['NAICSCode']}</NAICSCode>
          </Customer>
        </UpdateCustomer_Request>
      </soap12:Body>
    </soap12:Envelope>
    """

    # Set headers
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/UpdateCustomer",
        "Authorization": f"Bearer {auth_token}",  # Assuming Bearer token for Authorization
    }

    # Define the endpoint URL
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Make the POST request
    response = requests.post(url, data=soap_body, headers=headers)

    # Output the response
    if response.status_code == 200:
        print("UpdateCustomer Request Successful!")
        print(response.text)
        return response.text
    else:
        print(f"Request Failed with Status Code: {response.status_code}")
        print(response.text)

    return response

# Example usage
customer_data = {
    "CustomerNumber": 12345,
    "CustomerId": "CUST-001",
    "LastName": "Smith",
    "FirstName": "John",
    "MiddleName": "A",
    "FirmName": "Tech Solutions",
    "AddressLine1": "123 Main St",
    "AddressLine2": "Suite 101",
    "City": "Anytown",
    "State": "CA",
    "County": "AnyCounty",
    "ZipCode": "12345",
    "HomeAreaCode": "123",
    "HomePhone": "4567890",
    "HomeExtension": "101",
    "BusinessAreaCode": "124",
    "BusinessPhone": "5678901",
    "BusinessExtension": "102",
    "FaxAreaCode": "125",
    "FaxPhone": "6789012",
    "FaxExtension": "103",
    "CellAreaCode": "126",
    "CellPhone": "7890123",
    "CellExtension": "104",
    "PagerAreaCode": "127",
    "PagerPhone": "8901234",
    "PagerExtension": "105",
    "Email": "john.smith@example.com",
    "WebAddress": "www.techsolutions.com",
    "FormalSalutation": "Mr. Smith",
    "InformalSalutation": "John",
    "DateOfBirth": "1980-01-01",
    "Occupation": "Engineer",
    "MaritalStatus": "Single",
    "InBusinessSince": "2000-01-01",
    "BusinessEntityType": "LLC",
    "DoingBusinessAs": "Tech Solutions",
    "SICCode": "1234",
    "FederalTaxIdNumber": "12-3456789",
    "DUNSNumber": "123456789",
    "CustomerType": "Commercial",
    "IsBrokersCustomer": True,
    "IsActive": True,
    "AccountExecCode": "AE001",
    "AccountRepCode": "AR002",
    "BrokerCode": "BR003",
    "GLDivisionCode": "DIV001",
    "GLDepartmentCode": "DEP001",
    "GLBranchCode": "BR001",
    "GLGroupCode": "GR001",
    "DateCustomerAdded": "2024-11-20",
    "IsPersonal": False,
    "IsCommercial": True,
    "IsLife": False,
    "IsHealth": False,
    "IsNonPropertyAndCasualty": False,
    "IsFinancial": False,
    "IsBenefits": False,
    "NAICSCode": "9876",
}

def update_customer_profile_answer(auth_token, answer_text, customer_id, question_id):
    """
    Sends a SOAP request to update a customer profile answer.

    Parameters:
        auth_token (str): The authorization token for the API.
        answer_text (str): The answer text to be updated.
        customer_id (str): The customer ID.
        question_id (str): The question ID.

    Returns:
        Response: The response from the SOAP API.
    """
    # SOAP Request Body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <UpdateCustomerProfileAnswer_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <AnswerText>{answer_text}</AnswerText>
          <CustomerId>{customer_id}</CustomerId>
          <QuestionId>{question_id}</QuestionId>
        </UpdateCustomerProfileAnswer_Request>
      </soap12:Body>
    </soap12:Envelope>
    """

    # Set headers with Authorization
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/UpdateCustomerProfileAnswer",
        "Authorization": f"Bearer {auth_token}",  # Assuming Bearer token for Authorization
    }

    # Define the endpoint URL
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Make the POST request
    response = requests.post(url, data=soap_body, headers=headers)

    # Output the response
    if response.status_code == 200:
        print("Request was successful!")
        print(response.text)
        return response.text
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)

# Example usage
answer_text = "The customer prefers email communication."
customer_id = "CUST-001"
question_id = "QID-123"


def update_policy_transaction_premium(auth_token, policy_transaction_details):
    """
    Sends a SOAP request to update a policy transaction premium.

    Parameters:
        auth_token (str): The authorization token for the API.
        policy_transaction_details (dict): A dictionary containing the policy transaction details.

    Returns:
        Response: The response from the SOAP API.
    """
    # SOAP Request Body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <UpdatePolicyTransactionPremium_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <PolicyTransactionPremium>
            <PolicyId>{policy_transaction_details["PolicyId"]}</PolicyId>
            <TransactionEffectiveDate>{policy_transaction_details["TransactionEffectiveDate"]}</TransactionEffectiveDate>
            <PolicyTransactionPremiumId>{policy_transaction_details["PolicyTransactionPremiumId"]}</PolicyTransactionPremiumId>
            <LineOfBusiness>{policy_transaction_details["LineOfBusiness"]}</LineOfBusiness>
            <PlanType>{policy_transaction_details["PlanType"]}</PlanType>
            <WritingCompanyCode>{policy_transaction_details["WritingCompanyCode"]}</WritingCompanyCode>
            <Description>{policy_transaction_details["Description"]}</Description>
            <Premium>{policy_transaction_details["Premium"]}</Premium>
            <BilledPremium>{policy_transaction_details["BilledPremium"]}</BilledPremium>
            <WrittenPremium>{policy_transaction_details["WrittenPremium"]}</WrittenPremium>
            <FullTermPremium>{policy_transaction_details["FullTermPremium"]}</FullTermPremium>
            <EstRevenue>{policy_transaction_details["EstRevenue"]}</EstRevenue>
            <IncludePremium>{policy_transaction_details["IncludePremium"]}</IncludePremium>
            <HowBilled>{policy_transaction_details["HowBilled"]}</HowBilled>
            <IsBilled>{policy_transaction_details["IsBilled"]}</IsBilled>
            <IsSuspended>{policy_transaction_details["IsSuspended"]}</IsSuspended>
            <IsCorrected>{policy_transaction_details["IsCorrected"]}</IsCorrected>
            <Reconciled>{policy_transaction_details["Reconciled"]}</Reconciled>
          </PolicyTransactionPremium>
        </UpdatePolicyTransactionPremium_Request>
      </soap12:Body>
    </soap12:Envelope>
    """

    # Set headers with Authorization
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/UpdatePolicyTransactionPremium",
        "Authorization": f"Bearer {auth_token}",  # Assuming Bearer token for Authorization
    }

    # Define the endpoint URL
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Make the POST request
    response = requests.post(url, data=soap_body, headers=headers)

    # Output the response
    if response.status_code == 200:
        print("Request was successful!")
        print(response.text)
        return response.text
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)

# Example input format

policy_transaction_details = {
    "PolicyId": "POL-12345",
    "TransactionEffectiveDate": "2024-01-01",
    "PolicyTransactionPremiumId": "PTP-12345",
    "LineOfBusiness": "Auto",
    "PlanType": "Standard",
    "WritingCompanyCode": "WC01",
    "Description": "Policy premium update",
    "Premium": 1200.50,
    "BilledPremium": 1200.50,
    "WrittenPremium": 1200.50,
    "FullTermPremium": 1200.50,
    "EstRevenue": 1000.00,
    "IncludePremium": True,
    "HowBilled": "Monthly",
    "IsBilled": True,
    "IsSuspended": False,
    "IsCorrected": False,
    "Reconciled": "No"
}

def update_remark(auth_token, remark_details):
    """
    Sends a SOAP request to update a remark.

    Parameters:
        auth_token (str): The authorization token for the API.
        remark_details (dict): A dictionary containing the remark details.

    Returns:
        Response: The response from the SOAP API.
    """
    # SOAP Request Body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <UpdateRemark_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Remark>
            <RemarkId>{remark_details["RemarkId"]}</RemarkId>
            <RemarkText>{remark_details["RemarkText"]}</RemarkText>
            <IsSaved>{str(remark_details["IsSaved"]).lower()}</IsSaved>
            <IsUploaded>{str(remark_details["IsUploaded"]).lower()}</IsUploaded>
            <TransactionEffectiveDate>{remark_details["TransactionEffectiveDate"]}</TransactionEffectiveDate>
            <RemarkType>{remark_details["RemarkType"]}</RemarkType>
            <ParentId>{remark_details["ParentId"]}</ParentId>
            <PolicyId>{remark_details["PolicyId"]}</PolicyId>
            <LineOfBusinessId>{remark_details["LineOfBusinessId"]}</LineOfBusinessId>
          </Remark>
        </UpdateRemark_Request>
      </soap12:Body>
    </soap12:Envelope>
    """

    # Set headers with Authorization
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/UpdateRemark",  # Assuming SOAPAction for the API method
        "Authorization": f"Bearer {auth_token}",  # Authorization token
    }

    # Define the endpoint URL
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Make the POST request
    response = requests.post(url, data=soap_body, headers=headers)

    # Output the response
    if response.status_code == 200:
        print("Request was successful!")
        print(response.text)

        return  response.text
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)

# Example of input format
remark_details = {
    "RemarkId": "RMK-12345",
    "RemarkText": "This is an updated remark.",
    "IsSaved": True,
    "IsUploaded": False,
    "TransactionEffectiveDate": "2024-11-20T10:00:00",  # Example datetime format
    "RemarkType": 1,  # Example type
    "ParentId": "PARENT-12345",
    "PolicyId": "POL-12345",
    "LineOfBusinessId": "LOB-12345"
}

def update_suspense(auth_token, suspense_details):
    """
    Sends a SOAP request to update a suspense record.

    Parameters:
        auth_token (str): The authorization token for the API.
        suspense_details (dict): A dictionary containing the suspense details.

    Returns:
        Response: The response from the SOAP API.
    """
    # SOAP Request Body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <UpdateSuspense_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Suspense>
            <SuspenseId>{suspense_details["SuspenseId"]}</SuspenseId>
            <AttachedTo>{suspense_details["AttachedTo"]}</AttachedTo>
            <SuspenseType>{suspense_details["SuspenseType"]}</SuspenseType>
            <PolicyId>{suspense_details["PolicyId"]}</PolicyId>
            <PolicyEffectiveDate>{suspense_details["PolicyEffectiveDate"]}</PolicyEffectiveDate>
            <CompanyCode>{suspense_details["CompanyCode"]}</CompanyCode>
            <ClaimId>{suspense_details["ClaimId"]}</ClaimId>
            <SuspenseAction>{suspense_details["SuspenseAction"]}</SuspenseAction>
            <InitiatedDate>{suspense_details["InitiatedDate"]}</InitiatedDate>
            <InitiatedByEmpCode>{suspense_details["InitiatedByEmpCode"]}</InitiatedByEmpCode>
            <AssignedToEmpCode>{suspense_details["AssignedToEmpCode"]}</AssignedToEmpCode>
            <DueDate>{suspense_details["DueDate"]}</DueDate>
            <TimesRescheduled>{suspense_details["TimesRescheduled"]}</TimesRescheduled>
            <Description>{suspense_details["Description"]}</Description>
            <Priority>{suspense_details["Priority"]}</Priority>
            <IsComplete>{str(suspense_details["IsComplete"]).lower()}</IsComplete>
          </Suspense>
        </UpdateSuspense_Request>
      </soap12:Body>
    </soap12:Envelope>
    """

    # Set headers with Authorization
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/UpdateSuspense",  # Assuming SOAPAction for the API method
        "Authorization": f"Bearer {auth_token}",  # Authorization token
    }

    # Define the endpoint URL
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Make the POST request
    response = requests.post(url, data=soap_body, headers=headers)

    # Output the response
    if response.status_code == 200:
        print("Request was successful!")
        print(response.text)
        return response.text
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)

# Example of input format
suspense_details = {
    "SuspenseId": "SUSP-12345",
    "AttachedTo": "Claim",
    "SuspenseType": 1,  # Short value, e.g., 1 for high priority
    "PolicyId": "POL-12345",
    "PolicyEffectiveDate": "2024-11-20T10:00:00",  # Example datetime format
    "CompanyCode": "COMP-123",
    "ClaimId": "CLAIM-67890",
    "SuspenseAction": "Follow-up",
    "InitiatedDate": "2024-11-19T10:00:00",
    "InitiatedByEmpCode": "EMP-001",
    "AssignedToEmpCode": "EMP-002",
    "DueDate": "2024-12-01T10:00:00",
    "TimesRescheduled": 1,
    "Description": "Follow-up action required",
    "Priority": 2,  # Short value, e.g., 2 for medium priority
    "IsComplete": False
}


def validateAgentLogin(agency_no, login_id, password, auth_token):
    """
    Sends a SOAP request to authenticate a user.

    Parameters:
        agency_no (str): The agency number.
        login_id (str): The login ID.
        password (str): The password.
        auth_token (str): The authorization token for the API.

    Returns:
        Response: The response from the SOAP API.
    """
    # SOAP Request Body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Body>
        <AgencyNo xmlns="http://www.WSAPI.AMS360.com/v2.0">{agency_no}</AgencyNo>
        <LoginId xmlns="http://www.WSAPI.AMS360.com/v2.0">{login_id}</LoginId>
        <Password xmlns="http://www.WSAPI.AMS360.com/v2.0">{password}</Password>
      </soap12:Body>
    </soap12:Envelope>
    """

    # Set headers with Authorization
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/ValidateAgentLogin",  # Assuming SOAPAction for the API method
        "Authorization": f"Bearer {auth_token}",  # Authorization token
    }

    # Define the endpoint URL
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Make the POST request
    response = requests.post(url, data=soap_body, headers=headers)

    # Output the response
    if response.status_code == 200:
        print("Request was successful!")
        print(response.text)
        return response.text
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)


def begin_file_chunk_request(auth_token, number_of_chunks, total_bytes):
    """
    Sends a SOAP request to begin a file chunk operation.

    Parameters:
        token (str): The authorization token.
        number_of_chunks (int): The number of chunks.
        total_bytes (int): The total number of bytes.

    Returns:
        Response: The response from the SOAP API.
    """
    # SOAP Request Body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <BeginFileChunk_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <NumberOfChunks>{number_of_chunks}</NumberOfChunks>
          <TotalBytes>{total_bytes}</TotalBytes>
        </BeginFileChunk_Request>
      </soap12:Body>
    </soap12:Envelope>
    """

    # Set headers
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/BeginFileChunk",
        "Authorization": f"Bearer {auth_token}",  # Authorization token
    }

    # Define the endpoint URL
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Send the POST request
    response = requests.post(url, data=soap_body, headers=headers)

    # Output the response
    if response.status_code == 200:
        print("Request was successful!")
        print(response.text)
        return response.text
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)


def change_customer_service_group_personnel(auth_token, modify_list, delete_list, customer_id):
    """
    Sends a SOAP request to change customer service group personnel.

    Parameters:
        token (str): The authorization token.
        modify_list (list): List of customer service group personnel to be modified.
        delete_list (list): List of customer service group personnel to be deleted.
        customer_id (str): The customer ID.

    Returns:
        Response: The response from the SOAP API.
    """
    # Prepare Modify and Delete lists as XML entries
    modify_xml = ""
    for personnel in modify_list:
        modify_xml += f"""
        <CustomerServiceGroupPersonnel>
          <EmployeeCode>{personnel['EmployeeCode']}</EmployeeCode>
          <EmployeeType>{personnel['EmployeeType']}</EmployeeType>
          <TypeOfBusiness>{personnel['TypeOfBusiness']}</TypeOfBusiness>
          <IsPrimary>{personnel['IsPrimary']}</IsPrimary>
          <CustomerId>{personnel['CustomerId']}</CustomerId>
        </CustomerServiceGroupPersonnel>
        """

    delete_xml = ""
    for personnel in delete_list:
        delete_xml += f"""
        <CustomerServiceGroupPersonnel>
          <EmployeeCode>{personnel['EmployeeCode']}</EmployeeCode>
          <EmployeeType>{personnel['EmployeeType']}</EmployeeType>
          <TypeOfBusiness>{personnel['TypeOfBusiness']}</TypeOfBusiness>
          <IsPrimary>{personnel['IsPrimary']}</IsPrimary>
          <CustomerId>{personnel['CustomerId']}</CustomerId>
        </CustomerServiceGroupPersonnel>
        """

    # SOAP Request Body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <ChangeCustomerServiceGroupPersonnel_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <ModifyList>
            {modify_xml}
          </ModifyList>
          <DeleteList>
            {delete_xml}
          </DeleteList>
          <CustomerId>{customer_id}</CustomerId>
        </ChangeCustomerServiceGroupPersonnel_Request>
      </soap12:Body>
    </soap12:Envelope>
    """

    # Set headers
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "Authorization": f"Bearer {auth_token}",  # Authorization token
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/ChangeCustomerServiceGroupPersonnel"  # SOAPAction header
    }

    # Define the endpoint URL
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Send the POST request
    response = requests.post(url, data=soap_body, headers=headers)

    # Output the response
    if response.status_code == 200:
        print("Request was successful!")
        print(response.text)
        return response.text
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)

# Example usage
auth_token = "your_auth_token_here"
modify_list = [
    {"EmployeeCode": "12345", "EmployeeType": "TypeA", "TypeOfBusiness": 1, "IsPrimary": True, "CustomerId": "CUST001"},
    {"EmployeeCode": "67890", "EmployeeType": "TypeB", "TypeOfBusiness": 2, "IsPrimary": False, "CustomerId": "CUST001"}
]
delete_list = [
    {"EmployeeCode": "54321", "EmployeeType": "TypeC", "TypeOfBusiness": 1, "IsPrimary": False,
     "CustomerId": "CUST001"},
    {"EmployeeCode": "98765", "EmployeeType": "TypeD", "TypeOfBusiness": 2, "IsPrimary": True, "CustomerId": "CUST001"}
]
customer_id = "CUST001"


def change_policy_personnel(auth_token, policy_id, modify_list, delete_list):
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Construct SOAP request body
    modify_personnel_str = ""
    for personnel in modify_list:
        modify_personnel_str += f"""
        <PolicyPersonnel>
            <EmployeeCode>{personnel['EmployeeCode']}</EmployeeCode>
            <EmployeeType>{personnel['EmployeeType']}</EmployeeType>
            <IsPrimary>{str(personnel['IsPrimary']).lower()}</IsPrimary>
        </PolicyPersonnel>
        """

    delete_personnel_str = ""
    for personnel in delete_list:
        delete_personnel_str += f"""
        <PolicyPersonnel>
            <EmployeeCode>{personnel['EmployeeCode']}</EmployeeCode>
            <EmployeeType>{personnel['EmployeeType']}</EmployeeType>
            <IsPrimary>{str(personnel['IsPrimary']).lower()}</IsPrimary>
        </PolicyPersonnel>
        """

    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <ChangePolicyPersonnel_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <PolicyId>{policy_id}</PolicyId>
          <ModifyList>
            {modify_personnel_str}
          </ModifyList>
          <DeleteList>
            {delete_personnel_str}
          </DeleteList>
        </ChangePolicyPersonnel_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/ChangePolicyPersonnel",
        # The SOAP Action header for the operation
        "Authorization": f"Bearer {auth_token}"  # Add Authorization header with the auth_token
    }

    # Make the request
    response = requests.post(url, data=soap_body, headers=headers)

    if response.status_code == 200:
        print("Request Successful!")
        print(response.text)  # You can parse the response XML if necessary
        return response.text
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)


# Example usage:
token = "your_auth_token_here"
policy_id = "your_policy_id"
modify_list = [
    {"EmployeeCode": "E123", "EmployeeType": "TypeA", "IsPrimary": True},
    {"EmployeeCode": "E456", "EmployeeType": "TypeB", "IsPrimary": False}
]
delete_list = [
    {"EmployeeCode": "E789", "EmployeeType": "TypeC", "IsPrimary": True},
    {"EmployeeCode": "E101", "EmployeeType": "TypeD", "IsPrimary": False}
]


def delete_customer_profile_answer(auth_token, customer_id, question_id, username, password):
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Construct the SOAP request body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <DeleteCustomerProfileAnswer_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <CustomerId>{customer_id}</CustomerId>
          <QuestionId>{question_id}</QuestionId>
        </DeleteCustomerProfileAnswer_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/DeleteCustomerProfileAnswer",
        # The SOAP Action header for the operation
        "Authorization": f"Bearer {auth_token}"  # Authorization header with the Bearer token
    }

    # Add basic authentication with the provided username and password
    auth = (username, password)

    # Send the POST request
    response = requests.post(url, data=soap_body, headers=headers, auth=auth)

    # Check if the request was successful
    if response.status_code == 200:
        print("Request Successful!")
        print(response.text)  # You can parse the response XML if necessary
        return  response.text
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)

def delete_policy(auth_token, customer_id, policy_id, username, password):
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Construct the SOAP request body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <DeletePolicy_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <CustomerId>{customer_id}</CustomerId>
          <PolicyId>{policy_id}</PolicyId>
        </DeletePolicy_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Set the headers
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/DeletePolicy",  # SOAPAction for the operation
        "Authorization": f"Bearer {auth_token}"  # Bearer token for authorization
    }

    # Add basic authentication (if necessary)
    auth = (username, password)

    # Send the POST request
    response = requests.post(url, data=soap_body, headers=headers, auth=auth)

    # Check if the request was successful
    if response.status_code == 200:
        print("Request Successful!")
        print(response.text)  # You can parse the response XML if necessary
        return response.text
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)


def delete_policy_transaction_premium(auth_token, policy_transaction_premium_id, username, password):
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Construct the SOAP request body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <DeletePolicyTransactionPremium_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <PolicyTransactionPremiumId>{policy_transaction_premium_id}</PolicyTransactionPremiumId>
        </DeletePolicyTransactionPremium_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Set the headers
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/DeletePolicyTransactionPremium",  # SOAPAction for the operation
        "Authorization": f"Bearer {auth_token}"  # Bearer token for authorization
    }

    # Add basic authentication (if necessary)
    auth = (username, password)

    # Send the POST request
    response = requests.post(url, data=soap_body, headers=headers, auth=auth)

    # Check if the request was successful
    if response.status_code == 200:
        print("Request Successful!")
        print(response.text)  # You can parse the response XML if necessary
        return response.text
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)


def delete_remark(auth_token, remark_id, policy_id, line_of_business_id, remark_type, transaction_effective_date,
                  parent_id, username, password):
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Format the transaction effective date
    effective_date = datetime.strptime(transaction_effective_date, '%Y-%m-%dT%H:%M:%S').isoformat()

    # Construct the SOAP request body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <DeleteRemark_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <RemarkId>{remark_id}</RemarkId>
          <PolicyId>{policy_id}</PolicyId>
          <LineOfBusinessId>{line_of_business_id}</LineOfBusinessId>
          <RemarkType>{remark_type}</RemarkType>
          <TransactionEffectiveDate>{effective_date}</TransactionEffectiveDate>
          <ParentId>{parent_id}</ParentId>
        </DeleteRemark_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Set the headers
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/DeleteRemark",  # SOAPAction for the operation
        "Authorization": f"Bearer {auth_token}"  # Bearer token for authorization
    }

    # Add basic authentication (if necessary)
    auth = (username, password)

    # Send the POST request
    response = requests.post(url, data=soap_body, headers=headers, auth=auth)

    # Check if the request was successful
    if response.status_code == 200:
        print("Request Successful!")
        print(response.text)  # You can parse the response XML if necessary
        return response.text
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)


def delete_suspense(auth_token, suspense_id, username, password):
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Construct the SOAP request body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <DeleteSuspense_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <SuspenseId>{suspense_id}</SuspenseId>
        </DeleteSuspense_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Set the headers
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/DeleteSuspense",  # SOAPAction for the operation
        "Authorization": f"Bearer {auth_token}"  # Bearer token for authorization
    }

    # Add basic authentication (if necessary)
    auth = (username, password)

    # Send the POST request
    response = requests.post(url, data=soap_body, headers=headers, auth=auth)

    # Check if the request was successful
    if response.status_code == 200:
        print("Request Successful!")
        print(response.text)  # You can parse the response XML if necessary
        return response.text
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)


def direct_bill_entry_complete(auth_token, batch_id, total_batch_sent, username, password):
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Construct the SOAP request body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <DirectBillEntryComplete_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <BatchId>{batch_id}</BatchId>
          <TotalBatchSent>{total_batch_sent}</TotalBatchSent>
        </DirectBillEntryComplete_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Set the headers
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/DirectBillEntryComplete",  # SOAPAction for the operation
        "Authorization": f"Bearer {auth_token}"  # Bearer token for authorization
    }

    # Add basic authentication (if necessary)
    auth = (username, password)

    # Send the POST request
    response = requests.post(url, data=soap_body, headers=headers, auth=auth)

    # Check if the request was successful
    if response.status_code == 200:
        print("Request Successful!")
        print(response.text)  # You can parse the response XML if necessary
        return response.text
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)


def direct_bill_entry_detail(auth_token, batch_id, direct_bill_entry_details, username, password):
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Construct the SOAP request body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <DirectBillEntryDetail_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <DirectBillEntryDetailList>"""

    # Loop through the DirectBillEntryDetail list and add each entry to the SOAP body
    for detail in direct_bill_entry_details:
        soap_body += f"""
            <DirectBillEntryDetail>
              <DBEHKey>{detail['DBEHKey']}</DBEHKey>
              <DBEDKey>{detail['DBEDKey']}</DBEDKey>
              <CustomerId>{detail['CustomerId']}</CustomerId>
              <PolicyId>{detail['PolicyId']}</PolicyId>
              <WritingCompanyCode>{detail['WritingCompanyCode']}</WritingCompanyCode>
              <TransactionType>{detail['TransactionType']}</TransactionType>
              <TransactionEffectiveDate>{detail['TransactionEffectiveDate']}</TransactionEffectiveDate>
              <LineOfBusiness>{detail['LineOfBusiness']}</LineOfBusiness>
              <PlanType>{detail['PlanType']}</PlanType>
              <BillingChargeCode>{detail['BillingChargeCode']}</BillingChargeCode>
              <BillingChargeCategory>{detail['BillingChargeCategory']}</BillingChargeCategory>
              <NonPremiumRecipient>{detail['NonPremiumRecipient']}</NonPremiumRecipient>
              <GrossPremium>{detail['GrossPremium']}</GrossPremium>
              <SortOrderNumber>{detail['SortOrderNumber']}</SortOrderNumber>
              <Commissions>
                <DirectBillEntryCommission xsi:nil="true" />
                <DirectBillEntryCommission xsi:nil="true" />
              </Commissions>
            </DirectBillEntryDetail>"""

    # Close the DirectBillEntryDetailList and add BatchId
    soap_body += f"""
          </DirectBillEntryDetailList>
          <BatchId>{batch_id}</BatchId>
        </DirectBillEntryDetail_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Set the headers
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/DirectBillEntryDetail",  # SOAPAction for the operation
        "Authorization": f"Bearer {auth_token}"  # Bearer token for authorization
    }

    # Add basic authentication (if necessary)
    auth = (username, password)

    # Send the POST request
    response = requests.post(url, data=soap_body, headers=headers, auth=auth)

    # Check if the request was successful
    if response.status_code == 200:
        print("Request Successful!")
        print(response.text)  # You can parse the response XML if necessary
        return response.text

    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)


# Example usage:
auth_token = "your_auth_token_here"  # Replace with your actual auth token
batch_id = "your_batch_id_here"  # Replace with the actual batch ID

# Example direct bill entry details
direct_bill_entry_details = [
    {
        "DBEHKey": "string1",
        "DBEDKey": "string2",
        "CustomerId": "customer1",
        "PolicyId": "policy1",
        "WritingCompanyCode": "company1",
        "TransactionType": "type1",
        "TransactionEffectiveDate": "2024-11-20",
        "LineOfBusiness": "business1",
        "PlanType": "plan1",
        "BillingChargeCode": "charge1",
        "BillingChargeCategory": "category1",
        "NonPremiumRecipient": "recipient1",
        "GrossPremium": 100.00,
        "SortOrderNumber": 1
    },
    {
        "DBEHKey": "string3",
        "DBEDKey": "string4",
        "CustomerId": "customer2",
        "PolicyId": "policy2",
        "WritingCompanyCode": "company2",
        "TransactionType": "type2",
        "TransactionEffectiveDate": "2024-11-21",
        "LineOfBusiness": "business2",
        "PlanType": "plan2",
        "BillingChargeCode": "charge2",
        "BillingChargeCategory": "category2",
        "NonPremiumRecipient": "recipient2",
        "GrossPremium": 200.00,
        "SortOrderNumber": 2
    }
]


def direct_bill_entry_post(auth_token, batch_id, username, password):
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Construct the SOAP request body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <DirectBillEntryPost_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <BatchId>{batch_id}</BatchId>
        </DirectBillEntryPost_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Set the headers
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/DirectBillEntryPost",  # SOAPAction for the operation
        "Authorization": f"Bearer {auth_token}"  # Bearer token for authorization
    }

    # Add basic authentication (if necessary)
    auth = (username, password)

    # Send the POST request
    response = requests.post(url, data=soap_body, headers=headers, auth=auth)

    # Check if the request was successful
    if response.status_code == 200:
        print("Request Successful!")
        print(response.text)  # You can parse the response XML if necessary
        return response.text
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)


def direct_bill_entry_post_status(auth_token, batch_id, username, password):
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Construct the SOAP request body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <DirectBillEntryPostStatus_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <BatchId>{batch_id}</BatchId>
        </DirectBillEntryPostStatus_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Set the headers
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/DirectBillEntryPostStatus",  # SOAPAction for the operation
        "Authorization": f"Bearer {auth_token}"  # Bearer token for authorization
    }

    # Add basic authentication (if necessary)
    auth = (username, password)

    # Send the POST request
    response = requests.post(url, data=soap_body, headers=headers, auth=auth)

    # Check if the request was successful
    if response.status_code == 200:
        print("Request Successful!")
        print(response.text)  # You can parse the response XML if necessary
        return response.text
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)


def direct_bill_entry_start(auth_token, dbeh_key, statement_description, statement_for_company_code, statement_date,
                            gldivision_code, gldate, total_gross, total_commission_received, username, password):
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Construct the SOAP request body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <DirectBillEntryStart_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <DirectBillEntryHeader>
            <DBEHKey>{dbeh_key}</DBEHKey>
            <StatementDescription>{statement_description}</StatementDescription>
            <StatementForCompanyCode>{statement_for_company_code}</StatementForCompanyCode>
            <StatementDate>{statement_date}</StatementDate>
            <GLDivisionCode>{gldivision_code}</GLDivisionCode>
            <GLDate>{gldate}</GLDate>
            <TotalGross>{total_gross}</TotalGross>
            <TotalCommissionReceived>{total_commission_received}</TotalCommissionReceived>
          </DirectBillEntryHeader>
        </DirectBillEntryStart_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Set the headers
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/DirectBillEntryStart",  # SOAPAction for the operation
        "Authorization": f"Bearer {auth_token}"  # Bearer token for authorization
    }

    # Add basic authentication (if necessary)
    auth = (username, password)

    # Send the POST request
    response = requests.post(url, data=soap_body, headers=headers, auth=auth)

    # Check if the request was successful
    if response.status_code == 200:
        print("Request Successful!")
        print(response.text)  # You can parse the response XML if necessary
        return response.text
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)


def end_file_chunk(auth_token, doc_stage_id, username, password):
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Construct the SOAP request body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <EndFileChunk_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <DocStageId>{doc_stage_id}</DocStageId>
        </EndFileChunk_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Set the headers
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/EndFileChunk",  # SOAPAction for the operation
        "Authorization": f"Bearer {auth_token}"  # Bearer token for authorization
    }

    # Add basic authentication (if necessary)
    auth = (username, password)

    # Send the POST request
    response = requests.post(url, data=soap_body, headers=headers, auth=auth)

    # Check if the request was successful
    if response.status_code == 200:
        print("Request Successful!")
        print(response.text)  # You can parse the response XML if necessary
        return response.text
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)


def get_attachment_begin(auth_token, doc_a_id, chunk_size, username, password):
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Construct the SOAP request body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetAttachmentBegin_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <DocAId>{doc_a_id}</DocAId>
          <ChunkSize>{chunk_size}</ChunkSize>
        </GetAttachmentBegin_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Set the headers
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/GetAttachmentBegin",  # SOAPAction for the operation
        "Authorization": f"Bearer {auth_token}"  # Bearer token for authorization
    }

    # Add basic authentication (if necessary)
    auth = (username, password)

    # Send the POST request
    response = requests.post(url, data=soap_body, headers=headers, auth=auth)

    # Check if the request was successful
    if response.status_code == 200:
        print("Request Successful!")
        print(response.text)  # You can parse the response XML if necessary
        return response.text
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)


def get_attachment_chunk(auth_token, doc_stage_id, chunk_number, username, password):
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Construct the SOAP request body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetAttachmentChunk_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <DocStageId>{doc_stage_id}</DocStageId>
          <ChunkNumber>{chunk_number}</ChunkNumber>
        </GetAttachmentChunk_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Set the headers
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/GetAttachmentChunk",  # SOAPAction for the operation
        "Authorization": f"Bearer {auth_token}"  # Bearer token for authorization
    }

    # Add basic authentication (if necessary)
    auth = (username, password)

    # Send the POST request
    response = requests.post(url, data=soap_body, headers=headers, auth=auth)

    # Check if the request was successful
    if response.status_code == 200:
        print("Request Successful!")
        print(response.text)  # You can parse the response XML if necessary
        return response.text
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)


def get_attachment_end(auth_token, doc_a_id, doc_stage_id, username, password):
    url = "https://affwsapi.ams360.com/v2/service.asmx"

    # Construct the SOAP request body
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Header>
        <WSAPIAuthToken xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <Token>{auth_token}</Token>
        </WSAPIAuthToken>
      </soap12:Header>
      <soap12:Body>
        <GetAttachmentEnd_Request xmlns="http://www.WSAPI.AMS360.com/v2.0">
          <DocAId>{doc_a_id}</DocAId>
          <DocStageId>{doc_stage_id}</DocStageId>
        </GetAttachmentEnd_Request>
      </soap12:Body>
    </soap12:Envelope>"""

    # Set the headers
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "SOAPAction": "http://www.WSAPI.AMS360.com/v2.0/GetAttachmentEnd",  # SOAPAction for the operation
        "Authorization": f"Bearer {auth_token}"  # Bearer token for authorization
    }

    # Add basic authentication (if necessary)
    auth = (username, password)

    # Send the POST request
    response = requests.post(url, data=soap_body, headers=headers, auth=auth)

    # Check if the request was successful
    if response.status_code == 200:
        print("Request Successful!")
        print(response.text)  # You can parse the response XML if necessary
        return response.text
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)


def call_facade(function_name, **kwargs):
    result = function_name(escaped_token, **kwargs)
    try: 
        parsed_result = xmltodict.parse(result)
        return parsed_result
    except: 
        pass #pprint(result)
    return {}


