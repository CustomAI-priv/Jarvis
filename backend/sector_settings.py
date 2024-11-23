from pydantic import BaseModel

class SectorSettings(BaseModel):
    """settings for a sector and company in particular"""

    # define the company name
    company_name: str = "Inditex"

    # define the company description
    company_description: str = ""

    # define the company website
    company_website: str = ""

    # define the sector of work for the company 
    company_sector: str = ""
    