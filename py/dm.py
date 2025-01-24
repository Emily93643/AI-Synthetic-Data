#****************************************************************************************
#Project ID::               PharmaSUG-2025
#Program Name:              dm.py
#Original Author:           Max Ma
#Date Initiated:            20JAN25
#****************************************************************************************

import os
from dotenv import load_dotenv
import instructor
from pydantic import BaseModel, Field
from openai import OpenAI, OpenAIError
from typing import List
import pandas as pd
import pyreadstat
import random

class Metadata(BaseModel):

    BRTHDAT: str = Field(..., description="Date of Birth, in format capital dd MMM yyyy")
    AGE: int = Field(...,  description="Age between 18 and 80, a numeric value in the format 3,  derived from BRTHDAT based on today's date.")
    AGEU: str = Field(..., description="Age Units, Text coded with AGEU, the value must be in ('DAYS', 'HOURS', 'MONTHS', 'WEEKS', 'YEARS'), 'DAYS' has a 0.0% chance, 'HOURS' has a 0.0% chance, 'MONTHS' has a 0.0% chance, 'WEEKS' has a 0.0% chance, 'YEARS' has a 100.0% chance")
    SEX: str = Field(..., description="Sex, Text coded with SEX1, the value must be in ('F', 'M'), 'F' has a 50.0% chance, 'M' has a 50.0% chance")
    CHILDPOT: str = Field(..., description="Childbearing Potential, the value must be in ('Y', 'N') for Female, 'NA' for male.")
    CHILDPOTRSN: str = Field(..., description="Reason not childbearing, must be generated for CHILDPOT=N only, the value must be in ('Post-menopausal', 'Surgically sterile', 'Other') for female,'Post-menopausal' should be generated for female between the ages of 45 and 55, blank for male")
    CHILDPOTSP: str = Field(..., description="If Other Specify, generate it only the corresponding item is 'Other', free text up to 200 characters long.")
    ETHNIC: str = Field(..., description="Ethnicity, Text coded with ETHNIC, the value must be in ('HISPANIC OR LATINO', 'NOT HISPANIC OR LATINO', 'NOT REPORTED', 'UNKNOWN'), 'HISPANIC OR LATINO' has a 2.0% chance, 'NOT HISPANIC OR LATINO' has a 97.0% chance, 'NOT REPORTED' has a 0.5% chance, 'UNKNOWN' has a 0.5% chance")
    RACE_WHITE: str = Field(..., pattern=r"^(0|1)$", description="Race (White), a numeric value in the format 1, the value must be in ('1 = Checked', '0 = Not Checked'), '1' has a 40.0% chance")
    RACE_ASIAN: str = Field(..., pattern=r"^(0|1)$", description="Race (Asian), a numeric value in the format 1, the value must be in ('1 = Checked', '0 = Not Checked'), '1' has a 10.0% chance")
    RACE_BLACK: str = Field(..., pattern=r"^(0|1)$", description="Race (Black or African American), a numeric value in the format 1, the value must be in ('1 = Checked', '0 = Not Checked'), '1' has a 10.0% chance")
    RACE_AMERIND: str = Field(..., pattern=r"^(0|1)$", description="Race (American Indian or Alaska Native), a numeric value in the format 1, the value must be in ('1 = Checked', '0 = Not Checked'), '1' has a 10.0% chance")
    RACE_NHAWII: str = Field(..., pattern=r"^(0|1)$", description="Race (Native Hawaiian Pacific Islanders), a numeric value in the format 1, the value must be in ('1 = Checked', '0 = Not Checked'), '1' has a 10.0% chance")
    RACE_NR: str = Field(..., pattern=r"^(0|1)$", description="Race (Not Reported), a numeric value in the format 1, the value must be in ('1 = Checked', '0 = Not Checked'), '1' has a 10.0% chance")
    RACE_OTHER: str = Field(..., pattern=r"^(0|1)$", description="Race (Other), a numeric value in the format 1, the value must be in ('1 = Checked', '0 = Not Checked'), '1' has a 10.0% chance")
    RACEOTH: str = Field(..., description="If Other Specify, must generate it if the field RACE_OTHER is '1', free text up to 200 characters long, for example: 'Caribbean', 'Aboriginal Australian', 'Samoan' or 'Tongan', etc., if it's null and RACE_OTHER='1', set it to 'Unknown")

class OutList(BaseModel):
    outlist: List[Metadata] = Field(..., description="Multiple records")

# Patch the OpenAI client
load_dotenv()
client = instructor.from_openai(OpenAI(api_key=os.environ.get("OPENAI_API_KEY")))

# Extract structured data from natural language
def gen_data(Topic: str, Nrecords: str) -> OutList:
    prompt = f"Generate exactly {Nrecords} {Topic} records."

    try:
      # Make your OpenAI API request here
      #response = client.chat.completions.create_iterable(
      response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_retries=3, # Retry the request 3 times
        response_model=OutList,
        messages=[
           {"role": "system", "content": "You are a clinical expert specializing in the generation of high-quality synthetic data for clinical trial. Your task is to create synthetic data that strictly conforms to the provided specifications for each field, ensuring realism and logical consistency. The generated data must be accurate, plausible, and appropriate for rigorous testing and validation purposes."},
           {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        #top_p=0.1,
        #seed=8
      )
    except OpenAIError as e:
      # Handle all OpenAI API errors
      print(f'Error: {e}')

    records=vars(response).get('outlist')
    df=pd.DataFrame([vars(s) for s in records])
    #print(df)
    return df

def gen_records(topic, num_records, records_per_batch=10):
    all_records = pd.DataFrame()
    print('Generating:', num_records, 'records', 'for', topic, '...')
    for _ in range(num_records // records_per_batch):
      records=gen_data(Topic=topic, Nrecords=records_per_batch)
      all_records=pd.concat([all_records, records])
      print('Generated:', len(all_records), 'records', 'for', topic)

    # Generate the remaining records if num_records is not a multiple of records_per_batch
    remaining_records = num_records -len(all_records)
    if remaining_records > 0:
       records=gen_data(Topic=topic, Nrecords=remaining_records)
       all_records=pd.concat([all_records, records])
       print('Generated:', len(all_records), 'records', 'for', topic)
    return all_records

print('Generating:', 'DM', '...')
df=gen_records(topic='Demographics', num_records=100, records_per_batch=10)

# Load the CSV data into a DataFrame
matrix_df = pd.read_csv('./resources/matrix.csv', dtype={'SITE': str}, low_memory=False, encoding='ISO-8859-1')
form_df = matrix_df[matrix_df['FORM'] == 'DM']
# Loop through 'SITE', 'PATIENT_NAME', and 'FOLDER' columns
dfs = []
for index, row in form_df.iterrows():
    site = row['SITE']
    patient_name = row['PATIENT_NAME']
    folder = row['FOLDER']
    record_position= row['RECORD_POSITION']
    page_repeatnumber= row['PAGEREPEATNUMBER']
    rand_df = df.sample(n=1)
    rand_df.insert(0, 'Site', site)
    rand_df.insert(1, 'Patient_Name', patient_name)
    rand_df.insert(2, 'Folder', folder)
    rand_df.insert(3, 'Record_position', record_position)
    rand_df.insert(4, 'Page_repeatnumber', page_repeatnumber)
    dfs.append(rand_df)

result = pd.concat(dfs, ignore_index=True)
print(result)
result.to_csv("./outputs/csv/dm.csv", sep="|", index=False)
pyreadstat.write_xport(result, "./outputs/sas/dm.xpt", table_name="DM")
