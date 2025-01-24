#****************************************************************************************
#Project ID::               PharmaSUG-2025
#Program Name:              vs.py
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
from datetime import datetime, timedelta

class Metadata(BaseModel):

    VSPERF: str = Field(..., description="Vital Signs Performed, Text coded with NY1, the value must be in ('Y', 'N')")
    VSPERF_REASND: str = Field(..., description="Vital Signs Reason Not Per at Visit, generate only if the abvoe answer is 'No', Text coded with REASND, the value must be in ('Adverse Event', 'Missed in Error', 'Physician Decision', 'Other'), 'Adverse Event' has a 50.0% chance, 'Missed in Error' has a 10.0% chance, 'Physician Decision' has a 30.0% chance, 'Other' has a 10.0% chance")
    VSPERF_REASNDSP: str = Field(..., description="Vital Signs Reason Not Per at Visit Spec, generate only if the above reason is 'Other', free text up to 200 characters long.")
    VSDAT: str = Field(..., description="Vital Signs Date, in format capital dd MMM yyyy.")
    VSTIM: str = Field(..., description="Vital Signs Time, in format HH:nn.")
    VSTIMUNK: str = Field(..., description="Vital Signs Time Unknown, set to blank if it's ongoing, set to 0 if the corresponding time is not missing, otherwise set it to 1, a numeric value in the format 1, the value must be in ('1 = Checked', '0 = Not Checked')")
    TEMP_ORRES: str = Field(..., description="Temperature Result in degrees Celsius, a numeric value in the format 4.1.")
    TEMP_LOC: str = Field(..., description="Temperature Anatomical Location, Text coded with LOC1, the value must be in ('ORAL CAVITY', 'AXILLA', 'TYMPANIC MEMBRANE', 'OTHER'), 'ORAL CAVITY' has a 56.6% chance, 'AXILLA' has a 0.5% chance, 'TYMPANIC MEMBRANE' has a 20.4% chance, 'OTHER' has a 22.5% chance")
    TEMP_LOCSP: str = Field(..., description="Temperature Anatomical Location Specify, free text up to 200 characters long.")
    RESP_ORRES: str = Field(..., description="Respiratory Rate Result, a numeric value in the format 2.")
    SYSBP_ORRES: str = Field(..., description="Systolic Blood Pressure Result, a numeric value in the format 3.")
    DIABP_ORRES: str = Field(..., description="Diastolic Blood Pressure Result, a numeric value in the format 3.")
    PULSE_ORRES: str = Field(..., description="Pulse Result, a numeric value in the format 3.")
    OXYSAT_ORRES: str = Field(..., description="Oxygen Saturation Result, a numeric value in the format 3.")
    VSPOS: str = Field(..., description="Vital Signs Position of Subject, Text coded with POSITION, the value must be in ('PRONE', 'SUPINE', 'SITTING', 'STANDING'), 'PRONE' has a 0.0% chance, 'SUPINE' has a 0.0% chance, 'SITTING' has a 100.0% chance, 'STANDING' has a 0.0% chance")
    VSDESC: str = Field(..., description="Vital Signs Abnormal Findings, write one or two sentence descriptuon about the vital signs test results based on the TEMP_ORRES, RESP_ORRES, SYSBP_ORRES, DIABP_ORRES, PULSE_ORRES and OXYSAT_ORRES.")
    VSCLSIG: str = Field(..., description="Vital Signs Clinical Significance, generate only if the above result is 'Abnormal', Text coded with NY1, the value must be in ('Y', 'N')")

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
        temperature=0.6,
        #top_p=0.1,
        seed=8
      )
    except OpenAIError as e:
      # Handle all OpenAI API errors
      print(f'Error: {e}')

    records=vars(response).get('outlist')
    df=pd.DataFrame([vars(s) for s in records])
    print(df)
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

print('Generating:', 'VS', '...')
df=gen_records(topic='Vital Signs', num_records=100, records_per_batch=10)

# Load the CSV data into a DataFrame
matrix_df = pd.read_csv('./resources/matrix.csv', dtype={'SITE': str}, low_memory=False, encoding='ISO-8859-1')
matrix_df.loc[matrix_df['VISDAT'].isna(), 'VISDAT'] = matrix_df['LAST_VISDAT']
form_df = matrix_df[matrix_df['FORM'] == 'VS']
# Loop through 'SITE', 'PATIENT_NAME', and 'FOLDER' columns
dfs = []
for index, row in form_df.iterrows():
    site = row['SITE']
    patient_name = row['PATIENT_NAME']
    folder = row['FOLDER']
    record_position= row['RECORD_POSITION']
    page_repeatnumber= row['PAGEREPEATNUMBER']
    visdat= datetime.strptime(row['VISDAT'].replace('-', ''), '%d%b%Y')
    rand_df = df.sample(n=1)
    rand_df.insert(0, 'Site', site)
    rand_df.insert(1, 'Patient_Name', patient_name)
    rand_df.insert(2, 'Folder', folder)
    rand_df.insert(3, 'Record_position', record_position)
    rand_df.insert(4, 'Page_repeatnumber', page_repeatnumber)
    if rand_df['VSDAT'].iloc[0] != '':
       rand_df['VSDAT'] = visdat.strftime('%d %b %Y').upper()
    dfs.append(rand_df)

result = pd.concat(dfs, ignore_index=True)
print(result)
result.to_csv("./outputs/csv/vs.csv", sep="|", index=False)
pyreadstat.write_xport(result, "./outputs/sas/vs.xpt", table_name="VS")
