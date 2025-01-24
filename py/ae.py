#****************************************************************************************
#Project ID::               PharmaSUG-2025
#Program Name:              ae.py
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

    AESPID: str = Field(..., description="AE Number, free text up to 4 characters long.")
    AETERM: str = Field(..., description="Reported Term for the Adverse Event, free text up to 200 characters long.")
    AESTDAT: str = Field(..., description="Adverse Event Start Date, in format capital dd MMM yyyy.")
    AESTTIM: str = Field(..., description="Start Time of Adverse Event, in format HH:nn.")
    AESTTIMUNK: str = Field(..., description="Start Time of Adverse Event Unknown, set to blank if it's ongoing, set to 0 if the corresponding time is not missing, otherwise set it to 1, a numeric value in the format 1, the value must be in ('1 = Checked', '0 = Not Checked')")
    AEONGO: str = Field(..., description="Ongoing Adverse Event, Text coded with AEONGO_NY1, the value must be in ('Y', 'N'), 'Y' has a 18.0% chance, 'N' has a 82.0% chance")
    AEENDAT: str = Field(..., description="Adverse Event End Date, it shoul be after the start date or time , set to blank if it's ongoing, in format capital dd MMM yyyy.")
    AEENTIM: str = Field(..., description="End Time of Adverse Event, it shoul be after the start date or time , set to blank if it's ongoing, in format HH:nn.")
    AEENTIMUNK: str = Field(..., description="End Time of Adverse Event Unknown, set to blank if it's ongoing, set to 0 if the corresponding time is not missing, otherwise set it to 1, a numeric value in the format 1, the value must be in ('1 = Checked', '0 = Not Checked')")
    AEOUT: str = Field(..., description="Outcome of Adverse Event, Text coded with OUT, the value must be in ('RECOVERED/RESOLVED', 'RECOVERED/RESOLVED WITH SEQUELAE', 'RECOVERING/RESOLVING', 'NOT RECOVERED/NOT RESOLVED', 'FATAL', 'UNKNOWN'), 'RECOVERED/RESOLVED' has a 50.0% chance, 'RECOVERED/RESOLVED WITH SEQUELAE' has a 10.0% chance, 'RECOVERING/RESOLVING' has a 20.% chance, 'NOT RECOVERED/NOT RESOLVED' has a 10.0% chance, 'FATAL' has a 5.0% chance, 'UNKNOWN' has a 5.0% chance")
    AESEV: str = Field(..., description="AE Severity/Intensity, Text coded with AESEV, the value must be in ('MILD', 'MODERATE', 'SEVERE'), 'MILD' has a 65.0% chance, 'MODERATE' has a 26.0% chance, 'SEVERE' has a 9.0% chance")
    AESCALE: str = Field(..., description="AE Grading Scale, Text coded with AESCALE, the value must be in ('NCI CTC AE v5.0', 'CBER Vaccine Grading'), 'NCI CTC AE v5.0' has a 100.0% chance, 'CBER Vaccine Grading' has a 0.0% chance")
    AETOXGR: str = Field(..., description="AE Standard Toxicity Grade, Text coded with AETOXGR, the value must be in ('1', '2', '3', '4', '5'), must be '5' where AEOUT is 'FATAL', '1' has a 65.0% chance, '2' has a 26.0% chance, '3' has a 8.0% chance, '4' has a 0.5% chance, '5' has a 0.5% chance")
    AEREL: str = Field(..., description="AE Causality Primary, Text coded with AEREL, the value must be in ('NOT RELATED', 'UNLIKELY RELATED', 'POSSIBLY RELATED', 'RELATED'), 'NOT RELATED' has a 10.0% chance, 'UNLIKELY RELATED' has a 10.0% chance, 'POSSIBLY RELATED' has a 10.0% chance, 'RELATED' has a 70.0% chance")
    AEACN: str = Field(..., description="Action Taken with Primary Study Treatment, Text coded with ACN, the value must be in ('DOSE INCREASED', 'DOSE NOT CHANGED', 'DOSE RATE REDUCED', 'DOSE REDUCED', 'DRUG INTERRUPTED', 'DRUG WITHDRAWN', 'NOT APPLICABLE', 'UNKNOWN')")
    AECONTRT: str = Field(..., description="Concomitant or Additional Trtmnt Given, Text coded with AECONTRT_NY1, the value must be in ('Y', 'N'), 'Y' has a 36.0% chance, 'N' has a 64.0% chance")
    AEACNOTH: str = Field(..., pattern="^(0|1)$", description="Other Action Taken - Other, a numeric value in the format 1, the value must be in ('1 = Checked', '0 = Not Checked'), '1' has a 20.0% chance")
    AEACNOTHSP: str = Field(..., description="Other Action Taken Specify, generate it only the corresponding item is 'Other', free text up to 200 characters long.")
    AESER: str = Field(..., description="AE Serious Event, Text coded with AESER_NY1, the value must be in ('Y', 'N'), 'Y' has a 6.0% chance, 'N' has a 94.0% chance")
    AESI: str = Field(..., description="Adverse Event of Special Interest, Text coded with AESI_NY1, the value must be in ('Y', 'N'), 'Y' has a 2.5% chance, 'N' has a 97.5% chance")
    AEDLT: str = Field(..., description="Dose Limiting Toxicity, Text coded with AEDLT_NY1, the value must be in ('Y', 'N'), 'Y' has a 0.0% chance, 'N' has a 100.0% chance")
    AEMED: str = Field(..., description="AE Medically Attended, Text coded with AEMED_NY1, the value must be in ('Y', 'N'), 'Y' has a 25.0% chance, 'N' has a 75.0% chance")

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

print('Generating:', 'AE', '...')
df=gen_records(topic='Adverse Events', num_records=100, records_per_batch=10)

# Load the CSV data into a DataFrame
matrix_df = pd.read_csv('./resources/matrix.csv', dtype=str, low_memory=False, encoding='ISO-8859-1')
form_df = matrix_df[matrix_df['FORM'] == 'AE']
master_df = pd.read_csv('./resources/aeterms.csv')
# Loop through 'SITE', 'PATIENT_NAME', and 'FOLDER' columns
dfs = []
for index, row in form_df.iterrows():
    site = row['SITE']
    patient_name = row['PATIENT_NAME']
    folder = row['FOLDER']
    record_position= row['RECORD_POSITION']
    page_repeatnumber= row['PAGEREPEATNUMBER']
    ic_dsstdat= datetime.strptime(row['IC_DSSTDAT'].replace('-', ''), '%d%b%Y')
    max_visdatn= datetime.strptime(row['LAST_VISDAT'].replace('-', ''), '%d%b%Y')
    rand_df = df.sample(n=1)
    rand_df.insert(0, 'Site', site)
    rand_df.insert(1, 'Patient_Name', patient_name)
    rand_df.insert(2, 'Folder', folder)
    rand_df.insert(3, 'Record_position', record_position)
    rand_df.insert(4, 'Page_repeatnumber', page_repeatnumber)
    if rand_df['AESPID'].iloc[0] != '':
       rand_df['AESPID'] = row['SPID']
    rand_df['AETERM'] = master_df.sample(n=1)['AETERM'].values[0]
    if rand_df['AESTDAT'].iloc[0] != '':
       rand_df['AESTDAT'] = (ic_dsstdat + timedelta(days=random.randint(0, abs((max_visdatn - ic_dsstdat).days)))).strftime('%d %b %Y').upper()
    if rand_df['AEENDAT'].iloc[0] != '':
       rand_df['AEENDAT'] = (datetime.strptime(rand_df['AESTDAT'].iloc[0], '%d %b %Y') + timedelta(days=random.randint(0, 30))).strftime('%d %b %Y').upper()
    dfs.append(rand_df)

result = pd.concat(dfs, ignore_index=True)
print(result)
result.to_csv("./outputs/csv/ae.csv", sep="|", index=False)
pyreadstat.write_xport(result, "./outputs/sas/ae.xpt", table_name="AE")
