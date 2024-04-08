import re

from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


# llama2 || mistral || mixtral
MODEL = "mixtral"


def query_pre_processing(query):
    # Limit the length of the query to 500 characters
    query = query[:500]

    processed_query = query.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

    # Remove all text inside square brackets which contains only uppercase letters, number characters, and commas
    processed_query = re.sub(r"\[[A-Z0-9\,]+\]", '', processed_query)

    # Remove special characters, keeping only letters, numbers, whitespace, and punctuation, and apostrophes
    processed_query = re.sub(r"[^A-Za-z0-9\s\.\,\;\:\'\"\!\?\-]", '', processed_query) 

    while '  ' in processed_query:
        processed_query = processed_query.replace('  ', ' ')

    return processed_query


def query_post_processing(query):
    processed_query = query.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    processed_query = query.replace(',', '')

    while '  ' in processed_query:
        processed_query = processed_query.replace('  ', ' ')

    return processed_query
 

def run_query(query, prompt_template, llm, parser):
    processed_query = query_pre_processing(query)
    print(processed_query)

    prompt = prompt_template.format_prompt(drug_indication=processed_query, associated_conditions=conditions)

    output = llm(prompt.to_string())

    processed_output = query_post_processing(output)

    try:
        parsed_output = parser.parse(processed_output)
    except:
        parsed_output = {"disease": "Failed to parse"}
    
    return parsed_output


response_schemas = [
    ResponseSchema(name="disease", description="The disease associated with the drug indication."),
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

conditions_list = ["Alzheimer's", "Schizophrenia", "Parkinson"]
conditions = ", ".join([f'"{condition}"' for condition in conditions_list])

text_template= """Given the following drug indication text, please indicate if it is associated with one of the following diseases:\n
Drug Indication: "{drug_indication}"\n
The disease property should be one of the following diseases: {associated_conditions}.
If the drug indication is associated with one of the diseases, the disease property should be set to the name of the disease.
If the drug indication does not explicity mention any of the diseases, the disease property should be set to "Other".\n
{format_instructions}"""

prompt_template = PromptTemplate(
    template=text_template,
    input_variables=["drug_indication", "associated_conditions"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

llm = Ollama(model=MODEL, temperature=0)
