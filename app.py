from utils import *
from dotenv import load_dotenv
import os

#
# IBM Session Token 
#
load_dotenv()
TOKEN = os.getenv("MY_TOKEN")

if TOKEN is None:
    raise ValueError("MY_TOKEN not found in environment variables.")
else:
    print('**IBM TOKEN LOADED**')

#
# Load System Instructions
#

instructions = load_instructions('instructions.json')
system_instruct_0 = instructions.get('system_instruct_0') #granite-instruct
system_instruct_1 = instructions.get('system_instruct_1') #granite-code
system_instruct_2 = instructions.get('system_instruct_2') #granite-code


#print(system_instruct_0)

#
# Document Chunking 
#

source_file_path = r'assets/sample_docs/Environmental_imapact_of_data_centers_1.pdf' #from streamlit #debugged/runs
document_text = document_processing(source_file_path)
#print(f'***Document Text: {document_text}***')


#
# Call granite-3-8b-instruct To Extract Themes/Keywords
#

themes_relationships = generate_text_granite_instruct(system_instruct_0,
                                                      document_text,
                                                      TOKEN)
print(f'***granite-3-8b-instruct: THEMES/KEYWORDS/RELATIONSHOPS: {themes_relationships}***')
print("++++++++++++++++++++++++++")
print(type(themes_relationships))
print("++++++++++++++++++++++++++")

#
# Add special delimiter token
#

delimited_node_edge_str = special_delim_token(themes_relationships)

#
# Call granite-34b-code-instruct to Generate Cypher query for Neo4j Query (query draft)
#

cypher_query_draft = generate_code_granite_instruct(delimited_node_edge_str, 
                                                    system_instruct_1,
                                                    TOKEN)

print(f'***granite-34b-code-instruct: CYPHER QUERY DRAFT :: {cypher_query_draft}***')

#
# Call granite-34b-code-instruct to Generate Cypher query for Neo4j Query (fix any issues query)
#

cypher_query_final = generate_code_granite_instruct(cypher_query_draft, 
                                                    system_instruct_2,
                                                    TOKEN)

print(f'***granite-34b-code-instruct: CYPHER QUERY FINAL :: {cypher_query_final}***')

# #
# # Connect to Neo4j and input Knoweldge Graph Nodes and Edges
# #

#query_neo(cypher_query_final)  #@TODO add try, to catch errors and re-run throught LLM for new query

retries = 4  
attempt = 0
success = False

while attempt < retries:
    try:
        # Generate the Cypher query
        cypher_query_final = generate_code_granite_instruct(cypher_query_draft, 
                                                    system_instruct_2,
                                                    TOKEN) 
        
        # Run the query
        query_neo(cypher_query_final)  
        print("Query executed successfully.")
        success = True  
        break

    except Exception as e:
        print(f"Error: {e}. Retrying... (Attempt {attempt + 1}/{retries})")
        
        cypher_query_final = generate_code_granite_instruct(cypher_query_draft, 
                                                    system_instruct_2,
                                                    TOKEN) 

        attempt += 1

        if attempt == retries:
            print("Max retries reached. Query failed.")
            raise Exception("Query failed after multiple attempts.")


if not success:
    print("***==Attempting truncated cypher ...==**")

    try:
        truncated_cypher = clean_cypher_query(cypher_query_draft)
        query_neo(truncated_cypher)
    except Exception as e:
        print(f'Error:{e}')