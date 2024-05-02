import yaml
import json
import os 

def read_yaml_files(directory):
    yaml_data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                yaml_content = yaml.safe_load(file)
                yaml_data[filename] = yaml_content
    return yaml_data


def create_JSON(jobs_yaml_directory = "../../resources"):
    """
    Returns: 
    dic : 
    {"job_name" :  "payload" }
    """
    # Example usage:
    yaml_dict = read_yaml_files(jobs_yaml_directory)
    payloads = {}
    
    for file,elems in yaml_dict.items():
        for job_key, job in elems['resources'].get('jobs').items():
        
            payloads[job.get('name')] =  job
    
    return payloads


#print(create_JSON())
current_directory = os.path.dirname(os.path.realpath(__file__))
target_path = os.path.abspath(os.path.join(current_directory, "..", ".."))
target_path = os.path.abspath(os.path.join(target_path, "resources"))

print(create_JSON(jobs_yaml_directory = target_path))
