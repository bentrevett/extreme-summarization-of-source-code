import json
import os

DATA_DIR = 'data/json/'

START_TOKEN = '<title_start>'
END_TOKEN = '<title_end/>'

PROJECTS = ['cassandra', 'elasticsearch', 'gradle', 'hadoop-common', 'hibernate-orm', 'intellij-community', 'libgdx', 'liferay-portal', 'presto', 'spring-framework', 'wildfly']

def files_to_data(DIR, FILE):
    """
    DIR is the base directory containing all the json files
    FILES is the filename of the json file you want to get the data from
    """
    data = []
    with open(os.path.join(DIR, FILE), 'r') as r:
        project = json.load(r)
        for method in project:
            assert type(method['filename']) is str
            assert type(method['name']) is list
            assert type(method['tokens']) == list
            method_name = [START_TOKEN] + method['name'] + [END_TOKEN] 
            method_body = [x.lower() for x in method['tokens'] if (x != '<id>' and x != '</id>')]
            while '%self%' in method_body:
                self_idx = method_body.index('%self%')
                method_body = method_body[:self_idx] + method['name'] + method_body[self_idx+1:]
        
            data.append({'name':method_name, 'body':method_body})

    return data

for project in PROJECTS:

    print(f'Project: {project}')

    train_file = f'{project}_train_methodnaming.json'
    test_file = f'{project}_test_methodnaming.json'

    train_data = files_to_data(DATA_DIR, train_file)
    test_data = files_to_data(DATA_DIR, test_file)

    print(f'Training examples: {len(train_data)}')
    print(f'Testing examples: {len(test_data)}')
    
    with open(f'data/{project}_train.json', 'w') as w:
        for example in train_data:
            json.dump(example, w)
            w.write('\n')

    with open(f'data/{project}_test.json', 'w') as w:
        for example in test_data:
            json.dump(example, w)
            w.write('\n')