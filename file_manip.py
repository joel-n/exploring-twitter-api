import json

def append_ids_to_file(file_name: str, result: list) -> None:
    """Append a file with the 'id' attribute of a list of objects.
    Use case is dehydration of tweets or users, e.g., write the
    tweetIDs of a list of tweets.
    
    Args:
        - file_name: name of the file to append
        - result: list of objects to append, retrieved
        from expansions.flatten(result_page)    
        
    No return value.
    """
    with open(file_name, 'a+') as filehandle:
        for element in result:
                filehandle.write('%s\n' % element['id'])    
                
                
def append_objs_to_file(file_name: str, result: list) -> None:
    """Append a file with the objects in result.
    Use case: append a file with a flattened search
    query (tweets, users etc.).
    
    Args:
        - file_name: name of the file to append
        - result: list of objects to append, retrieved
        from expansions.flatten(result_page)
        
    No return value.
    """
    with open(file_name, 'a+') as filehandle:
        for obj in result:
            filehandle.write('%s\n' % json.dumps(obj))
                
                
def read_file(file_name: str) -> list[dict]:
    """Returns a list of the contents in a .jsonl-file.
    
    Args:
        - file_name: name of the file to read
        
    Returns:
        - objs: list of json objects in the file
    """
    objs = []
    with open(file_name, 'r') as filehandle:
        for obj in filehandle:
            objs.append(json.loads(obj))
    return objs

    
def file_generator(file_name: str):
    """Returns a generator to the contents in a file.
    
    Args:
        - file_name: name of the file to read
        
    Yields:
        - obj: a json objects in the file
    """
    with open(file_name, 'r') as filehandle:
        for obj in filehandle:
            yield json.loads(obj)
    
    
def write_text(file_name: str, text: str) -> None:
    """Writes a string to a file.
    
    Args:
        - file_name: name of the file to read
        - text: string to write to file
    
    No return value.    
    """
    with open(file_name, 'a+', encoding='utf-8') as filehandle:
        filehandle.write('%s\n' % text)
    return