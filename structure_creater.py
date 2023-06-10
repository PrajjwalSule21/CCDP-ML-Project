import os

def project_structure_creator():
    """
    This Function will help you to create entire Project Structure. 
    It does not need any argument just simply run this function inside your project directory.
    All the things are done by this function itself.
    
    -- Project Structure --
    
    > artifacts
    > conf
        >> main.yaml
    > data
    > logs
    > src
        >> components
            >>>  __init__.py
            >>> data_ingestion.py
            >>> data_transformation.py
            >>> model_trainer.py
            
        >> pipeline
            >>>  __init__.py
            >>> predict_pipeline.py
            >>> train_pipeline.py
            
        >> __init__.py
        >> exception.py
        >> logger.py
        >> utils.py
    > static
        >> CSS
            >>> style.css
    > templates
        >> index.html
    > aap.py
    > requirements.txt
    > setup.py
        
        
     
    """
    path = os.getcwd()
    
    # Create data direcotry
    data = os.path.join(path, 'data')
    os.mkdir(data)
    
    # Create log directory
    logs = os.path.join(path, 'logs')
    os.mkdir(logs)
    
    # Create artifacts directory
    artifacts = os.path.join(path, 'artifacts')
    os.mkdir(artifacts)

    # Create conf directory
    conf = os.path.join(path, 'conf')
    os.mkdir(conf)

    os.chdir(os.path.join('conf'))
    main = open('main.yaml', 'x')
    
    # Create some core files of project
    os.chdir("..")
    core_files = ['app.py', 'setup.py', 'requirements.txt']
    for filename in core_files:
        core = open(filename, 'x')
    
    # Create templates directory
    templates = os.path.join(path, 'templates')
    os.mkdir(templates)

    
    # Create the file of templates directory
    os.chdir(os.path.join('templates'))
    index = open('index.html', 'x')
    
    
    # Create static directory and static files
    static = os.path.join(path, 'static')
    os.mkdir(static)
    
    os.chdir("..")
    os.chdir(os.path.join('static'))
    os.mkdir('CSS')
    
    
    os.chdir(os.path.join('CSS'))
    f = open('style.css', 'x')
    
    
    # Create src directory
    src = os.path.join(path, 'src')
    os.mkdir(src)
    
    os.chdir("..")
    os.chdir("..")
    os.chdir(os.path.join('src'))
    
    # Create files of src directory
    src_filename = ['__init__.py', 'exception.py', 'logger.py', 'utils.py']
    for filename in src_filename:
        src_files = open(filename, 'x') 
    
    # Create the directories of src directory
    components = os.path.join(os.getcwd(), 'components')
    os.mkdir(components)
    
    
    pipeline = os.path.join(os.getcwd(), 'pipeline')
    os.mkdir(pipeline)
    
    
    os.chdir(os.path.join('components'))
    comp_filenames = ['__init__.py', 'data_ingestion.py', 'data_transformation.py', 'model_trainer.py']
    for filename in comp_filenames:
        components_files = open(filename, 'x') 
   

    os.chdir("..") 
    os.chdir(os.path.join('pipeline'))
    pipe_filenames = ['__init__.py', 'predict_pipeline.py', 'train_pipeline.py']
    for filename in pipe_filenames:
        pipeline_files = open(filename, 'x')



if __name__ == "__main__":
    project_structure_creator()