from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    '''
    A function that takes a file path and returns a list of requirements by reading the file.

    :param file_path: A string representing the file path of the requirements file.
    :return: A list of strings representing the requirements.
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()  # Read the requirements from the file
        requirements = [req.replace("\n", "") for req in requirements]  # Remove the new line character from the requirements
        
        if '-e .' in requirements:
            requirements.remove('-e .')  # Remove the development mode flag from the requirements if present
    
    return requirements


setup(
    name='EfficientFrontierOptimisation',  # The name of the package
    version='0.1',  # The version of the package
    author='ShubhanKamat',  # The author of the package
    author_email='shubhan.kamat@gmail.com',  # The email address of the author
    packages=find_packages(),  # Find all packages in the directory
    install_requires=get_requirements('requirements.txt')  # Get the requirements from the requirements file
)


     