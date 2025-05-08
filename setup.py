from setuptools import setup, find_packages

setup(
    name="resume_ai",
    version="0.1",  
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "torch",
        "transformers",
        "langchain",
        
    ],
    
)