from setuptools import setup, find_packages
# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "empower_functions/README.md").read_text()

setup(
    name='empower_functions',  # This is the package name
    version='0.1.3',
    packages=find_packages(),  # This will find the `empower_functions` directory
    install_requires=[
        "jinja2",
        "llama-cpp-python[server]",
        "pydantic"
        # Add your dependencies here
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
)
