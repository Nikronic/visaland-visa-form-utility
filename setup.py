from setuptools import setup, find_packages
from pathlib import Path


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='vizard', version='0.1.0-alpha', packages=find_packages(),
      description='Vizard: Visa chance predictor, powered by AI!',
      author='Nikan Doosti',
      author_email='nikan.doosti@outlook.com',
      long_description=long_description,
      long_description_content_type='text/markdown',
      )
