from setuptools import setup, find_packages


with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    install_requires = [line.strip() for line in f]

setup(
    name='sroom',
    version='0.1.0',
    description='Simulate a smart room by voice commands',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/thanhtvt/SmartRoomSimulator',
    packages=find_packages(),
    install_requires=install_requires,
    python_requires='>=3.7',
)
