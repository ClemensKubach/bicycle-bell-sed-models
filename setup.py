import setuptools
import os

readmePath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.md')
with open(readmePath, "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='bicycle-bell-sed-models',
    version='0.0.1',
    author='Clemens Kubach',
    author_email='clemens.kubach@gmail.com',
    description='Package of neural network models for detecting the sound event of a ringing bicycle bell.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/ClemensKubach/bicycle-bell-sed-models.git',
    project_urls = {
        "Bug Tracker": "https://github.com/ClemensKubach/bicycle-bell-sed-models/issues"
    },
    license='MIT',
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src", exclude=['visualizations']),
    install_requires=['tensorflow', 'tensorflow_io', 'tensorflow_hub', 'numpy'],
    python_requires=">=3.6",
)