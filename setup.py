import setuptools

with open('README.md','r') as f:
    long_description=f.read()

setuptools.setup(
    name='spaTrack',
    version='0.1.1',
    description='an algorithm that combine both gene expression and spot location to inference cell trajectory',
    long_description=long_description,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
    ],
    author='yzf072',
    author_email='yzfff6@outlook.com',
    url='https://github.com/yzf072/spaTrack',
    packages=setuptools.find_packages(),
    install_requires=[
        'pot>=0.9.0',
        'scanpy>=1.9.3',
        'plotly>=5.15.0',
        'ipywidgets>=8.0.7',
        'pygam>=0.8.0',
        'networkx>=3.0',
        'numpy==1.22.4',
        'scipy==1.9.1',
        'torch>=2.0.1',
        'pandas==1.4.3',
        'nbformat>=4.2.0',
    ],
    python_requires='>=3.7'
)
