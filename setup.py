from setuptools import setup, find_packages
from setuptools import find_namespace_packages

setup(
    name='chunk-py',
    version='1.0.4',
    author='Youpeng Yang',
    author_email='yypeng1999@gmail.com', 
    url='https://github.com/yyp1999/Chunk', 
    description='TransCriptomics pHeotype Unity aNalysis Kit', 
    long_description='Chunk is a computational framework that leverages phenotype information from bulk transcriptomic data to establish robust associations between CCIs and clinical or biological phenotypes in single-cell or spatial transcriptomic data. Chunk hypothesize that intercellular communication mediated by ligand–receptor interactions (LRIs) are key drivers of phenotypic heterogeneity across patients. Guided by diverse phenotypic data (binary, linear, ordinal, survival), Chunk identifies phenotype-associated LRIs from large-scale bulk cohorts and maps them to the single-cell or spatial level to uncover CCI events associated with disease-related phenotypic variation.',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'scikit-learn',
        'scipy',
        'seaborn',
        'statsmodels',
        'plotly',
        'IPython',
        'scanpy',
        'qnorm',
        'torch',
        'numba',
        'anndata'
    ], 
    classifiers=[
        'Development Status :: 5 - Production/Stable',  
        'Intended Audience :: Science/Research',  
        'License :: OSI Approved :: MIT License',  
        'Programming Language :: Python :: 3', 
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',  
    ],
    python_requires='>=3.8', 
    include_package_data=True,  
    package_data={
        'Chunk': ['data/*'], 
    },
    zip_safe=False,  
)