from setuptools import setup, find_packages

setup(
    name="multipred_lib",                      # Package name
    version="0.1.0",                           # Initial release version
    author="Marc Sabio-Albert",                        # Replace with your name
    author_email="marcsabioalbert@gmail.com",     # Your contact email
    description="Every function needed to run the analyses of my fMRI study.",  # Short description
    long_description=open("README.md").read(), # Detailed description from README
    long_description_content_type="text/markdown",
    url="https://your.repository.url",         # URL to the project homepage or repo
    packages=find_packages(),                  # Automatically find package folders
    classifiers=[                              # Optional metadata
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',                   # Minimum Python version requirement
    install_requires=[                         # Dependencies
        "numpy>=1.26.4",
        "pandas>=2.2.2",
        "matplotlib>=3.8.4",
        "seaborn>=0.13.2",
        "pingouin>=0.5.5",
        "scipy>=1.13.0",
    ],                                        
)
