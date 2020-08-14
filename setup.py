from setuptools import find_packages, setup

setup(
    name='buschwerkzeug',
    #package_dir={'': 'src'},
    packages=find_packages(),
    version='0.1.1',
    description='Buschwerkzeug',
    author='Martin',
    license='',
    install_requires=[ 
        'numba==0.48.0',
        'click',
        'soundfile',
        'scipy',
        'pandas',
        'scikit-image',
        'xlrd',
        'tensorflow==1.15.3',
        'librosa',
        #'keras',
        'deco',
        'dill',
        ],
    dependency_links=[ 'https://github.com/atomfried/noisereduce.git' ]
)

