from setuptools import find_packages, setup

setup(
    name='nidpack',
    packages=find_packages(),
    version='0.1',
    description='Basic example of a package in Python bringing a solution to the KDD99 Cup data (detection of network intrusions)',
    url='https://github.com/gatrikh/project_m05',
    author='Lucas Devanthery, Gary Folli',
    author_email='lucas.devanthery@etu.unidistance.ch, gary.folli@etu.unidistance.ch',
    license='MIT',

    install_requires=[
        'setuptools',
        'pandas==1.3.3',
        'matplotlib==3.4.3',
        'pytest==6.2.5',
        'tqdm==4.62.3',
        'numpy==1.21.2',
        'ipython==7.28.0',
        'scikit_learn==0.24.1'
    ],
)
