from setuptools import setup, find_packages

setup(
    name='RSSMFold',
    version='0.1',
    packages=find_packages(),
    url='',
    license='MIT licence',
    author='Zichao Yan',
    author_email='zichao.yan@mail.mcgill.ca',
    description='',
    entry_points={
        'console_scripts': [
            'RSSMFold = RSSMFold.single_seq_rssm:run',
        ], },
)
