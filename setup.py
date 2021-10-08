from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='independent_vector_analysis',
      version='0.1',
      description='Implementation of IVA-G and IVA-L-SOS',
      long_description=readme(),
      long_description_content_type='text/markdown',
      classifiers=[
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
      ],
      keywords='iva, independent vector analysis, bss, multiset analysis',
      url='https://github.com/SSTGroup/independent_vector_analysis',
      author='Isabell Lehmann',
      author_email='isabell.lehmann@sst.upb.de',
      license='LICENSE',
      packages=['independent_vector_analysis'],
      python_requires='>=3.6',
      install_requires=[
          'numpy',
          'scipy',
          'pytest',
          'joblib',
          'tqdm',
      ],
      include_package_data=True,  # to include non .py-files listed in MANIFEST.in
      zip_safe=False)
