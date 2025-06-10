from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Sarcasm detection in audio speech using CNN and f0 variations',
    author='Group1',
    license='',
		install_requires=['ujson', 'simplejson', 'numpy', 'resampy', 'tensorflow', 'tf_slim', 'six', 'soundfile', 'librosa', 'imbalanced-learn', 'requests', 'rich', 'pyyaml', 'gdown', 'click', 'torch', 'torchaudio', 'deepfilternet']
)

