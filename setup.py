from distutils.core import setup,  find_namespace_packages


setup(
    name='visionlib',
    version='1.0.0',
    packages=find_namespace_packages(include=["albumentations.*","models.*","utils.*"]),
    url='git+https://github.com/tfdeepnet/visionlib.git',
    license='MIT',
    author='Deepak',
    author_email='',
    description='pytorch cnn models for visualization'
    #install_requires=get_install_requirements(INSTALL_REQUIRES, CHOOSE_INSTALL_REQUIRES)
   
)
