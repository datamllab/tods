from distutils.command.sdist import sdist as sdist_orig
from distutils.errors import DistutilsExecError

from setuptools import setup, find_packages


class install(sdist_orig):

    def run(self):
        try:
            self.spawn(['sh', '.install.sh'])
        except DistutilsExecError:
            self.warn('lost installation script')
        super().run()


setup(name='tods',
    version='0.0.1',
    packages=find_packages(exclude=['contrib', 'docs', 'site', 'test*']),
    cmdclass={
        'install': install
    },
)
