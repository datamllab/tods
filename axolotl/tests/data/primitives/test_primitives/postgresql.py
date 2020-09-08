import tempfile
import os
import os.path
import pwd
import re
import shutil
import signal
import subprocess
import time
import typing

import prctl  # type: ignore
import psycopg2  # type: ignore

from d3m import container, utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

from . import __author__, __version__

__all__ = ('PostgreSQLPrimitive',)


Inputs = container.List
Outputs = container.List


class Hyperparams(hyperparams.Hyperparams):
    pass


class PostgreSQLPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which which uses PostgreSQL to compute a value.
    """

    metadata: typing.ClassVar[metadata_base.PrimitiveMetadata] = metadata_base.PrimitiveMetadata({
        'id': 'f23ea340-ce22-4b15-b2f3-e63885f192b3',
        'version': __version__,
        'name': "PostgreSQL operator",
        'keywords': ['test primitive'],
        'source': {
            'name': __author__,
            'contact': 'mailto:author@example.com',
            'uris': [
                'https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/postgresql.py',
                'https://gitlab.com/datadrivendiscovery/tests-data.git',
            ],
        },
        'installation': [{
                'type': metadata_base.PrimitiveInstallationType.UBUNTU,
                'package': 'build-essential',
                'version': '12.4ubuntu1',
            }, {
                'type': metadata_base.PrimitiveInstallationType.UBUNTU,
                'package': 'libcap-dev',
                'version': '1:2.25-1.2',
            }, {
                'type': metadata_base.PrimitiveInstallationType.UBUNTU,
                'package': 'postgresql-10',
                'version': '10.8-0ubuntu0.18.04.1',
            }, {
                'type': metadata_base.PrimitiveInstallationType.UBUNTU,
                'package': 'libpq-dev',
                'version': '10.8-0ubuntu0.18.04.1',
            }, {
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package': 'psycopg2',
                'version': '2.8.2',
            }, {
                # "python-prctl" requires "build-essential" and "libcap-dev". We list it here instead of
                # "setup.py" to not have to list these system dependencies for every test primitive (because
                # we cannot assure this primitive annotation gets installed first).
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package': 'python-prctl',
                'version': '1.7',
            }, {
            'type': metadata_base.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://gitlab.com/datadrivendiscovery/tests-data.git@{git_commit}#egg=test_primitives&subdirectory=primitives'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        }],
        'location_uris': [
            'https://gitlab.com/datadrivendiscovery/tests-data/raw/{git_commit}/primitives/test_primitives/postgresql.py'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        ],
        'python_path': 'd3m.primitives.operator.postgresql.Test',
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.COMPUTER_ALGEBRA,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.OPERATOR,
    })

    def __init__(self, *, hyperparams: Hyperparams, temporary_directory: str = None) -> None:
        super().__init__(hyperparams=hyperparams, temporary_directory=temporary_directory)

        # Initialize it early so that "__del__" has access to these attributes.
        self._connection: psycopg2.connection = None
        self._process: subprocess.Popen = None
        self._postgresql_base: str = None

        self._postgresql_base = tempfile.mkdtemp()
        os.chmod(self._postgresql_base, 0o755)

        self._config_dir = os.path.join(self._postgresql_base, 'conf')
        self._data_dir = os.path.join(self._postgresql_base, 'data')
        self._run_dir = os.path.join(self._postgresql_base, 'run')
        self._config_file = os.path.join(self._config_dir, 'postgresql.conf')

        shutil.copytree('/etc/postgresql/10/main', self._config_dir)
        shutil.copy('/etc/ssl/certs/ssl-cert-snakeoil.pem', os.path.join(self._config_dir, 'server.pem'))
        shutil.copy('/etc/ssl/private/ssl-cert-snakeoil.key', os.path.join(self._config_dir, 'server.key'))
        os.chmod(os.path.join(self._config_dir, 'server.key'), 0o600)

        with open(self._config_file, 'r', encoding='utf8') as config_file:
            config_file_lines = config_file.readlines()
        with open(self._config_file, 'w', encoding='utf8') as config_file:
            for line in config_file_lines:
                line = re.sub('/etc/ssl/certs/ssl-cert-snakeoil.pem', os.path.join(self._config_dir, 'server.pem'), line)
                line = re.sub('/etc/ssl/private/ssl-cert-snakeoil.key', os.path.join(self._config_dir, 'server.key'), line)
                line = re.sub('/var/lib/postgresql/10/main', self._data_dir, line)
                line = re.sub('/etc/postgresql/10/main/pg_hba.conf', os.path.join(self._config_dir, 'pg_hba.conf'), line)
                line = re.sub('/etc/postgresql/10/main/pg_ident.conf', os.path.join(self._config_dir, 'pg_ident.conf'), line)
                line = re.sub('/var/run/postgresql/10-main.pid', os.path.join(self._run_dir, '10-main.pid'), line)
                line = re.sub('/var/run/postgresql/10-main.pg_stat_tmp', os.path.join(self._run_dir, '10-main.pg_stat_tmp'), line)
                line = re.sub('/var/run/postgresql', self._run_dir, line)
                config_file.write(line)

        with open(os.path.join(self._config_dir, 'conf.d', 'local.conf'), 'w', encoding='utf8') as config_file:
            # We disable TCP access.
            config_file.write("listen_addresses = ''\n")

        with open(os.path.join(self._config_dir, 'pg_hba.conf'), 'w', encoding='utf8') as config_file:
            config_file.write("local all all trust\n")

        # 700 is required by PostgreSQL.
        os.mkdir(self._data_dir, mode=0o700)
        os.mkdir(self._run_dir)
        os.mkdir(os.path.join(self._run_dir, '10-main.pg_stat_tmp'))

        if os.getuid() == 0:
            self._username = 'postgres'

            # We have to run PostgreSQL as non-root user.
            shutil.chown(self._data_dir, 'postgres', 'postgres')
            shutil.chown(self._run_dir, 'postgres', 'postgres')
            shutil.chown(os.path.join(self._run_dir, '10-main.pg_stat_tmp'), 'postgres', 'postgres')
            shutil.chown(os.path.join(self._config_dir, 'pg_hba.conf'), 'postgres', 'postgres')
            shutil.chown(os.path.join(self._config_dir, 'pg_ident.conf'), 'postgres', 'postgres')
            shutil.chown(os.path.join(self._config_dir, 'server.key'), 'postgres', 'postgres')
        else:
            self._username = pwd.getpwuid(os.getuid())[0]

        self._init_and_start_database()

    @staticmethod
    def _process_configure() -> None:
        if os.getuid() == 0:
            os.setgid(shutil._get_gid('postgres'))  # type: ignore
            os.setuid(shutil._get_uid('postgres'))  # type: ignore

        # Setting "pdeathsig" will make the process be killed if our process dies for any reason.
        prctl.set_pdeathsig(signal.SIGTERM)

    def _init_and_start_database(self) -> None:
        args = [
            '/usr/lib/postgresql/10/bin/initdb',
            '-D',
            self._data_dir,
            '--locale',
            'en_US.UTF-8',
            '--encoding',
            'UTF-8',
        ]

        try:
            subprocess.run(
                args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                encoding='utf8', check=True, preexec_fn=self._process_configure,
            )
        except subprocess.CalledProcessError as error:
            self.logger.error("Error running initdb: %(stdout)s", {'stdout': error.stdout})
            raise error

        args = [
            '/usr/lib/postgresql/10/bin/postgres',
            '-D',
            self._data_dir,
            '-c',
            'config_file={config_file}'.format(config_file=self._config_file),
        ]

        self._process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, encoding='utf8', preexec_fn=self._process_configure)

        # Waits for 2 seconds.
        connection_error = None
        for i in range(20):
            try:
                self._connection = psycopg2.connect(dbname=self._username, user=self._username, host=self._run_dir)
                break
            except psycopg2.OperationalError as error:
                connection_error = error
                time.sleep(0.1)
        else:
            raise connection_error

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        with self._connection.cursor() as cursor:
            cursor.execute("SELECT 42;")
            return base.CallResult(container.List([cursor.fetchone()[0]], generate_metadata=True))

    def __del__(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

        if self._process is not None and self._process.poll() is None:
            self._process.terminate()

        if self._postgresql_base is not None:
            shutil.rmtree(self._postgresql_base, ignore_errors=True)
