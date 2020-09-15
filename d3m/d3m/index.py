import argparse
import contextlib
import json
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import inspect
import logging
import os.path
import pprint
import subprocess
import shutil
import sys
import time
import traceback
import typing
from xmlrpc import client as xmlrpc  # type: ignore

import frozendict  # type: ignore
import pycurl  # type: ignore

from d3m import exceptions, namespace, utils
from d3m.primitive_interfaces import base

__all__ = ('search', 'get_primitive', 'get_primitive_by_id', 'get_loaded_primitives', 'load_all', 'register_primitive', 'discover')

logger = logging.getLogger(__name__)

DEFAULT_INDEX = 'https://pypi.org/pypi'
DEFAULT_OUTPUT = '.'


class _SENTINEL_TYPE:
    __slots__ = ()

    def __repr__(self) -> str:
        return '_SENTINEL'


_SENTINEL = _SENTINEL_TYPE()

_loaded_primitives: typing.Set[typing.Type[base.PrimitiveBase]] = set()


def search(*, primitive_path_prefix: str = None) -> typing.Sequence[str]:
    """
    Returns a list of primitive paths (Python paths under ``d3m.primitives`` namespace)
    for all known (discoverable through entry points) primitives, or limited by the
    ``primitive_path_prefix`` search argument.

    Not all returned primitive paths are not necessary loadable and it is not necessary that
    they are all really pointing to primitive classes, because this method does not try to
    load them yet to determine any of that.

    Parameters
    ----------
    primitive_path_prefix:
        Optionally limit returned primitive paths only to those whose path start with ``primitive_name_prefix``.

    Returns
    -------
    A list of primitive paths.
    """

    if primitive_path_prefix is None:
        primitive_path_prefix = ''

    results = []

    for entry_point in namespace.entry_points():
        primitive_path = 'd3m.primitives.{entry_point_name}'.format(
            entry_point_name=entry_point.name,
        )

        if primitive_path.startswith(primitive_path_prefix):
            results.append(primitive_path)

    # We also go over all loaded primitives to also search over any primitives directly
    # registered using "register_primitive" and not through an entry point.
    for primitive in get_loaded_primitives():
        primitive_path = primitive.metadata.query()['python_path']

        if primitive_path in results:
            continue

        if primitive_path.startswith(primitive_path_prefix):
            results.append(primitive_path)

    return sorted(results)


def get_primitive(primitive_path: str) -> typing.Type[base.PrimitiveBase]:
    """
    Loads (if not already) a primitive class and returns it.

    Parameters
    ----------
    primitive_path:
        A Python path under ``d3m.primitives`` namespace of a primitive.

    Returns
    -------
    A primitive class.
    """

    if not primitive_path:
        raise exceptions.InvalidArgumentValueError("Primitive path is required.")

    #if not primitive_path.startswith('d3m.primitives.'):
    #    raise exceptions.InvalidArgumentValueError("Primitive path does not start with \"d3m.primitives\".")

    path, name = primitive_path.rsplit('.', 1)

    module = importlib.import_module(path)

    return getattr(module, name)


def get_primitive_by_id(primitive_id: str) -> typing.Type[base.PrimitiveBase]:
    """
    Returns a primitive class based on its ID from all currently loaded primitives.

    Parameters
    ----------
    primitive_id:
        An ID of a primitive.

    Returns
    -------
    A primitive class.
    """

    for primitive in get_loaded_primitives():
        if primitive.metadata.query()['id'] == primitive_id:
            return primitive

    raise exceptions.InvalidArgumentValueError("Unable to get primitive '{primitive_id}'.".format(primitive_id=primitive_id))


def get_loaded_primitives() -> typing.Sequence[typing.Type[base.PrimitiveBase]]:
    """
    Returns a list of all currently loaded primitives.

    Returns
    -------
    A list of all currently loaded primitives.
    """

    return list(_loaded_primitives)


def load_all(blocklist: typing.Collection[str] = None) -> None:
    """
    Loads all primitives available and populates ``d3m.primitives`` namespace with them.

    If a primitive cannot be loaded, an error is logged, but loading of other primitives
    continue.

    Parameters
    ----------
    blocklist:
        A collection of primitive path prefixes to not (try to) load.
    """

    if blocklist is None:
        blocklist = []

    for primitive_path in search():
        if any(primitive_path.startswith(blocklist_prefix) for blocklist_prefix in blocklist):
            continue

        try:
            get_primitive(primitive_path)
        except Exception:
            logger.exception("Could not load the primitive: %(primitive_path)s", {'primitive_path': primitive_path})


# TODO: "primitive_path" is not really necessary because it could just be extracted from primitive's metadata.
#       We do not allow them to be different anyway.
def register_primitive(primitive_path: str, primitive: typing.Type[base.PrimitiveBase]) -> None:
    """
    Registers a primitive under ``d3m.primitives`` namespace.

    This is useful to register primitives not necessary installed on the system
    or which are generated at runtime. It is also useful for testing purposes.

    ``primitive_path`` has to start with ``d3m.primitives``.

    Parameters
    ----------
    primitive_path:
        A primitive path to register a primitive under.
    primitive:
        A primitive class to register.
    """

    if not primitive_path:
        raise exceptions.InvalidArgumentValueError("Path under which to register a primitive is required.")

    if not primitive_path.startswith('d3m.primitives.'):
        raise exceptions.InvalidArgumentValueError("Path under which to register a primitive does not start with \"d3m.primitives\".")

    if not inspect.isclass(primitive):
        raise exceptions.InvalidArgumentTypeError("Primitive to register has to be a class.")

    if not issubclass(primitive, base.PrimitiveBase):
        raise exceptions.InvalidArgumentTypeError("Primitive to register is not a subclass of PrimitiveBase.")

    if primitive.metadata.query()['python_path'] != primitive_path:
        raise exceptions.InvalidArgumentValueError("Primitive's \"python_path\" in metadata does not match the path under which to register it: {python_path} vs. {primitive_path}".format(
            python_path=primitive.metadata.query()['python_path'],
            primitive_path=primitive_path,
        ))

    modules_path, name = primitive_path.rsplit('.', 1)
    # We remove "d3m.primitives" from the list of modules.
    modules = modules_path.split('.')[2:]

    if 'd3m.primitives' not in sys.modules:
        import d3m.primitives  # type: ignore

    # Create any modules which do not yet exist.
    current_path = 'd3m.primitives'
    for module_name in modules:
        module_path = current_path + '.' + module_name

        if module_path not in sys.modules:
            try:
                importlib.import_module(module_path)
            except ModuleNotFoundError:
                # This can happen if this module is not listed in any of entry points. But we want to allow
                # registering primitives also outside of existing entry points, so we create a module here.

                # Because we just could not load the module, we know that if the attribute exists,
                # it has to be something else, which we do not want to clobber.
                if hasattr(sys.modules[current_path], module_name):
                    raise ValueError("'{module_path}' is already defined.".format(module_path))

                module_spec = importlib.machinery.ModuleSpec(module_path, namespace.Loader(), is_package=True)
                module = importlib.util.module_from_spec(module_spec)
                module_spec.loader.exec_module(module)

                sys.modules[module_path] = module
                setattr(sys.modules[current_path], module_name, module)

        current_path = module_path

    if hasattr(sys.modules[current_path], name):
        existing_value = getattr(sys.modules[current_path], name)
        # Registering twice the same primitive is a noop.
        if existing_value is primitive:
            return

        # Maybe we are just registering this primitive. But if not...
        if existing_value is not _SENTINEL:
            raise ValueError("'{module}.{name}' is already defined as '{existing_value}'.".format(module=current_path, name=name, existing_value=existing_value))

    setattr(sys.modules[current_path], name, primitive)
    _loaded_primitives.add(primitive)


def discover(index: str = 'https://pypi.org/pypi') -> typing.Tuple[str, ...]:
    """
    Returns package names from PyPi which provide D3M primitives.

    This is determined by them having a ``d3m_primitive`` among package keywords.

    Parameters
    ----------
    index:
        Base URL of Python Package Index to use.

    Returns
    -------
    A list of package names.
    """

    client = xmlrpc.ServerProxy(index)
    hits = client.search({'keywords': 'd3m_primitive'})
    return tuple(sorted({package['name'] for package in hits}))


def download_files(primitive_metadata: frozendict.FrozenOrderedDict, output: str, redownload: bool) -> None:
    last_progress_call = None

    def curl_progress(download_total: int, downloaded: int, upload_total: int, uploaded: int) -> None:
        nonlocal last_progress_call

        # Output at most once every 10 seconds.
        now = time.time()
        if last_progress_call is None or now - last_progress_call > 10:
            last_progress_call = now

            print("Downloaded {downloaded}/{download_total} B".format(
                downloaded=downloaded,
                download_total=download_total,
            ), flush=True)

    for installation_entry in primitive_metadata.get('installation', []):
        if installation_entry['type'] not in ['FILE', 'TGZ']:
            continue

        # We store into files based on digest. In this way we deduplicate same
        # files used by multiple primitives.
        output_path = os.path.join(output, installation_entry['file_digest'])

        if installation_entry['type'] == 'FILE':
            if os.path.isfile(output_path) and not redownload:
                print("File for volume {type}/{key} for primitive {python_path} ({primitive_id}) already exists, skipping: {file_uri}".format(
                    python_path=primitive_metadata['python_path'],
                    primitive_id=primitive_metadata['id'],
                    type=installation_entry['type'],
                    key=installation_entry['key'],
                    file_uri=installation_entry['file_uri'],
                ), flush=True)
                continue
        elif installation_entry['type'] == 'TGZ':
            if os.path.isdir(output_path) and not redownload:
                print("Directory for volume {type}/{key} for primitive {python_path} ({primitive_id}) already exists, skipping: {file_uri}".format(
                    python_path=primitive_metadata['python_path'],
                    primitive_id=primitive_metadata['id'],
                    type=installation_entry['type'],
                    key=installation_entry['key'],
                    file_uri=installation_entry['file_uri'],
                ), flush=True)
                continue

        # Cleanup.
        if os.path.isdir(output_path):
            shutil.rmtree(output_path)
        elif os.path.exists(output_path):
            os.remove(output_path)

        print("Downloading file for volume {type}/{key} for primitive {python_path} ({primitive_id}): {file_uri}".format(
            python_path=primitive_metadata['python_path'],
            primitive_id=primitive_metadata['id'],
            type=installation_entry['type'],
            key=installation_entry['key'],
            file_uri=installation_entry['file_uri'],
        ), flush=True)

        output_file_obj: typing.BinaryIO = None
        output_tar_process = None

        try:
            if installation_entry['type'] == 'FILE':
                output_file_obj = open(output_path, 'wb')
            elif installation_entry['type'] == 'TGZ':
                os.makedirs(output_path, mode=0o755, exist_ok=True)
                output_tar_process = subprocess.Popen(['tar', '-xz', '-C', output_path], stdin=subprocess.PIPE)
                output_file_obj = typing.cast(typing.BinaryIO, output_tar_process.stdin)

            hash = hashlib.sha256()
            downloaded = 0
            start = time.time()

            def write(data: bytes) -> None:
                nonlocal hash
                nonlocal downloaded

                hash.update(data)
                downloaded += len(data)

                output_file_obj.write(data)

            while True:
                try:
                    with contextlib.closing(pycurl.Curl()) as curl:
                        curl.setopt(curl.URL, installation_entry['file_uri'])
                        curl.setopt(curl.WRITEFUNCTION, write)
                        curl.setopt(curl.NOPROGRESS, False)
                        curl.setopt(curl.FOLLOWLOCATION, True)
                        curl.setopt(getattr(curl, 'XFERINFOFUNCTION', curl.PROGRESSFUNCTION), curl_progress)
                        curl.setopt(curl.LOW_SPEED_LIMIT, 30 * 1024)
                        curl.setopt(curl.LOW_SPEED_TIME, 30)
                        curl.setopt(curl.RESUME_FROM, downloaded)

                        curl.perform()
                        break

                except pycurl.error as error:
                    if error.args[0] == pycurl.E_OPERATION_TIMEDOUT:
                        # If timeout, retry/resume.
                        print("Timeout. Retrying.", flush=True)
                    else:
                        raise

            end = time.time()

            print("Downloaded {downloaded} B in {seconds} second(s).".format(
                downloaded=downloaded,
                seconds=end - start,
            ), flush=True)

            if output_tar_process is not None:
                # Close the input to the process to signal that we are done.
                output_file_obj.close()
                output_file_obj = None

                # Wait for 60 seconds to finish writing everything out.
                if output_tar_process.wait(60) != 0:
                    raise subprocess.CalledProcessError(output_tar_process.returncode, output_tar_process.args)
                output_tar_process = None

            if installation_entry['file_digest'] != hash.hexdigest():
                raise ValueError("Digest for downloaded file does not match one from metadata. Metadata digest: {metadata_digest}. Computed digest: {computed_digest}.".format(
                    metadata_digest=installation_entry['file_digest'],
                    computed_digest=hash.hexdigest(),
                ))

        except Exception:
            # Cleanup.
            if output_tar_process is not None:
                try:
                    output_tar_process.kill()
                    output_tar_process.wait()
                    output_file_obj = None
                except Exception:
                    # We ignore errors cleaning up.
                    pass
            if os.path.isdir(output_path):
                shutil.rmtree(output_path)
            elif os.path.exists(output_path):
                os.remove(output_path)

            raise

        finally:
            if output_file_obj is not None:
                output_file_obj.close()


# TODO: Add more ways to search for primitives (by name, keywords, etc.).
# TODO: Allow displaying results with more than just a primitive path.
def search_handler(arguments: argparse.Namespace) -> None:
    for primitive_path in search(primitive_path_prefix=getattr(arguments, 'prefix', None)):
        print(primitive_path)


def discover_handler(arguments: argparse.Namespace) -> None:
    for package_name in discover(index=getattr(arguments, 'index', DEFAULT_INDEX)):
        print(package_name)


def describe_handler(arguments: argparse.Namespace) -> None:
    output_stream = getattr(arguments, 'output', sys.stdout)

    has_errored = False

    for primitive_path in arguments.primitives:
        if getattr(arguments, 'list', False):
            print(primitive_path, file=output_stream)

        try:
            try:
                primitive = get_primitive(primitive_path)
            except Exception:
                primitive = None

            if primitive is None:
                load_all()
                primitive = get_primitive_by_id(primitive_path)
        except Exception as error:
            if getattr(arguments, 'continue', False):
                traceback.print_exc(file=output_stream)
                print(f"Error loading primitive: {primitive_path}", file=output_stream)
                has_errored = True
                continue
            else:
                raise Exception(f"Error loading primitive: {primitive_path}") from error

        try:
            # Using "to_json_structure" and not "to_internal_json_structure" because
            # it is not indented that this would be parsed back directly, but just used
            # to know where to find the primitive (using "installation" section).
            primitive_description = primitive.metadata.to_json_structure()

            if getattr(arguments, 'print', False):
                pprint.pprint(primitive_description, stream=output_stream)

            else:
                json.dump(
                    primitive_description,
                    output_stream,
                    indent=(getattr(arguments, 'indent', 2) or None),
                    sort_keys=getattr(arguments, 'sort_keys', False),
                    allow_nan=False,
                )  # type: ignore
                output_stream.write('\n')
        except Exception as error:
            if getattr(arguments, 'continue', False):
                traceback.print_exc(file=output_stream)
                print(f"Error describing primitive: {primitive_path}", file=output_stream)
                has_errored = True
                continue
            else:
                raise Exception(f"Error describing primitive: {primitive_path}") from error

    if has_errored:
        sys.exit(1)


def download_handler(arguments: argparse.Namespace) -> None:
    for primitive_path in search(primitive_path_prefix=getattr(arguments, 'prefix', None)):
        try:
            primitive_class = get_primitive(primitive_path)
        except Exception:
            logger.exception("Could not load the primitive: %(primitive_path)s", {'primitive_path': primitive_path})
            continue

        try:
            download_files(primitive_class.metadata.query(), getattr(arguments, 'output', DEFAULT_OUTPUT), getattr(arguments, 'redownload', False))
        except Exception:
            logger.exception("Error downloading files for: %(primitive_path)s", {'primitive_path': primitive_path})


def main(argv: typing.Sequence) -> None:
    # We have to disable importing while type checking because it makes
    # an import cycle in mypy which makes many typing errors.
    if not typing.TYPE_CHECKING:
        # Importing here to prevent import cycle.
        from d3m import cli

        logging.basicConfig()

        logger.warning("This CLI is deprecated. Use \"python3 -m d3m index\" instead.")

        parser = argparse.ArgumentParser(description="Explore D3M primitives.")
        cli.primitive_configure_parser(parser)

        arguments = parser.parse_args(argv[1:])

        cli.primitive_handler(arguments, parser)


if __name__ == '__main__':
    main(sys.argv)
