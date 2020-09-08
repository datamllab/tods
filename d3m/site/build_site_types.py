"""
Constructs sites for ``semantic types``, site, which hierarchically displays all types, and site, which
lists all available versions of schemas.

Sites are placed under ``types`` folder inside ``public`` folder, which should exist at the root of the repository.
"""

import json
import os
import typing
from shutil import copyfile

from pyquery import PyQuery
from yattag import Doc

PREFIX = 'https://metadata.datadrivendiscovery.org/types/'

types = {}


def cycle_detection(url: str, past_urls: typing.List[str]) -> None:
    """
    Detects cycle in semantic types' hierarchy.

    Also checks if referenced urls in ``parents`` exist.

    Parameters
    ----------
    url : str
        URL of the semantic type that is to be analyzed.
    past_urls : typing.List[str]
        List of previously called urls.
    """

    global types

    if url not in types:
        raise Exception("Cannot find referenced semantic type '{url}'".format(url=url))
    if url in past_urls:
        raise Exception("Cycle in semantic types hierarchy. Cycle: '{cycle}'".format(
            cycle=(' -> '.join(past_urls + [url]))
        ))

    for parent in types[url]['parents']:
        cycle_detection(parent, past_urls + [url])


def template(tag, line):
    """
    Generates HTML base for the site.

    Yields the result, so HTML end brackets (e.g. ``</div>``) are not closed.

    Usage::

        for temp in template(tag, line):
            ...

    Parameters
    ----------
    tag : yattag.tag
        ``tag`` from the ``yattag`` module.
    line : yattag.line
        ``line`` from the ``yattag`` module.

    Returns
    -------
        Element of the ``yattag`` module representing container of the page.
    """

    global types

    with tag('html'):
        with tag('head'):
            line('title', "D3M Metadata")
            line('meta', '', charset='utf-8')
            line('meta', '', name='viewport', content='width=device-width, initial-scale=1')
            line('link', '', rel='stylesheet', href='/schema-org.css')
        with tag('body'):
            with tag('div', id='container'):
                with tag('div', id='intro'):
                    with tag('div', id='pageHeader'):
                        with tag('div', klass='wrapper'):
                            with tag('div', id='sitename'):
                                with tag('h1'):
                                    line('a', "metadata.datadrivendiscovery.org", href='/')
            with tag('div', id='selectionbar'):
                with tag('div', klass='wrapper'):
                    with tag('ul'):
                        with tag('li'):
                            line('a', "Types", href='/types')
                        with tag('li'):
                            line('a', "Schemas", href='/devel')
            with tag('div', id='mainContent'):
                yield


def construct_types(site, parent: str) -> None:
    """
    Constructs hierarchy displayed semantic types at ``/types/`` path.

    More specifically, constructs list (HTML ``<ul>``) of semantic types that have ``parent`` for ancestor.

    Parameters
    ----------
    site
        ``site`` from the ``yattag`` module.
    parent : str
        URL of the parent.
    """

    global types

    with site.tag('ul'):
        for url in sorted(types, key=lambda key: types[key]['label']):
            if parent in types[url]['parents'] or len(parent) == 0 and len(types[url]['parents']) == 0:
                with site.tag('li'):
                    site.line('a', types[url]['label'], href=url)
                    construct_types(site, url)


def construct_breadcrumbs(site, ancestors: typing.List) -> None:
    """
    Constructs breadcrumbs in the page.

    E.g. if ``ancestors`` equals [c, b, a], then breadcrumbs should be "... > c > b > a".

    Parameters
    ----------
    site
        ``site`` from the ``yattag`` module.
    ancestors : typing.List
        URLs of elements that are to be displayed in breadcrumbs.
    """

    global types

    parents = types[ancestors[0]]['parents']
    for parent in parents:
        construct_breadcrumbs(site, [parent] + ancestors)

    if len(parents) == 0:
        with site.tag('span'):
            for url in ancestors:
                site.line('a', types[url]['label'], href=url)
                site.line('span', " > ", klass='hide-last')
        site.stag('br')


def define_external_type(url: str) -> None:
    """
    Used for adding types from ``schema.org`` domain to ``types``. Fetches ``url`` and looks for parents,
    which are also recursively added to ``types``.

    Parameters
    ----------
    url : str
        URL of the type. Should be from ``schema.org`` domain.
    """

    global types

    if url in types:
        return

    types[url] = {
        'label': url[url.rfind('/') + 1:],
        'description': '',
        'parents': []
    }
    candidates = PyQuery(url)('link')

    for i in range(len(candidates)):
        link = candidates.eq(i)
        if link.attr('property') == 'rdfs:subClassOf':
            parent = link.attr('href')

            if len(parent) > 0:
                if parent not in types[url]['parents']:
                    types[url]['parents'].append(parent)
                define_external_type(parent)


def main() -> None:
    global types

    os.makedirs('public/types', 0o755, exist_ok=True)
    copyfile('site/schema-org.css', 'public/schema-org.css')

    schema = json.load(open('d3m/metadata/schemas/v0/definitions.json'))

    # Filling "types".
    for semantic_type in schema['definitions']['semantic_types']['items']['anyOf']:
        if 'enum' not in semantic_type:
            continue

        url = semantic_type['enum'][0]

        description = semantic_type.get('description', '')
        parents = semantic_type.get('parents', [])
        label = url[url.rfind('/') + 1:]

        if not isinstance(parents, list):
            raise Exception("This semantic type does not have type 'list' for 'parents': {url}".format(url=url))

        # Defining parents from 'schema.org'.
        for parent in parents:
            if '//schema.org' in parent:
                define_external_type(parent)

        if '//schema.org' in url:
            if len(parents) > 0:
                raise Exception("This URL should not have parents defined in 'D3M' schema: {url}".format(url=url))

            define_external_type(url)
            types[url]['description'] = description
        else:
            types[url] = {
                'label': label,
                'description': description,
                'parents': parents
            }

    # Cycle detection.
    for url in types:
        cycle_detection(url, [])

    # Constructing site at the root of the domain.
    site, tag, text, line = Doc().ttl()

    for temp in template(tag, line):
        line('h1', "Versions", klass='page-title', style='margin-bottom: 20px')

        with tag('div', klass='breadcrumbs'):
            line('a', "devel", href='devel')

        # Sorting versions by release number.
        versions = {}
        for folder in os.listdir('public'):
            try:
                versions[folder] = [-int(number) for number in folder[1:].split('.')]
            except:
                pass

        for version in sorted(versions, key=lambda key: versions[key]):
            if version.startswith('v'):
                with tag('div', klass='breadcrumbs'):
                    line('a', version, href=version)

    with open('public/index.html', 'w') as file:
        file.write(site.getvalue())

    # Constructing site for all types.
    site, tag, text, line = Doc().ttl()

    for temp in template(tag, line):
        line('h1', "Semantic Types", klass='page-title', style='margin-bottom: 20px')
        line('div', schema['definitions']['semantic_types'].get('description', ''), style='margin-bottom: 20px')
        construct_types(site, '')

    with open('public/types/index.html', 'w') as file:
        file.write(site.getvalue())

    # Constructing site for each type.
    for url in types:
        if url.startswith(PREFIX):
            site, tag, text, line = Doc().ttl()

            for temp in template(tag, line):
                line('h1', types[url]['label'], klass='page-title')
                with tag('span', klass='canonicalUrl'):
                    text("Canonical URL: ")
                    line('a', url, href=url)
                with tag('h4'):
                    if len(types[url]['parents']) > 0:
                        construct_breadcrumbs(site, [url])
                line('div', types[url]['description'])

            with open('public/types/' + types[url]['label'], 'w') as file:
                file.write(site.getvalue())


if __name__ == '__main__':
    main()
