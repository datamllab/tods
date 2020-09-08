import $ from 'jquery';

import {reloadPage, encodeID, referenceKey, expandAll, shrinkAll} from './html_construction'

/**
 * Names of schemas as they appear in the `schemas/v0` folder, without the extension.
 * @type {string[]}
 */
const names = [
  'primitive',
  'container',
  'data',
  'problem',
  'pipeline',
  'pipeline_run',
  'definitions',
];

/**
 * Placeholder for schemas. Keys are names of the files without the extension.
 * @type {{}}
 */
export let schemas = {};

/**
 * Schema that contains definitions. It also appears in `schemas`.
 */
let definitions;

/**
 * True while fetching schemas from the server.
 * @type {boolean}
 */
export let fetching = true;

/**
 * Key to be used in the cycle detection algorithm.
 * @type {string}
 */
const cycleKey = 'cycle_detection_pflk2jds32ljfi2jfoja-p-2rla';

// Fetching schemas.
Promise.all(names.map((name) => {
  return fetch(`schemas/v0/${name}.json`, {credentials: 'same-origin'}).then((response) => {
    if (response.ok) {
      return response.json();
    }
    else {
      throw new Error("Fetch failed.");
    }
  });
})).then((fetchedSchemas) => {
  // Moving schemas into "schemas" and "definitions".
  fetchedSchemas.forEach((schema, i) => {
    schemas[names[i]] = schema;
    if (names[i] === 'definitions')
      definitions = schema;
  });

  // Adding references in "definitions" to their pages.
  Object.keys(definitions['definitions']).forEach((i) => {
    definitions['definitions'][i][referenceKey] = {
      'url': '?definitions#' + encodeID('definitions/' + i, false),
      'name': 'definitions/' + i
    };
  });

  // Connecting schemas with "definitions" and applying workarounds.
  Object.keys(schemas).forEach((schema) => {
    schemas[schema] = resolveReferences(schemas[schema]);
  });

  fetching = false;
  reloadPage();
});

$('#shrink-all').click(function (event) {
  removeUrlParameter();
  shrinkAll();
});
$('#expand-all').click(function (event) {
  // Constructing url with added "expanded" parameter.
  let search = window.location.search.substring(1).split('&');
  let parameters = ['expanded'];
  search.forEach((parameter) => {
    if (parameter.length > 0 && parameters.indexOf(parameter) === -1)
      parameters.unshift(parameter);
  });
  let url = '?' + parameters.join('&') + window.location.hash;

  history.replaceState(history.state, undefined, url);
  expandAll();
});

$(window).on('popstate', function (event) {
  reloadPage();
});

// Event handler for "scroll to top" button.
let $scrollToTop = $('#scroll-to-top').click(function (event) {
  $('html, body').animate({scrollTop: 0, scrollLeft: 0});
});

// Showing and hiding "scroll to top" button.
window.onscroll = function (event) {
  if (document.body.scrollTop > 50 || document.documentElement.scrollTop > 50)
    $scrollToTop.show();
  else $scrollToTop.hide();
};

// Creating links to the different schemas at the top of the page.
let $container = $('#links');
names.forEach((name, i) => {
  $('<a>').attr('href', '?' + name).text(name).appendTo($container).click(function (event) {
    event.preventDefault();
    history.pushState(undefined, undefined, '?' + name);
    reloadPage();
  });

  // Appending comma.
  if (i < names.length - 1) {
    $('<span>').text(', ').appendTo($container);
  }
});

// Saving scrolling position when leaving this application, relevant if user returns.
window.onbeforeunload = function (event) {
  history.state.position = window.pageYOffset || document.documentElement.scrollTop;
  history.replaceState(history.state, undefined);
};

/**
 * Replaces references (objects with the only key `$ref`) with the actual object they are pointing
 * to. Objects are not copied, so there always exists only one copy of each object. That also enables
 * easier cycle detection.
 *
 * Also applies workarounds.
 *
 * @param iterator Object, over which we are currently iterating.
 * @returns {*} Updated object.
 */
function resolveReferences(iterator) {
  if (typeof iterator === 'object') {

    // Workarounds.
    iterator = workaround1(iterator);
    workaround2(iterator);

    // Marking the current object, so we know that we have been here.
    iterator[cycleKey] = true;

    for (let i in iterator) {
      if (!iterator.hasOwnProperty(i))
        continue;

      if (i === '$ref') {

        // Getting object from "definitions".
        let subSchema = getFromDefinitions(iterator[i]);

        // Checking if we have already been there.
        if (subSchema.hasOwnProperty(cycleKey)) {
          // Constructing new object with some description.
          let ret = {'[Cycle in JSON schema, see link]': ''};

          // Adding type, reference and description to the returned object, so user will get some idea of
          // first repeated layer.
          if (subSchema.hasOwnProperty('type')) ret['type'] = subSchema['type'];
          if (subSchema.hasOwnProperty(referenceKey)) ret[referenceKey] = subSchema[referenceKey];
          if (subSchema.hasOwnProperty('description')) ret['description'] = subSchema['description'];

          delete iterator[cycleKey];
          return ret;
        }

        // References can also appear in the sub schemas.
        let ret = resolveReferences(subSchema);

        delete iterator[cycleKey];
        return ret;
      }

      // If it's not a reference, go deeper.
      iterator[i] = resolveReferences(iterator[i]);
    }
    delete iterator[cycleKey];
  }
  return iterator;
}

/**
 * Url represents path to object in `definitions`, this function finds it and returns it. If object at given
 * path does not exist, function does not crash, but rather returns url.
 *
 * @param url Url of the object.
 * @returns {*} Retrieved object.
 */
function getFromDefinitions(url) {
  try {

    // If url is like '.../pipeline.json'.
    if (url.indexOf('#') === -1) {
      let ret = {'[Cycle in JSON schema, see link]': ''};
      ret[referenceKey] = {
        'name': url,
        'url': '?' + url.substring(url.lastIndexOf('/') + 1, url.lastIndexOf('.'))
      };
      return ret;
    }

    // Supposing that 'url' is like '#/definitions/sth'.
    else {
      let subSchema = definitions;
      let path = url.substring(url.lastIndexOf('#') + 1, url.length).split('/');
      path.forEach((part) => {
        if (part.length > 0) {
          subSchema = subSchema[part];
        }
      });
      return subSchema;
    }
  }
  catch (error) {
    console.log(error);
  }
  return {"[couldn't show this schema]": url};
}

/**
 * Removes `expanded` parameter from the url.
 */
export function removeUrlParameter() {
  let search = window.location.search.substring(1).split('&');
  let parameters = [];
  search.forEach((parameter) => {
    if (parameter.length > 0 && parameter !== 'expanded')
      parameters.push(parameter);
  });

  let url = '?' + parameters.join('&') + window.location.hash;
  history.replaceState(history.state, undefined, url);
}

/**
 * If `data` has following structure:
 *
 * {
 *   'allOf': [one schema with description1]
 *   'description': description2
 * }
 *
 * Then description1 is replaced with description2, and whole object is replaced with the schema.
 *
 * @param data Some schema or part of the schema.
 * @returns {boolean} Updated object.
 */
function workaround1(data) {
  if (Object.keys(data).length === 2 &&
      data.hasOwnProperty('allOf') && data['allOf'].length === 1 &&
      data.hasOwnProperty('description')) {
    let child = data['allOf'][0];
    child['description'] = data['description'];
    data = child;
  }
  return data;
}

/**
 * If `data` has following structure:
 *
 * {
 *   'anyOf' or 'oneOf': [
 *     {
 *       'enum': array of length 1
 *       ...
 *     },
 *     {
 *       'enum': array of length 1
 *       ...
 *     },
 *     ...
 *   ],
 *   ...
 * }
 *
 * Then all objects with `enum` are put into one array.
 *
 * @param data Some schema or part of the schema.
 */
function workaround2(data) {
  if (data.hasOwnProperty('anyOf') || data.hasOwnProperty('oneOf')) {
    let key = data.hasOwnProperty('anyOf') ? 'anyOf' : 'oneOf';
    // Placeholder for other properties.
    let other = [];
    // Placeholder for enums.
    let enums = [];

    // Iterating over data and searching for enums.
    Object.keys(data[key]).forEach((i) => {
      let object = data[key][i];
      // If it contains "enum".
      if (typeof object === 'object' && object.hasOwnProperty('enum') && object['enum'].length === 1) {
        object['text'] = object['enum'][0];
        delete object['enum'];
        enums.push(object);
      }
      else {
        other.push(object);
      }
    });

    if (enums.length > 0) {
      // Sorting.
      enums.sort(function (a, b) {
        return a['text'].localeCompare(b['text']);
      });

      // "oneOf -> enum" is represented as "enum" only.
      if (key === 'oneOf' && other.length === 0) {
        data['enum'] = enums;
        delete data[key];
      }
      // Typical case.
      else {
        // Adding "enum" at the end.
        other.push({
          'enum': enums
        });
        // Updating.
        data[key] = other;
      }
    }
  }
}
