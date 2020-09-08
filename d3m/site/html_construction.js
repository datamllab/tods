import $ from 'jquery';

import {schemas, fetching, removeUrlParameter} from './client';

/**
 * Ids that were already given by the `encodeID` function.
 * @type {Set<string>}
 */
let usedIDs = new Set();

/**
 * Key to be used when linking some object to its original definition.
 * @type {string}
 */
export const referenceKey = 'reference_kfemjkvfi39rfj39fjckslgfgv2frskfj';

/**
 * List of keys that are not to be treated or are treated differently.
 * @type {string[]}
 */
const knownSpecifiers = ['title', 'type', 'required', 'description', 'properties', '$schema', 'id', referenceKey, 'text'];

/**
 * Map between some keys and more readable version of them.
 * @type {{string}}
 */
const prettierKeys = {
  'allOf': 'all of',
  'anyOf': 'any of',
  'oneOf': 'one of',
  'additionalProperties': 'additional properties',
  'patternProperties': 'pattern properties',
  'minItems': 'min items'
};

/**
 * Array for caching content inside `#container` element. When html for schema is constructed,
 * it is placed inside `#container`, and pushed here. Index of it is pushed to history.state,
 * so content can be restored when the user presses browser's `back` button.
 * @type {Array}
 */
let pastContents = [];

/**
 * When schema is requested and page is constructed, its copy is thrown here, so at next visit (within same session)
 * construction of html doesn't need to be performed again.
 * @type {{}}
 */
let cache = {};

/**
 * Session key used to differentiate between different sessions of the page. If the page is reloaded, new session
 * key is generated, so content is reloaded even though matching number may be present in `history.state`.
 * @type {number}
 */
let session = Math.random();

/**
 * Simulates fresh visit of the page:
 *   - clears content
 *   - ensures html representing schema
 *   - applies event handlers
 *   - expands desired parts of the schema
 *   - scrolls to the desired point in page
 */
export function reloadPage() {
  if (fetching) return;

  // Clearing content. ".children().detach()" would keep event
  // listeners, but they are lost either way when content is cloned.
  let $container = $('#container').empty();

  $('#title').hide();
  let $instructions = $('.instruction').hide();
  usedIDs.clear();

  // Getting name of the schema.
  let name = undefined;
  window.location.search.substring(1).split('&').forEach((parameter) => {
    if (schemas.hasOwnProperty(parameter)) {
      name = parameter;
    }
  });
  if (name === undefined) {
    return;
  }

  // Creating new content, if "back" was NOT pressed.
  let state = history.state, getNewContent = false;
  try {
    if (typeof state['index'] !== 'number' || state['index'] >= pastContents.length || state['index'] < 0 ||
        state['session'] !== session)
      getNewContent = true;
  }
  catch (e) {
    getNewContent = true;
  }

  // Ensuring new content.
  let scrollPosition, title = schemas[name].hasOwnProperty('title') ? schemas[name]['title'] : '';
  if (getNewContent) {
    // If not in "cache", construct new HTML content and save it.
    if (!cache.hasOwnProperty(name)) {
      // Placeholder.
      let content = div();
      constructHTML(schemas[name], content, title, [], []);
      cache[name] = content;
    }

    // Getting expanded buttons. Item inside expands plus at the root.
    let expanded = [encodeID(title, false)];
    try {
      if (Array.isArray(state['expanded'])) {
        expanded = state['expanded'];
      }

      // Getting scrolling position, relevant when user leaves this domain and returns.
      scrollPosition = state['position'];
    }
    catch (e) {}

    // Replacing this state with new data.
    history.replaceState({
      'index': pastContents.length,
      'session': session,
      'expanded': expanded
    }, undefined);

    pastContents.push(cache[name].clone());
  }

  // Showing content.
  pastContents[history.state['index']].appendTo($container);

  // Event listener for links.
  $container.find('.link').click(function (event) {
    event.preventDefault();
    history.pushState(undefined, undefined, $(this).attr('href'));
    reloadPage();
  });

  // Event listener for buttons (+/-).
  $container.find('.button').click(function (event) {
    buttonClick($(this), event);
  }).next('.title').css('cursor', 'pointer').click(function (event) {
    buttonClick($(this).prev(), event);
  });

  // Event listener for coloring paragraph.
  $container.find('.paragraph-row').mouseenter(function (event) {
    $(this).children('.paragraph').css('color', '#e2e0e5');
  }).mouseleave(function (event) {
    $(this).children('.paragraph').css('color', 'transparent');
  });

  // Expanding rows above one in the URL's hash and coloring it.
  if (window.location.hash.length > 1) {
    let id = window.location.hash.substring(1);
    // We use "getElementById" and not jQuery because "." can be in "id".
    let $el = $(document.getElementById(id));
    $el.parentsUntil($container, '.container').children('.first-row').children('.button').each(function () {
      expand($(this));
    });
    $el.addClass('yellow');
  }

  // Expanding rows that were expanded before.
  Object.keys(history.state['expanded']).forEach((i) => {
    let id = history.state['expanded'][i];
    if (id.length > 0) {
      // We use "getElementById" and not jQuery because "." can be in "id".
      let $row = $(document.getElementById(id));
      expand($row.children('.button').first());
    }
  });

  // Expanding all.
  if (window.location.search.substring(1).split('&').indexOf('expanded') !== -1)
    expandAll();

  // Showing title, buttons.
  $('#title').show().text(title);
  $instructions.show();

  // Scrolling content to the location in the URL's hash.
  scrollToHash();

  // Scrolling content when returning from different domain.
  if (scrollPosition) {
    document.documentElement.scrollTop = document.body.scrollTop = scrollPosition;
  }
}

/**
 * Constructs HTML for given schema in `data`.
 *
 * @param data Some schema or part of the schema that is to be displayed.
 * @param container jQuery element, in which schema is to be displayed.
 * @param title Title of the element, which should equal the key object was retrieved with.
 * @param required An array of required properties.
 * @param path List of titles of previously called objects.
 */
export function constructHTML(data, container, title, required, path) {
  // Needed when coloring row in yellow.
  container.addClass('container');
  let firstRow = div('first-row').appendTo(container);
  let showButton = false;

  // Simple case.
  if (typeof data !== 'object') {
    div('el').html(urlify(data)).appendTo(firstRow);
    return;
  }

  // Title.
  if (title.length > 0) {
    div('el title').html(urlify(title)).appendTo(firstRow);
    path.push(title);
  }

  // Text.
  if (data.hasOwnProperty('text')) {
    div('el').html(urlify(data['text'])).appendTo(firstRow);
  }

  // Type and required.
  let parenthesis = [];
  if (data.hasOwnProperty('type')) {
    parenthesis.push(data['type']);
  }
  if (required.indexOf(title) !== -1) {
    parenthesis.push('required');
  }
  if (parenthesis.length > 0) {
    div('el parenthesis', `(${parenthesis.join(', ')})`).appendTo(firstRow);
  }

  // Reference.
  if (data.hasOwnProperty(referenceKey) && path.join('/') !== data[referenceKey]['name']) {
    $('<a>').addClass('el link').attr('href', data[referenceKey]['url']).text(data[referenceKey]['name']).appendTo(firstRow);
  }

  if (firstRow.children().length > 0) {
    // Applying margin between rows.
    firstRow.before(div('margin'));

    // ID.
    let id = encodeID(path.join('/'), true);
    firstRow.attr('id', id).addClass('paragraph-row');

    // Adding link.
    let $link = $('<a>').addClass('el paragraph link').attr('href', `#${id}`).appendTo(firstRow);
    $('<i>').addClass('fa fa-paragraph').attr('aria-hidden', 'true').appendTo($link);

    // Description.
    if (data.hasOwnProperty('description'))
      div('el description').html(urlify(data['description'])).appendTo(div('shift').appendTo(firstRow));

    // Updating container.
    container = div('shift').appendTo(container);
  }
  else {
    firstRow.remove();
  }

  // All keys that are not listed in "knownSpecifiers".
  // Constructing HTML elements. They are not shown, but pushed into "buffer".
  let buffer = [];
  Object.keys(data).forEach((key) => {
    let el = data[key];

    if (knownSpecifiers.indexOf(key) === -1) {
      showButton = true;
      path.push(key);

      // Creating wrapping "div" and appending it to buffer.
      let wrapper = div();
      buffer.push([wrapper, undefined]);

      // First row.
      let row = div().appendTo(wrapper);
      div('el color', prettierKeys.hasOwnProperty(key) ? prettierKeys[key] : key).appendTo(row);

      // Simple case.
      if (typeof el !== 'object') {
        div('el').html(urlify(el)).appendTo(row);
      }
      else {
        // Shifting.
        row = div('shift').appendTo(wrapper);

        if (key === 'definitions') {
          // Sorting.
          let keys = Object.keys(el).sort(function (a, b) {
            return a.localeCompare(b);
          });

          // Constructing html for each element.
          keys.forEach((definition) => {
            constructHTML(el[definition], div().appendTo(row), definition, [], path);
          });
        }
        else {
          // Ensuring it's an array.
          if (!Array.isArray(el))
            el = [el];

          // Sorting if "el" consists only of strings.
          let sort = true;
          for (let i in el) {
            if (el.hasOwnProperty(i) && typeof el[i] !== "string") {
              sort = false;
              break;
            }
          }
          if (sort) {
            el.sort(function (a, b) {
              return a.localeCompare(b);
            });
          }

          el.forEach((child, i) => {
            if (el.length > 1) {
              path.push(i);
            }
            constructHTML(child, row, '', [], path);
            if (el.length > 1) {
              path.pop();
            }

            // Division markers.
            if (parseInt(i) !== data[key].length - 1) {
              if (key === 'allOf') {
                div('or').html('---------&nbsp&nbspand&nbsp&nbsp---------').appendTo(row);
              }
              else if (key === 'oneOf' || key === 'anyOf') {
                div('or').html('---------&nbsp&nbspor&nbsp&nbsp---------').appendTo(row);
              }
            }
          });
        }
      }
      path.pop();
    }
  });

  // Sorting and showing content from "buffer".
  if (buffer.length > 1) {
    buffer.forEach((el) => {
      el[1] = el[0].find('*').length;
    });
    buffer.sort(function (a, b) {
      return a[1] - b[1];
    });
  }
  buffer.forEach((el) => {
    el[0].children().appendTo(container);
  });

  // Properties.
  if (data.hasOwnProperty('properties')) {
    showButton = true;

    // Sorting.
    let keys = Object.keys(data['properties']).sort(function (a, b) {
      return ('' + a).localeCompare(b);
    });

    // Constructing html for each element.
    keys.forEach((key) => {
      let row = div().appendTo(container);
      constructHTML(data['properties'][key], row, key, data.hasOwnProperty('required') ? data['required'] : [], path);
    });
  }

  // Button.
  if (showButton && firstRow.children().length > 0) {
    let button = div('button');
    $('<i>').attr('aria-hidden', 'true').addClass('el fa fa-plus').appendTo(button);
    firstRow.prepend(button);
    container.hide();
  }

  if (title.length > 0) {
    path.pop();
  }
}

/**
 * Creates html `<div>` element with given classes and text.
 *
 * @param classes
 * @param text
 * @returns {jQuery} Created element.
 */
function div(classes, text) {
  let ret = document.createElement('div');

  if (typeof classes !== 'undefined') {
    ret.className = classes;
  }
  if (typeof text !== 'undefined') {
    ret.innerText = text;
  }

  return $(ret);
}

/**
 * Replaces characters in `text` so it can be used as an id of html element. If `unique` is True,
 * then function remembers the id and doesn't return the same one twice.
 *
 * @param text
 * @param unique
 * @returns {string} Formatted text.
 */
export function encodeID(text, unique) {
  text = text.replace(/\//g, '.');
  text = text.replace(/ /g, '_');
  text = text.replace(/[^a-z0-9-_:.]/gi, '');

  if (unique) {
    if (usedIDs.has(text)) {
      let i = 0;
      while (usedIDs.has(text + ++i)) {}
      text += i;
    }
    usedIDs.add(text);
  }

  return text;
}

function scrollToHash() {
  if (window.location.hash.length > 1) {
    let id = window.location.hash.substring(1);
    // We use "getElementById" and not jQuery because "." can be in "id".
    let $offset = $(document.getElementById(id)).offset();
    if (typeof $offset !== 'undefined') {
      $offset.left -= 20;
      $offset.top -= 20;
      $('html, body').animate({
        scrollTop: $offset.top,
        scrollLeft: $offset.left
      }, 0);
    }
  }
}

function urlify(text) {
  let urlRegex = /https?:\/\/[^\s]+\B./g;
  return ('' + text).replace(urlRegex, function (url) {
    return `<a href="${url}">${url}</a>`;
  })
}

/**
 * Shrinks content below the element, which should represent button.
 * @param $el jQuery element.
 */
function shrink($el) {
  $el.children().removeClass('fa-minus').addClass('fa-plus');
  $el.parent().next().hide();
}

/**
 * Expands content below the element, which should represent button.
 * @param $el jQuery element.
 */
function expand($el) {
  $el.children().removeClass('fa-plus').addClass('fa-minus');
  $el.parent().next().show();
}

/**
 * Handles the event when button or title next to the button is clicked.
 *
 * @param $element jQuery button that was clicked.
 * @param event Event, which holds information whether `Ctrl` key was pressed during the click.
 */
function buttonClick($element, event) {
  if ($element.children().first().hasClass('fa-minus')) {
    shrink($element);
    $element.parent().next().find('.button').each(function () {
      shrink($(this));
    });
  }
  else {
    expand($element);
    if (event.ctrlKey) {
      $element.parent().next().find('.button').each(function () {
        expand($(this));
      });
    }
  }
  removeUrlParameter();
  updateHistoryState();
}

/**
 * Updates which buttons are expanded in `history.state`.
 */
function updateHistoryState() {
  history.state['expanded'] = [];
  $('#container').find('.button').each(function () {
    let $el = $(this);
    if ($el.children().first().hasClass('fa-minus')) {
      history.state['expanded'].push($el.parent().attr('id'));
    }
  });
  history.replaceState(history.state, undefined);
}

export function shrinkAll() {
  $('.button').each(function () {
    shrink($(this));
  });
  updateHistoryState();
}

export function expandAll() {
  $('.button').each(function () {
    expand($(this));
  });
  updateHistoryState();
}
