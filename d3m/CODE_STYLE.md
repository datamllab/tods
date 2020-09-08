# Code Style

## Python

**Consistency is the main code style guideline** and if in doubt, try to find a similar existing code and style
your code the same. Our code style is very similar to [PEP8](https://www.python.org/dev/peps/pep-0008/) with few
more details.

Indent with 4 spaces. Never with tabs. No trailing whitespace.

**Be verbose**. Always fully spell out any part of the function, class, or variable name.

### Blank lines

Use blank lines to organize long code blocks into units of what they do. Often a block is preceded by a
comment, explaining what the block does.

This will help someone new understand the code quicker when they read it. You are leaving little hints behind,
what parts of code to understand as one unit, one step of your algorithm. Imagine you were writing the code
to be published in an article and you try to make everything as easy to learn as possible. It's the same
here, because we assume our teammates are going to use the code after us.

Comments always have one blank line before them, except when they are the first line of an indented block of code.

```python
for item in items:
    # No new line above this comment.
    ...

# New line above this comment.
...
```

Do not have multiple (two or more) blank lines beyond what is expected by PEP8.

### Line wrapping

We **do not wrap lines** except when logically reasonable or when it greatly increases readability
(we still wrap logically and not just at the end of the line).

We do wrap comments at the 120 characters right margin. If the comment wraps to two lines, balance the lines
so they are both approximately the same length.

The closing brace/bracket/parenthesis on multi-line constructs should align with the first character of the
line that starts the multi-line construct, as in:

```python
my_list = [
    1, 2, 3,
    4, 5, 6,
]
result = some_function_that_takes_arguments(
    'a', 'b', 'c',
    'd', 'e', 'f',
)
```

Always include a trailing comma in such cases.

When defining a function which takes too many arguments to leave all of them in one line, use hanging indentation:

```python
def some_function_that_takes_arguments(
    a, b, c,
    d, e, f,
):
    return a + b + c + d + e + f
```

[Black](https://github.com/python/black) generally formats according to this style so you can use
it to help you.

### Strings

Use `'single_quote_strings'` for constant strings and `"double_quote_strings"` for any string shown to the
user (like exception messages, or warning). A general guideline is: if a string might be ever translated to a
different language, use double quotes for it.

This means all dictionary key names should use single quotes.

Always use keyword based string formatting. When only simple variable name interpolation is being done,
[f-Strings](https://realpython.com/python-f-strings/) are the preferred format.

```python
f"Value is '{value}' and message is '{message}'."
```

If longer expressions are being computed, then `.format()` should be used, with keywords.

```python
"This complicated string lists all values: {values}".format(
    values=[x.lowercase() for x in values],
)
```

Inline values wrap inside messages with `'`. If value is at the end of the message, there is no
need for wrapping and also no need for trailing dot.

When creating logging statements, use `%`-based format, also with keyword based arguments.

```python
logger.misc("Step '%(requirement)s'.", {'requirement': requirement})
```

### Logging

Use [Python logging facility](https://docs.python.org/3/library/logging.html) for all output and never use
`print()` (except when used in CLI commands). Obtain `logger` instance by using `__name__`, at the very
beginning of the module:

```python
import logging

logger = logging.getLogger(__name__)
```

### Imports

Imports should be **just modules** divided into multiples sections, in order from more global to more local, separated by empty line:
 * core Python packages
 * external Python packages
 * non-local imports (for example, imports from some other top-level `d3m.` module)
 * local relative imports for the current module and sub-modules

Inside each section, imports should be ordered alphabetically, first based on package name, then on model imported.
Each package should be on its own line, but importing multiple modules from the same package should be in one line.

Example:

```python
import os
import time

import numpy
from sklearn import metrics, model_selection

from d3m import exceptions
from d3m.metadata import problem

from . import data
```

If you are importing multiple modules with the same name from different package, rename more global one with a prefix
of the package:

```python
from sklearn import metrics as sklearn_metrics

from d3m import metrics
```

### Docstrings

Every class, method and function has a docstring with description. Docstrings should be split into multiple lines
when needed to improve readability. Docstrings should use the [numpy style](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard)
to document arguments, return value and everything else, which means also using [ReST/Sphinx](http://www.sphinx-doc.org/en/stable/rest.html)
syntax for formatting.

Always separate the docstring from the rest of the code with an empty line, and have `"""` on their own line, even
for one-line docstrings.

We use a custom [Python metaclasses](https://docs.python.org/3/reference/datamodel.html#metaclasses) for d3m classes which
[automatically inherits or extends docstrings from parent methods](https://github.com/meowklaski/custom_inherit):

```python
from d3m import utils


class MyBaseClass(metaclass=utils.Metaclass):
    pass


class MyAbstractBaseClass(metaclass=utils.AbstractMetaclass):
    pass
```

### Comments

Both standalone one-sentence one-line comments and multi-sentence comments should have grammatically correct punctuation.
For formatting, use [ReST/Sphinx](http://www.sphinx-doc.org/en/stable/rest.html) syntax.

- When you are explaining what the code will do, end the sentence with a dot. 

  ```python
  # Calculate total value.
  value = quantity * price
  ```
- Short after-the-line comments (which should not be sentences) do not have an ending dot:

  ```python
  sleep(10)  # seconds
  ```

- Titles that are separating sections of code are also not a sentence (no dot).

  ```python
  ### Vector operations ###

  def dot_product(vector1, vector2):
      ...

  def cross_product(vector1, vector2):
      ...

  ### Matrix operations ###

  def transform(vector, matrix):
      ...
  ```

If TODO comments cannot be a short one-line with grammatically correct punctuation, then split it into multiple lines in this way:

```python
# TODO: Short description of a TODO.
#       A longer description of what we could potentially do here. Maybe we
#       could do X or Y, but Y has this consequences. We should probably
#       wait for server rendering feature to be implemented.
#       See: https://github.com/example/project/issues/123
```

Try to keep the formatting of the first line exactly as shown above so that it is easier parsed by IDEs.
Including the space after `#` and space after `:`.

## Code repository

Commit often and make sure each commit is a rounded change. Do not squash commits, unless that helps making a set of commits
into a clearer change. We leave unsuccessful attempts in the repository because maybe in the future we can come back to them
and use them, maybe in a different context or way.

For almost all changes to the repository, we make first a feature branch from `devel` branch. We make all necessary changes in
that new branch, potentially make multiple commits. We make a merge request against the `devel` branch for the change
to be reviewed and merged. We should make a merge request even before all changes are finished so that others can comment
and discuss the development. We can continue adding more commits to this branch even after the merge request has been made
and GitLab will update the merge request automatically. Until a merge request is finished and is deemed ready to be merged
by its author, merge request's title should be prefixed with `WIP:` so that it is clear that it is not yet meant
to be merged (and thoroughly reviewed). Make sure you include also a change to the [changelog](#changelog) in your merge request.

### Changelog

We are maintaining `HISTORY.md` file where we document changes to the project so that
everyone involved can have one location where they can see what has changed and what
they might adapt in their code or the way they are working on the project.

### Commit messages

Commit messages should be descriptive and full sentences, with grammatically correct punctuation.
If possible, they should reference relevant tickets (by appending something like `See #123.`) or even close them
(`Fixes #123.`). GitLab recognizes that. If longer commit message is suitable (which is always a good thing),
first one line summary should be made (50 characters is a soft limit), followed by an empty line, followed
by a multi-line message:

    Added artificially lowering of the height in IE.
    
    In IE there is a problem when rendering when user is located
    higher than 2000m. By artificially lowering the height rendering
    now works again.

    Fixes #123.
