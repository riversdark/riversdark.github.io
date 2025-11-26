+++
title = "Welcome is in order"
author = ["olivier"]
date = 2025-11-21
draft = true
showtoc = true
tocopen = true
math = true
tags = ["welcome"]
+++

Kind of you to drop by! This is the personal website of Olivier Ma.

The new Hugo/PaperMod setup keeps TOCs and KaTeX on by default. If a post
should hide the TOC, set in the subtree properties:

```text
:EXPORT_HUGO_CUSTOM_FRONT_MATTER: :showtoc false :tocopen false :math false
```

A math snippet for testing:

\\[
   \int\_0^1 x^2 \\, dx = \frac{1}{3}.
   \\]

And a code block for syntax highlighting:

```python
import math
print(math.sqrt(2))
```

```text
1.4142135623730951
```
