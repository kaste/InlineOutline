## Hi 👋🏾

This is a plugin for Sublime Text.  It started as a temporary Outline-mode, 

https://github.com/kaste/InlineOutline/assets/8558/ae50d907-3004-4320-9740-8d3467e223cc

and now is a complete replacement but different approach to *Goto Symbol*.

E.g. fuzzy-search for a symbol,

https://github.com/kaste/InlineOutline/assets/8558/42a8a375-2bae-4385-b0df-8580d6c83f69

E.g. walk using the arrow keys (or `,`and `.`)

https://github.com/kaste/InlineOutline/assets/8558/d6eca69b-c9b9-46b3-9b66-e61dd6303b47

As usual, `<enter>` will go to to the selected symbol and `<esc>` will reset the
cursor and viewport.


# Key binding

Currently it binds `primary+shift+o` (`primary` is `ctrl`), `o` as in outline.

You can set an initial search term, e.g.

```json
{
    "keys": ["primary+shift+o"],
    "command": "enter_outline_mode",
    "args": { "enter_search": "class " }
}
```
would let you see all defined *classes* in a python file.
