### InlineOutline

You just installed InlineOutline.  That's awesome.

The plugin claims `primary+shift+o` as its main entry point.  That's just
the default.  You can change the binding in the plugin settings.  Just open
the Command Palette and type `Preferences: InlineOutline Settings`.  If you
don't want any global bindings, just set `never`.


## Try it

But before that, just try the default for now.  Right in this view, hit
`primary-shift+o` repeatedly.  See how the text collapses to just the
headlines, and then back again.

Now, in Outline mode, use `up` and `down` to navigate.  Press `enter` to
commit, i.e., exit the Outline mode and scroll to the selected headline. `esc`
will abort the Outline mode and restore the original cursor and scroll
positions.


** If you already use `primary+shift+o` in your User package, the plugin
   will *not* override that.  In that case, the demo above will not work for
   you.  Either you comment out the key binding section in your User
   package _or_ you define a different binding as described above. **

