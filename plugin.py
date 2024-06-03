from __future__ import annotations
from contextlib import contextmanager
from functools import lru_cache
from itertools import chain, tee

import sublime
import sublime_plugin


from typing import Iterable, Iterator, NamedTuple, Sequence, TypeVar
from typing_extensions import TypeAlias
T = TypeVar("T")
flatten = chain.from_iterable


class LineSpan(NamedTuple):
    start: int
    end: int

    @classmethod
    def from_region(cls, view: sublime.View, region: sublime.Region) -> LineSpan:
        return cls(
            view.rowcol(region.begin())[0],
            view.rowcol(region.end())[0]
        )


class TextRange(NamedTuple):
    text: str
    region: sublime.Region

    @classmethod
    def from_region(cls, view: sublime.View, region: sublime.Region) -> TextRange:
        return cls(view.substr(region), region)


ViewState: TypeAlias = "tuple[tuple[float, float], list[tuple[int, int]]]"


def view_state(view: sublime.View) -> ViewState:
    return (
        view.viewport_position(),
        [s.to_tuple() for s in view.sel()]
    )


class OutlineModeQueryContext(sublime_plugin.EventListener):
    def on_query_context(
        self,
        view: sublime.View,
        key: str,
        operator,
        operand,
        match_all: bool
    ) -> bool | None:
        if key == "outline_mode":
            if operator != sublime.OP_EQUAL:
                print(f"Context '{key}' only supports operator 'equal'.")
                return False

            if operand is not True:
                print(f"Context '{key}' only supports operand 'true'.")
                return False

            return (
                bool(view.folded_regions())
                and view.settings().get("outline_mode")
            )

        return None


class enter_outline_mode(sublime_plugin.TextCommand):
    def run(self, edit: sublime.Edit) -> None:
        view = self.view
        original_view_state = view_state(view)
        frozen_sel = [s for s in view.sel()]
        sel = frozen_sel[0]
        caret = sel.begin()
        row, _ = view.rowcol(caret)
        symbols = symbol_regions(view)
        if not symbols:
            flash(view, "This view defines no symbols.")
            return
        focus_regions(view, [s.region for s in symbols])
        _, y = view.viewport_position()
        view.set_viewport_position((0, y))

        _, nearest_region = min(
            (
                (
                    min(abs(line.start - row), abs(line.end - row)),
                    s.region
                )
                for s in symbols
                if (line := LineSpan.from_region(view, s.region))
            ),
            key=lambda distance_symbol: distance_symbol[0],
            default=(caret, sublime.Region(caret))
        )
        set_sel(view, [flip_region(nearest_region)])
        view.settings().set("outline_mode", True)
        view.settings().set("original_view_state", original_view_state)


class abort_outline_mode(sublime_plugin.TextCommand):
    def run(self, edit: sublime.Edit) -> None:
        view = self.view
        original_view_state: ViewState | None
        original_view_state = view.settings().get("original_view_state")

        view.run_command("exit_outline_mode")
        if original_view_state:
            apply_view_state(view, original_view_state)


def apply_view_state(view: sublime.View, state: ViewState) -> None:
    viewport_position, _sel = state
    frozen_sel = [
        sublime.Region(*s)
        for s
        in _sel
    ]
    set_sel(view, frozen_sel)
    view.set_viewport_position(viewport_position)


class exit_outline_mode(sublime_plugin.TextCommand):
    def run(self, edit: sublime.Edit) -> None:
        view = self.view
        view.run_command("fixed_unfold_all")
        view.settings().erase("outline_mode")
        view.settings().erase("original_view_state")


class fixed_unfold_all(sublime_plugin.TextCommand):
    def run(self, edit):
        view = self.view
        with stable_viewport(view):
            view.run_command("unfold_all")


@contextmanager
def stable_viewport(view):
    frozen_sel = [s for s in view.sel()]
    sel = frozen_sel[0]
    caret = sel.begin()

    offset = y_offset(view, caret)
    yield
    apply_offset(view, caret, offset)


def y_offset(view, cursor):
    # type: (sublime.View, int) -> float
    _, cy = view.text_to_layout(cursor)
    _, vy = view.viewport_position()
    return cy - vy


def apply_offset(view: sublime.View, cursor: int, offset: float) -> None:
    _, cy = view.text_to_layout(cursor)
    vy = cy - offset
    vx, _ = view.viewport_position()
    view.set_viewport_position((vx, vy), animate=False)


def focus_regions(
    view: sublime.View,
    regions: Iterable[sublime.Region],
    context: int = 0
) -> None:
    selected_row_spans = list(
        LineSpan.from_region(view, r)
        for r in sorted(regions, key=lambda r: r.begin())
    )
    # print("selected_row_spans", selected_row_spans)

    last_row_of_view = view.rowcol(view.size())[0]

    line_spans_to_fold = list(
        LineSpan(a_end + context, b.start - context)
        for a, b in pairwise(chain(
            [LineSpan(0 - context, 0 - context)],
            selected_row_spans,
            [LineSpan(last_row_of_view + context, last_row_of_view + context)]
        ))
        if (a_end := a.end + 1) or True
        if (b.start - a_end) > 2 * context
        # if (print(a, b) or True)
    )
    # print("line_spans_to_fold", line_spans_to_fold)
    regions_to_fold = list(
        sublime.Region(
            view.text_point(start, 0) - (1 if context == 0 else 0),
            view.text_point(end, 0) - 1
        )
        for start, end in line_spans_to_fold
    )
    # print("regions_to_fold", regions_to_fold)
    view.run_command("unfold_all")
    view.fold(list(regions_to_fold))
    view.show(view.sel())


class outline_next_symbol2(sublime_plugin.WindowCommand):
    def run(self):
        window = self.window
        for view in visible_views(window):
            if view.settings().get("outline_mode"):
                view.run_command("outline_next_symbol")
                return


class outline_prev_symbol2(sublime_plugin.WindowCommand):
    def run(self):
        window = self.window
        for view in visible_views(window):
            if view.settings().get("outline_mode"):
                view.run_command("outline_prev_symbol")
                return


def visible_views(window: sublime.Window) -> Iterator[sublime.View]:
    num_groups = window.num_groups()
    for group_id in range(num_groups):
        if (view := window.active_view_in_group(group_id)):
            yield view


class outline_next_symbol(sublime_plugin.TextCommand):
    def run(self, edit):
        # type: (sublime.Edit) -> None
        view = self.view
        frozen_sel = [s for s in view.sel()]
        sel = frozen_sel[0]
        pt = sel.end()
        folded_regions = view.folded_regions()
        for s in symbol_regions(view):
            if any(s.region in region for region in folded_regions):
                continue
            if s.region.begin() > pt:
                break
        else:
            return
        set_sel(view, [flip_region(s.region)])
        view.show(s.region)


class outline_prev_symbol(sublime_plugin.TextCommand):
    def run(self, edit):
        # type: (sublime.Edit) -> None
        view = self.view
        frozen_sel = [s for s in view.sel()]
        sel = frozen_sel[0]
        caret = sel.begin()
        folded_regions = view.folded_regions()
        for s in reversed(symbol_regions(view)):
            if any(s.region in region for region in folded_regions):
                continue
            if s.region.end() < caret:
                break
        else:
            return
        set_sel(view, [flip_region(s.region)])
        view.show(s.region)


class outline_enter_search(sublime_plugin.TextCommand):
    def run(self, edit: sublime.Edit, initial_text: str = "") -> None:
        view = self.view
        original_view_state = view_state(view)
        folded_regions = view.folded_regions()
        if not folded_regions:
            return
        window = view.window()
        if not window:
            return

        def restore_initial_state():
            view.run_command("fixed_unfold_all")
            view.fold(folded_regions)
            apply_view_state(view, original_view_state)
            view.add_regions("matched_chars", [])

        def on_done(term: str) -> None:
            view.add_regions("matched_chars", [])
            view.run_command("exit_outline_mode")

        def on_change(term: str) -> None:
            if not term:
                restore_initial_state()
                return

            symbols = symbol_regions(view)
            lines = [
                TextRange.from_region(view, view.line(s.region))
                for s in symbols
            ]
            matches = fuzzyfind(term, lines)
            focus_regions(view, [
                line.region
                for line, positions in matches
            ])

            regions_per_line: list[list[sublime.Region]] = [
                reduce_regions(
                    sublime.Region(p + line.region.a, p + line.region.a + 1)
                    for p in sorted(positions)
                )
                for line, positions in matches
                if positions
            ]

            view.add_regions(
                "matched_chars",
                list(flatten(regions_per_line)),
                scope="region.bluish",
                flags=(
                    64
                    | sublime.RegionFlags.DRAW_NO_FILL
                    | sublime.RegionFlags.NO_UNDO
                ),
            )

            if regions_per_line:
                best_match = regions_per_line[0]
                set_sel(view, [sublime.Region(best_match[0].a, best_match[-1].b)])

        def on_cancel() -> None:
            if view.settings().get("outline_mode") and view.folded_regions():
                view.add_regions("matched_chars", [])
                view.run_command("abort_outline_mode")
            # restore_initial_state()

        panel = window.show_input_panel(
            "Search",
            initial_text,
            on_done=on_done,
            on_change=on_change,
            on_cancel=on_cancel,
        )
        panel.settings().set("outline_mode_search_panel", True)


class outline_apply_view_state(sublime_plugin.TextCommand):
    def run(self, edit: sublime.Edit, state: ViewState) -> None:
        view = self.view
        apply_view_state(view, state)


def fuzzyfind(primer: str, collection: Iterable[TextRange]) -> list[tuple[TextRange, list[int]]]:
    """
    Fuzzy match a primer, e.g. a search term, against the items in collection.
    """
    suggestions = []
    for item in collection:
        if score := fuzzy_score(primer, item.text):
            suggestions.append((score, item))

    # print("\n-", primer)
    # for score, item in sorted(suggestions):
    #     print("score", score, item.text)
    return [(item, positions) for (_, positions), item in sorted(suggestions)]


def fuzzy_score(primer: str, item: str) -> tuple[float, list[int]] | None:
    pos, score = -1, 0.0
    item_l = item.lower()
    primer_l = primer.lower()
    positions: list[int] = []
    for idx in range(len(primer)):
        try:
            pos, _score = find_char(primer_l[idx:], item, item_l, pos + 1)
        except ValueError:
            return None

        positions.append(pos)
        score += 2 * _score
        if pos == 0:
            score -= 1
        if _score <= 0 and primer[idx] == item[pos]:
            score -= 0.5 if primer[idx] == primer_l[idx] else 2

        if score > 10:
            return None

    return (score, positions)


def find_char(primer_rest, item, item_l, start: int) -> tuple[int, float]:
    prev = ''
    first_seen = -1
    needle = primer_rest[0]
    for idx, ch in enumerate(item[start:], start):
        if needle == ch.lower():
            # consecutive letters/matches don't get a penalty
            if idx == start:
                return start, 0
            # start at a word is a bonus
            if start == 0 and prev == " ":
                return idx, -1
            # uppercase letters count as word boundaries
            if ch.isupper() and (not prev or prev.islower()):
                return idx, 0
            # so do these word boundaries
            if prev in "-_":
                return idx, 0
            if first_seen == -1:
                first_seen = idx
        prev = ch

    # try a *reverse* lookup if needle can't be found
    if first_seen == -1:
        if (
            start > 1
            and (pos := item_l.rfind(needle, 0, start - 1)) != -1
        ):
            return pos, start - 1 - pos
        raise ValueError(f"can't match {primer_rest!r} with {item!r}")
    # a jump to a complete suffix match does not get the full penalty
    if item.endswith(primer_rest):
        return len(item) - len(primer_rest), 1
    # typically the penalty is the distance we have to jump
    return first_seen, first_seen - (start - 1)


def flip_region(region: sublime.Region) -> sublime.Region:
    return sublime.Region(region.b, region.a)


def set_sel(view: sublime.View, selection: Sequence[sublime.Region | sublime.Point]) -> None:
    sel = view.sel()
    sel.clear()
    sel.add_all(selection)


def flash(view: sublime.View, message: str):
    window = view.window()
    if window:
        window.status_message(message)


def reduce_regions(regions: Iterable[sublime.Region]) -> list[sublime.Region]:
    prev = None
    rv = []
    for r in regions:
        if prev is None or prev.b != r.a:
            rv.append(r)
            prev = r
        else:
            prev.b = r.b
    return rv


def pairwise(iterable: Iterable[T]) -> Iterable[tuple[T, T]]:
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def symbol_regions(view: sublime.View) -> list[sublime.SymbolRegion]:
    return _symbol_regions(view, view.change_count())


@lru_cache(maxsize=16)
def _symbol_regions(view: sublime.View, _cc: int) -> list[sublime.SymbolRegion]:
    return view.symbol_regions()
