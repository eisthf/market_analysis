from dataclasses import dataclass
import numpy as np
from typing import ClassVar
from inspect import Signature, Parameter
import pandas as pd
import base64
import io
import streamlit as st


def auto_label(self):
    cname = self.__class__.__qualname__
    signature = Signature.from_callable(self.__init__)
    args, keyword_only = [], False

    for p in signature.parameters.values():
        v = getattr(self, p.name, p.default)

        if p.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            raise ValueError(f"Unsupported parameter type {p.kind}")

        if p.kind == Parameter.KEYWORD_ONLY:
            keyword_only = True
        elif isinstance(p.default, (type(None), str, bool)):
            keyword_only = True

        if v == p.default:
            # skip argument if not equal to default
            if keyword_only or not isinstance(v, (int, float)):
                keyword_only = True
                continue

        if keyword_only:
            args.append(f"{p.name}={v!r}")
        else:
            args.append(f"{v!r}")

    args = ", ".join(args)

    return f"{cname}({args})"


def calc_envelope(prices, period: int = 20, pct: float = 10.0):
    middle = prices["close"].rolling(period).mean()
    upper = middle * (1 + pct/100)
    lower = middle * (1 - pct/100)

    result = dict(upperband=upper, middleband=middle, lowerband=lower)
    result = pd.DataFrame(result)
    return result



def series_xy(data, item=None, dropna=False):
    """split series into x, y arrays"""

    if item is not None:
        data = data[item]

    if dropna:
        data = data.dropna()

    x = data.index.values
    y = data.values

    return x, y


class Indicator:
    """Injects a basic __repr__ based on __init__ signature"""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__str__ = auto_label



class ENVELOPE(Indicator):
    same_scale: ClassVar[bool] = True
    COLOR: ClassVar[str] = "orange"

    def __init__(self, period: int, pct: float):
        self.period = period
        self.pct = pct

    def __call__(self, prices):
        return calc_envelope(prices, self.period, self.pct)

    def plot_result(self, data, chart, ax=None):
        if ax is None:
            ax = chart.get_axes("samex")

        label = str(self)

        upper = data.iloc[:, 0]
        middle = data.iloc[:, 1]
        lower = data.iloc[:, 2]

        color = self.COLOR

        xs, ms = series_xy(middle)
        ax.plot(xs, ms, color=color, linestyle="dashed", label=label)

        xs, hs = series_xy(upper)
        ax.plot(xs, hs, color=color, linestyle="dotted")

        xs, ls = series_xy(lower)
        ax.plot(xs, ls, color=color, linestyle="dotted")

        ax.fill_between(xs, ls, hs, color=color, interpolate=True, alpha=0.2)


def svg_write(fig, center=True):
    """
    Renders a matplotlib figure object to SVG.
    Disable center to left-margin align like other objects.
    """
    # Save to stringIO instead of file
    imgdata = io.StringIO()
    fig.savefig(imgdata, format="svg")

    # Retrieve saved string
    imgdata.seek(0)
    svg_string = imgdata.getvalue()

    # Encode as base 64
    b64 = base64.b64encode(svg_string.encode("utf-8")).decode("utf-8")

    # Add some CSS on top
    css_justify = "center" if center else "left"
    css = '<p style="text-align:center; display: flex; justify-content: {};">'.format(css_justify)
    html = r'{}<img src="data:image/svg+xml;base64,{}"/>'.format(
        css, b64
    )
    # Write the HTML
    st.write(html, unsafe_allow_html=True)