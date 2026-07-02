"""
Patsy formula helper functions.

This module provides utility functions to manipulate and inspect patsy formula
strings and extract variable/categorical names and base level treatments.

Features:
- Extract variable names and categorical variable names from formulas.
- Identify categorical base level treatments.
- Update formulas with factor levels from data.
- Strip metadata from categorical formulas.
- Clean up redundant whitespace in formulas.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/.

Author: Richard Foltyn
"""

from collections.abc import Iterable, Sequence
import re
from typing import Any, overload

from patsy.desc import ModelDesc, Term

from pydynopt.pandas import anything_to_dataframe
from pydynopt.utils import anything_to_list


def patsy_formula_to_varnames(*formulas: str) -> list[str]:
    """
    Extract unique list of variable names from patsy formulas.

    Parameters
    ----------
    formulas
        Patsy formula strings to process.

    Returns
    -------
    List of unique variable names.
    """
    varnames: dict[str, None] = {}

    def find_names(s: str) -> list[str]:
        names: list[str] = []

        s = s.strip()

        if m := re.match(r'[^(]+\(([^(]+)\)', s):
            names.extend(find_names(m.group(1)))
            return names

        while (ifrom := s.find('(')) != -1:
            popen = 1
            for i, char in enumerate(s[ifrom + 1 :]):
                if char == '(':
                    popen += 1
                elif char == ')':
                    popen -= 1
                    if popen == 0:
                        subs = s[ifrom + 1 : ifrom + 1 + i]
                        if subs.strip():
                            names.extend(find_names(subs))
                        s = s[:ifrom] + s[ifrom + i + 2 :]
                        break

        tokens = s.split()
        for token in tokens:
            if re.match(r'[*/:+-=!<>]+', token):
                # Ignore operators
                continue
            else:
                try:
                    # Try whether this can be interpreted as a number
                    float(token)
                except ValueError:
                    names.append(token)

        return names

    for formula in formulas:
        if not formula:
            continue
        mdesc = ModelDesc.from_formula(formula)
        terms = mdesc.lhs_termlist + mdesc.rhs_termlist

        for term in terms:
            for factor in term.factors:
                expr = factor.name()

                if m := re.match(r'C\((?P<name>[^,)]+)', expr):
                    # Categorical variable probably cannot have any additional
                    # tokens other than variable name and options.
                    varnames[m.group('name')] = None
                    continue
                else:
                    if m := re.match(r'I\((.+)\)', expr):
                        # Extract content of I(); presumably, I() cannot be
                        # nested.
                        expr = m.group(1)
                    names = find_names(expr)
                    varnames.update(dict.fromkeys(names))

    return list(varnames.keys())


def patsy_formula_to_categorical_varnames(*formulas: str) -> list[str]:
    """
    Extract unique list of categorical variable names from patsy formulas.

    Categorical variable names are those surrounded by C() in the formula.

    Parameters
    ----------
    formulas
        Patsy formula strings to process.

    Returns
    -------
    List of unique categorical variable names.
    """
    varnames: dict[str, None] = {}

    for formula in formulas:
        if not formula:
            continue
        mdesc = ModelDesc.from_formula(formula)
        terms = mdesc.lhs_termlist + mdesc.rhs_termlist

        for term in terms:
            for factor in term.factors:
                expr = factor.name()

                if m := re.match(r'C\((?P<name>[^,)]+)', expr):
                    # Categorical variable probably cannot have any additional
                    # tokens other than variable name and options.
                    varnames[m.group('name')] = None

    return list(varnames.keys())


def patsy_formula_to_categorical_treatments(*formulas: str) -> dict[str, str]:
    """
    Extract unique dictionary of categorical variable names and their treatments.

    Extracts variable names surrounded by C() and their treatment (i.e., base level)
    from patsy formulas. Categorical variables which do not have a Treatment()
    term are ignored.

    Parameters
    ----------
    formulas
        Patsy formula strings to process.

    Returns
    -------
    Mapping from categorical variable names to their base levels.
    """
    treatments: dict[str, str] = {}

    for formula in formulas:
        if not formula:
            continue
        mdesc = ModelDesc.from_formula(formula)
        terms = mdesc.lhs_termlist + mdesc.rhs_termlist

        for term in terms:
            for factor in term.factors:
                expr = factor.name().strip()

                # Categorical variable probably cannot have any additional
                # tokens other than variable name and options.
                if m := re.match(r'C\((?P<name>[^,)]+)(?P<rest>.*)\)', expr):
                    var = m.group('name')
                    # Extract treatment spec from the remainder
                    if mt := re.match(r'.*Treatment\((.+)\).*', m.group('rest')):
                        value = mt.group(1)
                        treatments[var] = value

    return treatments


def patsy_add_levels(formula: str, data: Any) -> tuple[str, list[str]]:
    """
    Add levels information to categorical variables based on values in the data.

    Parameters
    ----------
    formula
        Patsy formula string.
    data
        DataFrame or data structure that can be converted to one.

    Returns
    -------
    formula_upd
        Updated formula with added factor levels.
    factors
        Names of factors found in the formula.
    """
    if not formula:
        return formula, []

    df = anything_to_dataframe(data)

    cache: dict[str, Any] = {}

    mdesc = ModelDesc.from_formula(formula)

    # Check whether term w/o factors is in term list which corresponds to
    # intercept
    has_intercept = Term([]) in mdesc.rhs_termlist

    def add_levels(termlist: Sequence[Term]) -> tuple[str, list[str]]:

        factors_found: list[str] = []

        for term in termlist:
            for factor in term.factors:
                if not factor:
                    continue

                code = factor.code

                if not (m := re.match(r'C\((?P<name>[^,)]+)', code)):
                    continue

                # levels already present, no updating needed
                if re.match(r'.*levels=.*', code, re.IGNORECASE):
                    continue

                name = m.group('name')
                factors_found.append(name)

                if name in df.columns:
                    if name in cache:
                        values = cache[name]
                    else:
                        values = sorted(df[name].unique())
                        cache[name] = values

                    code = code.strip()
                    code = code[: len(code) - 1]
                    code += ', levels=[' + ','.join(str(v) for v in values) + '])'

                factor.code = code

        tokens = [
            ':'.join(factor.code for factor in term.factors if factor)
            for term in termlist
            if term.factors
        ]

        frml = ' + '.join(tokens)
        return frml, factors_found

    formula_upd, factors = add_levels(mdesc.rhs_termlist)

    if not has_intercept:
        formula_upd += ' -1'
    else:
        formula_upd = ' + '.join(token for token in ('1', formula_upd) if token)

    if mdesc.lhs_termlist:
        formula_lhs, factors_lhs = add_levels(mdesc.lhs_termlist)
        formula_upd = ' ~ '.join((formula_lhs, formula_upd))
        factors.extend(factors_lhs)

    factors = list(dict.fromkeys(factors).keys())

    return formula_upd, factors


@overload
def patsy_strip_categorical(terms: str) -> str: ...  # pyright: ignore[reportOverlappingOverload]


@overload
def patsy_strip_categorical(terms: Iterable[str]) -> list[str]: ...


def patsy_strip_categorical(terms: str | Iterable[str]) -> str | list[str]:
    """
    Strip categorical metadata from variable definitions in a patsy formula.

    Removes additional metadata such as Treatment() and levels from categorical
    variable definitions.

    Parameters
    ----------
    terms
        Patsy terms or iterable of terms to strip.

    Returns
    -------
    Patsy terms with categorical metadata removed.
    """
    terms_list = anything_to_list(terms, force=True)

    pattern = re.compile(r'.*C\(.*')
    pattern_cat = re.compile(r'C\((?P<inner>.+)\)(?P<suffix>.*)?')
    cleaned: list[str] = []
    for label in terms_list:
        if not (m := pattern.match(label)):
            cleaned.append(label)
            continue

        factors: list[str] = [s.strip() for s in label.split(':')]
        tokens: list[str] = []
        for factor in factors:
            if not (m := pattern_cat.match(factor)):
                tokens.append(factor)
                continue

            inner = m.group('inner')
            suffix = m.group('suffix') or ''
            name_match = re.match(r'(?P<name>[^,)]+)', inner)
            name = name_match.group('name') if name_match else ''

            lbl = f'C({name}){suffix}'

            tokens.append(lbl)

        cleaned.append(':'.join(tokens))

    if isinstance(terms, str):
        return cleaned[0]

    return cleaned


def patsy_strip_formula(formula: str | None) -> str:
    """
    Strip formulas of redundant white space.

    Parameters
    ----------
    formula
        Patsy formula string.

    Returns
    -------
    Formula string with redundant white space removed.
    """
    if not formula:
        return ''

    # Get rid of multiple consecutive white space characters
    formula = ' '.join(formula.strip().split())

    # Make sure some operators are surrounded by spaces. Process only single instance
    # of operators, not **
    ops = ['+', '*', '~']
    for op in ops:
        eop = re.escape(op)
        pattern = re.compile(rf'\s*(?<!{eop}){eop}(?!{eop})\s*')
        formula = pattern.sub(f' {op} ', formula)

    return formula
