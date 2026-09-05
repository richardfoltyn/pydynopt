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

import ast
from collections.abc import Iterable, Sequence
import math
import re
from typing import Any, overload

import numpy as np
import pandas as pd
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
                parsed = ast.parse(expr, mode='eval')
                categorical_call = parsed.body

                if not (
                    isinstance(categorical_call, ast.Call)
                    and isinstance(categorical_call.func, ast.Name)
                    and categorical_call.func.id == 'C'
                ):
                    continue

                if categorical_call.args:
                    variable_node = categorical_call.args[0]
                else:
                    variable_node = next(
                        (
                            keyword.value
                            for keyword in categorical_call.keywords
                            if keyword.arg == 'data'
                        ),
                        None,
                    )
                if variable_node is None:
                    continue

                contrast_node = (
                    categorical_call.args[1]
                    if len(categorical_call.args) > 1
                    else next(
                        (
                            keyword.value
                            for keyword in categorical_call.keywords
                            if keyword.arg == 'contrast'
                        ),
                        None,
                    )
                )
                if not isinstance(contrast_node, ast.Call):
                    continue
                contrast_func = contrast_node.func
                is_treatment = (
                    isinstance(contrast_func, ast.Name)
                    and contrast_func.id == 'Treatment'
                ) or (
                    isinstance(contrast_func, ast.Attribute)
                    and contrast_func.attr == 'Treatment'
                )
                if not is_treatment:
                    continue

                variable = ast.get_source_segment(expr, variable_node)
                if variable is None:
                    continue

                if contrast_node.args:
                    treatment_node = contrast_node.args[0]
                else:
                    treatment_node = next(
                        (
                            keyword.value
                            for keyword in contrast_node.keywords
                            if keyword.arg == 'reference'
                        ),
                        None,
                    )
                if treatment_node is None:
                    raise ValueError(
                        f'Malformed Treatment() call for categorical factor '
                        f'{variable!r} in {expr!r}: expected at least one argument.'
                    )

                treatment = ast.get_source_segment(expr, treatment_node)
                if treatment is None:
                    continue

                treatments[variable] = treatment

    return treatments


def _get_clean_factor_levels(series: pd.Series, factor_name: str) -> list[Any]:
    """
    Get clean, sorted, observed levels of a factor from a Series.

    Parameters
    ----------
    series
        The pandas Series to process.
    factor_name
        The name of the factor for error reporting.

    Returns
    -------
    Sorted list of Python scalar values.
    """
    raw_unique = series.unique()

    clean_values: list[Any] = []
    seen: set[tuple[type, Any]] = set()

    for val in raw_unique:
        if val is None or val is pd.NA:
            continue
        if isinstance(val, (float, np.floating)) and np.isnan(val):
            continue
        try:
            if pd.isna(val):
                continue
        except (TypeError, ValueError):
            pass

        if isinstance(val, np.generic):
            val = val.item()

        if val is None or val is pd.NA:
            continue
        if isinstance(val, (float, np.floating)) and np.isnan(val):
            continue
        try:
            if pd.isna(val):
                continue
        except (TypeError, ValueError):
            pass

        is_safe = False
        if isinstance(val, (bool, int)):
            is_safe = True
        elif isinstance(val, float):
            is_safe = math.isfinite(val)
        elif isinstance(val, str):
            is_safe = True

        if not is_safe:
            raise ValueError(
                f'Factor {factor_name!r} has value {val!r} of type {type(val).__name__} '
                f'which cannot be represented safely in Patsy formula text.'
            )

        val_key = (type(val), val)
        if val_key not in seen:
            seen.add(val_key)
            clean_values.append(val)

    try:
        sorted_values = sorted(clean_values)
    except TypeError as e:
        raise ValueError(
            f'Values for factor {factor_name!r} cannot be sorted deterministically: {clean_values}'
        ) from e

    return sorted_values


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
                        values = _get_clean_factor_levels(df[name], name)
                        cache[name] = values

                    code = code.strip()
                    code = code[: len(code) - 1]
                    code += f', levels={values!r})'

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
