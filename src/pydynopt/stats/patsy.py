"""
This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

import re
from typing import Iterable, Optional

from patsy.desc import ModelDesc, Term

from pydynopt.pandas import anything_to_dataframe
from pydynopt.utils import anything_to_list


def patsy_formula_to_varnames(*formulas: str) -> list[str]:
    """
    Extract unique list of variable names from patsy formulas.

    Parameters
    ----------
    *formulas : str

    Returns
    -------
    list of str
    """

    varnames = dict()

    def find_names(s: str) -> list:
        names = []

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
            if re.match(r'[*/:+-=!]+', token):
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
                    varnames.update({name: None for name in names})

    varnames = list(varnames.keys())

    return varnames


def patsy_formula_to_categorical_varnames(*formulas: str) -> list[str]:
    """
    Extract unique list of categorical variable names (i.e., variable
    names surrounded by C()) from patsy formulas.

    Parameters
    ----------
    *formulas : str

    Returns
    -------
    list of str
    """

    varnames = dict()

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

    varnames = list(varnames.keys())

    return varnames


def patsy_add_levels(formula: str, data) -> tuple[str | None, list[str]]:
    """
    Add levels information to categorical variables based on categorical
    values present in the data.

    Parameters
    ----------
    formula : str
    data :
        DataFrame or something that can be turned into one.

    Returns
    -------
    formula_upd: str
        Update formula with added factor levels
    factors : list
        Name of factors found in formula
    """

    if not formula:
        return formula, []

    df = anything_to_dataframe(data)

    cache = dict()

    mdesc = ModelDesc.from_formula(formula)

    # Check whether term w/o factors is in term list which corresponds to
    # intercept
    has_intercept = Term([]) in mdesc.rhs_termlist

    def add_levels(termlist) -> tuple[str, list[str]]:

        factors_found = []

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

                name = m.group("name")
                factors_found.append(name)

                if name in df.columns:
                    if name in cache:
                        values = cache[name]
                    else:
                        values = df[name].unique()
                        values.sort()
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


def patsy_strip_categorical(terms: str | Iterable[str]) -> str | list[str]:
    """
    Strip additional meta-data such as Treatment() and levels from categorical
    variable definitions in a patsy formula.

    Parameters
    ----------
    terms : str or Iterable of str

    Returns
    -------
    str or list of str
    """

    terms_list = anything_to_list(terms)

    pattern = re.compile(r'.*C\(.*')
    pattern_cat = re.compile(r'C\((?P<inner>.+)\)(?P<suffix>.*)?')
    cleaned = []
    for label in terms_list:
        if not (m := pattern.match(label)):
            cleaned.append(label)
            continue

        factors = [s.strip() for s in label.split(':')]
        tokens = []
        for factor in factors:
            if not (m := pattern_cat.match(factor)):
                tokens.append(factor)
                continue

            inner = m.group('inner')
            suffix = m.group('suffix')
            name = re.match(r'(?P<name>[^,)]+)', inner).group('name')

            lbl = f'C({name}){suffix}'

            tokens.append(lbl)

        cleaned.append(':'.join(tokens))

    if isinstance(terms, str):
        cleaned = cleaned[0]

    return cleaned


def patsy_strip_formula(formula: Optional[str]) -> str | None:
    """
    Strip formulas of redudant white space.

    Parameters
    ----------
    formula : str

    Returns
    -------
    str or None
    """
    if not formula:
        return formula

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
