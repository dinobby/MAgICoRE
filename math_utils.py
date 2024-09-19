import re
import json
import signal
from collections import Counter
from sympy import sympify, simplify
from latex2sympy2 import latex2sympy

def floatify(num):
    try:
        num = float(num)
        return num
    except Exception as e:
        return None
        
def within_eps(pred, gt):
    pred = floatify(pred)
    gt = floatify(gt)
    if pred is None or gt is None:
        return False
    eps = abs(gt-pred)
    if eps < 0.01:
        return True
    else:
        return False
        
def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    string = str(string)
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        # assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def parse_math_boxed(s):
    if not s:
        return "N/A"
    s = last_boxed_only_string(s)
    s = remove_boxed(s)
    return s
    
def get_latex_value(expr):
    # get the actual numeric value of latex expression
    # handle cases like \sqrt{66}  => 8.12
    # handle cases like 6\sqrt{2}  => 8.49
    # handle cases like 2 \sqrt{5} => 4.47
    # handle cases like 3\pi       => 9.42
    
    try:
        if '\\sqrt' in expr and expr.split("\\sqrt")[0]:
            multi_str = expr.split("\\sqrt")[0]
            multiplier = float(multi_str.strip())
            multiplicand = "\\" + expr.split(multi_str+"\\")[-1]
            value = multiplier * float(latex2sympy(multiplicand).evalf())
        else:    
            value = float(latex2sympy(expr).evalf())
    except:
        return expr
    return value


def get_fraction_value(expr):
    # get the actual numeric value of fractions
    # handle cases like \frac{1}{4}  => 0.25
    # handle cases like \frac{\pi}{3}  => 1.05
    # handle cases like \\frac{3}{4}  => 0.75
    # handle cases like \dfrac{1}{4}  => 0.25
    # handle cases like \tfrac{1}{4}  => 0.25
    # does not handle \frac{\sqrt{a}}{b}
    
    try:
        expr = expr.replace('\pi', '3.1416')
        frac_pattern = r"(\\\\d*t*frac|\\d*t*frac|rac)\{([^}]+)\}\{([^}]+)\}"
        def replace_frac(match):
            _, num, den = match.groups()
            return f"{num}/{den}"
        return float(eval(re.sub(frac_pattern, replace_frac, expr)))
    except Exception:
        return expr

def normalize_fraction_notation(expr):
    """
    Normalizes different fraction notations (\\frac, \frac, rac) into a consistent format.

    :param expr: A string containing the expression with fraction notations.
    :return: A string with the normalized fraction format.
    """
    # Regular expression to find different fraction notations
    frac_pattern = r"(\\\\d*t*frac|\\d*t*frac|rac)\{([^}]+)\}\{([^}]+)\}"

    # Function to replace the fraction notations with (numerator)/(denominator)
    def replace_frac(match):
        _, num, den = match.groups()
        return f"({num})/({den})"

    return re.sub(frac_pattern, replace_frac, expr)


def is_string_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2


def is_frac_equiv(expr1, expr2):
    """
    Determines whether two mathematical expressions, possibly containing different fraction notations,
    are equivalent.

    :param expr1: A string representing the first mathematical expression.
    :param expr2: A string representing the second mathematical expression.
    :return: True if the expressions are equivalent, False otherwise.
    """
    try:
        # Normalize fraction notations
        expr1_sympy = normalize_fraction_notation(expr1)
        expr2_sympy = normalize_fraction_notation(expr2)

        # Convert the string expressions into sympy expressions
        sympy_expr1 = sympify(expr1_sympy)
        sympy_expr2 = sympify(expr2_sympy)

        # Simplify both expressions and check for equality
        return simplify(sympy_expr1 - sympy_expr2) == 0
    except Exception:
        
        return False

def math_check1(pred, gt):

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)  # Set the alarm for timeout_duration seconds

    try:
        if is_frac_equiv(pred, gt):
            return True
        if is_string_equiv(pred, gt):
            return True
        numeric_gt_value = get_fraction_value(gt)
        numeric_pred_value = get_fraction_value(pred)
        if within_eps(pred, numeric_gt_value):
            return True
        if within_eps(numeric_pred_value, numeric_gt_value):
            return True            
        numeric_gt_value = get_latex_value(gt)
        numeric_pred_value = get_latex_value(pred)
        if within_eps(pred, numeric_gt_value):
            return True  
        if within_eps(numeric_pred_value, numeric_gt_value):
            return True  
    except TimeoutException:
        return False
    finally:
        signal.alarm(0)
    return False

def math_check2(pred, gt):
    if math_check1(_remove_right_units(pred), _remove_right_units(gt)):
        return True
    return False

def is_math_correct(pred, gt):
    if math_check1(pred, gt):
        return True
    if math_check2(pred, gt):
        return True
    return False

# Define a timeout exception class
class TimeoutException(Exception):
    pass

# Timeout handler function
def timeout_handler(signum, frame):
    raise TimeoutException
    
def evaluate_math(results):
    num_correct = 0
    for i in results:
        if is_math_correct(i['pred'], i['gold_answer']):
            num_correct += 1
    acc = round((num_correct / len(results)), 4)
    return acc