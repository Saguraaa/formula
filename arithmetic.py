import ast
import re
import unicodedata
from fractions import Fraction


def judge(text):
    """Evaluate arithmetic without eval; preserve wrong answers rather than correcting OCR."""
    normalized = ''.join(unicodedata.normalize('NFKC', text).split())
    normalized = normalized.translate(str.maketrans({'×': '*', '∗': '*', '÷': '/', '−': '-', '–': '-'}))
    # Only remove commas from a complete, correctly grouped numeric token.
    # In particular, never turn a decimal comma such as 1,5 into 15.
    normalized = re.sub(r'[0-9.,]+', lambda m: m[0].replace(',', '')
                        if re.fullmatch(r'[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?', m[0])
                        else m[0], normalized)
    result = {'raw_text': text, 'normalized_text': normalized, 'status': 'uncertain', 'is_correct': None}
    if not normalized or normalized.endswith('='):
        return dict(result, reason='empty_or_unanswered')
    if len(normalized) > 200:
        return dict(result, reason='expression_too_long')
    unsupported = sorted(set(re.findall(r'[^0-9.+*/()=\-]', normalized)))
    if unsupported:
        return dict(result, reason='unsupported_characters', unsupported_characters=unsupported)

    def value(node, expression):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            token = ast.get_source_segment(expression, node)
            if not re.fullmatch(r'(?:\d+(?:\.\d*)?|\.\d+)', token):
                raise ValueError('unsupported_number')
            return Fraction(token)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = value(node.operand, expression)
            return v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp):
            a, b = value(node.left, expression), value(node.right, expression)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, ast.Div): return a / b
        raise ValueError('unsupported_expression')

    try:
        parts = normalized.split('=')
        if len(parts) != 2 or not all(parts):
            raise ValueError('expected_one_equals_sign')
        values = [value(ast.parse(p, mode='eval').body, p) for p in parts]
        correct = values[0] == values[1]
        return dict(result, status='correct' if correct else 'incorrect', is_correct=correct,
                    left_value=str(values[0]), right_value=str(values[1]))
    except (ValueError, SyntaxError, ZeroDivisionError, RecursionError) as exc:
        return dict(result, reason=type(exc).__name__)
