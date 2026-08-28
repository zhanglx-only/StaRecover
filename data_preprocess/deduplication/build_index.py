import argparse
import json
import re


class CodeFeaturizer:
    def __init__(self):
        # Support numeric suffixes such as u and LL.
        self.re_const = re.compile(r'0x[0-9a-fA-F]+|\b\d{4,}\b')
        self.re_keywords = re.compile(r'\b(if|for|while|switch|case|goto|return|break|continue|default)\b')
        self.re_operators = re.compile(r'(<<|>>|==|!=|&&|\|\||[+\-*/%&|^])')

        # Capture function calls.
        self.re_calls = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')

        self.re_function_name = re.compile(r'([a-zA-Z_$?][a-zA-Z0-9_$?@]*)\s*$')

        # Clean strings and comments before extracting index features. String
        # matching must come first so // and /* ... */ inside strings are kept.
        self.re_ignored_code = re.compile(
            r'"(?:\\.|[^"\\])*"|/\*.*?\*/|//[^\r\n]*',
            flags=re.DOTALL,
        )

    def _clean_code_for_features(self, code):
        """Remove comments and clear string contents while preserving newlines."""
        def replace_ignored(match):
            text = match.group(0)
            if text.startswith('"'):
                return '""'
            if text.startswith('/*'):
                # Preserve newlines so code around a comment is not joined.
                return ''.join(char for char in text if char in '\r\n')
            return ''

        return self.re_ignored_code.sub(replace_ignored, code)

    @staticmethod
    def _find_matching_parenthesis(text, open_index):
        """Return the right parenthesis matching the one at open_index."""
        depth = 0
        for index in range(open_index, len(text)):
            char = text[index]
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
                if depth == 0:
                    return index
        return -1

    @staticmethod
    def _find_parameter_open(header):
        """Find the function-argument parenthesis outside C++ template parameters."""
        angle_depth = 0
        for index, char in enumerate(header):
            if char == '<':
                angle_depth += 1
            elif char == '>':
                angle_depth = max(angle_depth - 1, 0)
            elif char == '(' and angle_depth == 0:
                return index
        return -1

    @staticmethod
    def _count_parameters(args_str):
        """Count only top-level commas, ignoring nested function-pointer structures."""
        args_str = args_str.strip()
        if not args_str or args_str.lower() == 'void':
            return 0

        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        angle_depth = 0
        comma_count = 0

        for char in args_str:
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth = max(paren_depth - 1, 0)
            elif char == '[':
                bracket_depth += 1
            elif char == ']':
                bracket_depth = max(bracket_depth - 1, 0)
            elif char == '{':
                brace_depth += 1
            elif char == '}':
                brace_depth = max(brace_depth - 1, 0)
            elif char == '<':
                angle_depth += 1
            elif char == '>':
                angle_depth = max(angle_depth - 1, 0)
            elif (
                char == ','
                and paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
                and angle_depth == 0
            ):
                comma_count += 1

        return comma_count + 1

    def _parse_function_header(self, code, function_name=None):
        """Extract parameter count and return type from a complete declaration."""
        header = code.lstrip()
        if not header:
            return 0, "", -1

        name_start = -1
        open_index = -1

        # If the current function name is available, use it to locate the
        # parameter list and avoid parentheses in return types, templates, and
        # anonymous namespaces.
        if isinstance(function_name, str) and function_name:
            name_start = header.find(function_name)
            if name_start != -1:
                search_start = name_start + len(function_name)
                register_annotation = re.match(r'\s*@<[^>]+>', header[search_start:])
                if register_annotation:
                    search_start += register_annotation.end()
                open_index = header.find('(', search_start)

        # Fall back to ordinary C declaration parsing when no name is available.
        if open_index == -1:
            open_index = self._find_parameter_open(header)
            if open_index == -1:
                return 0, "", -1

        close_index = self._find_matching_parenthesis(header, open_index)
        if close_index == -1:
            return 0, "", -1

        if name_start != -1:
            return_type = header[:name_start].strip()
        else:
            prefix = header[:open_index].rstrip()
            # IDA __usercall declarations may append register annotations such
            # as @<rax> after the function name.
            prefix = re.sub(r'@<[^>]+>\s*$', '', prefix).rstrip()
            convention_match = re.match(
                r'^(.*?\b__(?:fastcall|cdecl|stdcall|thiscall|vectorcall|usercall)\b)',
                prefix,
                flags=re.DOTALL,
            )
            if convention_match:
                return_type = convention_match.group(1).strip()
            else:
                function_match = self.re_function_name.search(prefix)
                if not function_match:
                    return 0, "", close_index
                return_type = prefix[:function_match.start()].strip()

        return_type = re.sub(r'\s+', ' ', return_type).lower()
        args_str = header[open_index + 1:close_index]
        arg_count = self._count_parameters(args_str)
        return arg_count, return_type, close_index

    @staticmethod
    def _count_effective_lines(code):
        """Count non-empty code lines that are not just braces."""
        return sum(
            1
            for line in code.splitlines()
            if line.strip() and line.strip() not in {'{', '}'}
        )

    @classmethod
    def _count_body_lines(cls, code, parameter_close_index):
        """Count effective body lines, excluding multiline declarations and braces."""
        code = code.lstrip()
        body_open_index = code.find('{', parameter_close_index + 1)
        if body_open_index == -1:
            return 0
        return cls._count_effective_lines(code[body_open_index + 1:])

    def get_features(self, code, function_name=None):
        """
        Extract code features:
        - arg_count: number of function parameters
        - return_type: return type
        - loc: effective body lines
        - full_loc: effective lines in the complete function, used as an
          additional retrieval filter
        """
        if not isinstance(code, str) or not code.strip():
            return {
                'arg_count': 0,
                'return_type': '',
                'loc': 0,
                'full_loc': 0,
            }

        clean_code = self._clean_code_for_features(code)
        arg_count, return_type, parameter_close_index = self._parse_function_header(
            clean_code, function_name
        )
        # Keep the current definition: count body lines, not declaration lines.
        loc = self._count_body_lines(clean_code, parameter_close_index)
        full_loc = self._count_effective_lines(clean_code)

        return {
            'arg_count': arg_count,
            'return_type': return_type,
            'loc': loc,
            'full_loc': full_loc,
        }


def process_json_file(input_file_path, output_file_path):
    # Load the JSON file.
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Create a CodeFeaturizer instance.
    featurizer = CodeFeaturizer()

    # Extract features from every code record.
    for entry in data:
        code = entry.get("code", "")
        features = featurizer.get_features(code, entry.get("funname"))

        # Add the index and feature data to the record.
        entry["index"] = {
            'arg_count': features['arg_count'],
            'return_type': features['return_type'],
            'loc': features['loc'],
            'full_loc': features['full_loc'],
        }

    # Save the results to the output file.
    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        json.dump(data, f_out, indent=4, ensure_ascii=False)

    print(f"[+] Processing complete; feature data saved to {output_file_path}")


# Program entry point.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Build a RAG retrieval index using the current LOC definition.'
    )
    parser.add_argument(
        'input_file_path',
        help='Input JSON file',
    )
    parser.add_argument(
        'output_file_path',
        help='Output JSON file',
    )
    args = parser.parse_args()

    process_json_file(args.input_file_path, args.output_file_path)
