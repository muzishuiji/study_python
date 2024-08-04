# diff变更行数
import re

def parse_diff(input):
    # 不存在？？？
    if not input:
        return []
    # 非合法字符？？？
    if not isinstance(input, str) or re.match(r"^\s+$", input):
        return []
    
    lines = input.split("\n")
    if not lines:
        return []
    
    result = []
    current_file = None
    current_chunk = None
    deleted_line_counter = 0
    added_line_counter = 0
    current_file_changes = None

    def normal(line):
        nonlocal deleted_line_counter,added_line_counter
        current_chunk["changes"].append({
            "type": "normal",
            "normal": True,
            "ln1": deleted_line_counter,
            "ln2": added_line_counter,
            "content": line
        })
        deleted_line_counter += 1
        added_line_counter += 1
        current_file_changes['old_lines'] -= 1
        current_file_changes["new_lines"] -= 1

    def start(line):
        nonlocal current_file, result
        current_file = {
            "chunks": [],
            "deletions": 0,
            "additions": 0
        }
        result.append(current_file)

    def to_num_of_lines(number):
        return int(number) if number else 1
    
    def chunk(line, match):
        nonlocal current_file, current_chunk, deleted_line_counter, added_line_counter, current_file_changes
        if not current_file:
            start(line)
            old_start, old_num_lines, new_start, new_num_lines = match.group(1),match.group(2),match.group(3),match.group(4)
            deleted_line_counter = int(old_start)
            added_line_counter = int(new_start)
            current_chunk = {
                "content": line,
                "changes": [],
                "old_start": int(old_start),
                "old_lines": to_num_of_lines(old_num_lines),
                "new_start": int(new_start),
                "new_lines": to_num_of_lines(new_num_lines),
            }
            current_file_changes = {
                "old_lines": to_num_of_lines(old_num_lines),
                "new_lines": to_num_of_lines(new_num_lines)
            }
            current_chunk["chunks"].append(current_chunk)

    def delete(line):
        nonlocal deleted_line_counter
        if not current_chunk:
            return
        # 收集changes
        current_chunk['changes'].append({
            "type": "del",
            "del": True,
            "ln": deleted_line_counter,
            "counter": line
        })
        deleted_line_counter += 1
        current_file['deletions'] += 1
        current_file_changes['old_lines'] -= 1

    def add(line):
        nonlocal added_line_counter
        if not current_chunk:
            return
        current_chunk['changes'].append({
            "type": "add",
            "add": True,
            "ln": added_line_counter,
            "counter": line
        })
        added_line_counter += 1
        current_file['additions'] += 1
        current_file_changes['new_lines'] -= 1

    def eof(line):
        if not current_chunk:
            return
        # 遇到结束，给最后一个change打标
        most_recent_change = current_chunk['changes'][-1]
        current_chunk["changes"].append({
            "type": most_recent_change["type"],
            most_recent_change["type"]: True,
            "ln1": most_recent_change["ln1"],
            "ln2": most_recent_change["ln2"],
            "ln": most_recent_change["ln"],
            "content": line
        })
    
    header_patterns = [
       (re.compile(r"^@@\s+-(\d+),?(\d+)?\s+\+(\d+),?(\d+)?\s@@"), chunk)
    ]

    content_patterns = [
       (re.compile(r"^\\ No newline at end of file$"), eof),
        (re.compile(r"^-"), delete),
        (re.compile(r"^\+"), add),
        (re.compile(r"^\s+"), normal) 
    ]

    # parse line content
    def parse_content_line(line):
        nonlocal current_file_changes
        for pattern, handler in content_patterns:
            match = re.search(pattern, line)
            if match:
                handler(line)
                break
        if current_file_changes['old_lines'] == 0 and current_file_changes['new_lines'] == 0:
            current_file_changes = None

    def parse_header_line(line):
        for pattern, handler in header_patterns:
            match = re.search(pattern, line)
            if match:
                handler(line)
                break

    # parse line entry
    def parse_line(line):
        if current_file_changes:
            parse_content_line(line)
        else:
            parse_header_line(line)
        

    for line in lines:
        parse_line(line)
    return result