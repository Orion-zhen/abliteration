
import sys
import shutil

class Output:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    @staticmethod
    def _colorize(text, color):
        if sys.stdout.isatty():
            return f"{color}{text}{Output.RESET}"
        return text

    @staticmethod
    def header(title: str):
        width = shutil.get_terminal_size().columns
        print("\n" + Output._colorize(f" {title.upper()} ", Output.BOLD + Output.BLUE).center(width, "="))

    @staticmethod
    def subheader(title: str):
        print(f"\n{Output._colorize(title, Output.BOLD)}")

    @staticmethod
    def info(msg: str):
        print(f"{Output._colorize('ℹ', Output.BLUE)} {msg}")

    @staticmethod
    def success(msg: str):
        print(f"{Output._colorize('✔', Output.GREEN)} {msg}")

    @staticmethod
    def warning(msg: str):
        print(f"{Output._colorize('⚠', Output.YELLOW)} {msg}")

    @staticmethod
    def error(msg: str):
        print(f"{Output._colorize('✖', Output.RED)} {msg}")
    
    @staticmethod
    def key_value(key: str, value: str, indent: int = 2):
        print(f"{' ' * indent}{Output._colorize(key + ':', Output.BOLD)} {value}")

    @staticmethod
    def table(rows: list[dict], headers: list[str] = None):
        """
        Prints a list of dictionaries as a simple table.
        """
        if not rows:
            return
        
        if not headers:
            headers = list(rows[0].keys())
        
        # Calculate column widths
        widths = {h: len(h) for h in headers}
        for row in rows:
            for h in headers:
                val = str(row.get(h, ""))
                widths[h] = max(widths[h], len(val))
        
        # Print Header
        header_row = "  ".join(h.ljust(widths[h]) for h in headers)
        print(Output._colorize(header_row, Output.BOLD))
        print("  ".join("-" * widths[h] for h in headers))
        
        # Print Rows
        for row in rows:
            print("  ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers))
