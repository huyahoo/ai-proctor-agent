# src/core/logger.py
import colorama
from colorama import Fore, Style, Back

# Initialize colorama to work on all platforms
colorama.init(autoreset=True)

class ColorLogger:
    @staticmethod
    def info(message: str):
        """Prints an informational message in cyan."""
        print(f"{Fore.CYAN}{message}{Style.RESET_ALL}")

    @staticmethod
    def success(message: str):
        """Prints a success message in green."""
        print(f"{Fore.GREEN}{Style.BRIGHT}{message}{Style.RESET_ALL}")

    @staticmethod
    def warning(message: str):
        """Prints a warning message in yellow."""
        print(f"{Fore.YELLOW}{message}{Style.RESET_ALL}")

    @staticmethod
    def error(message: str):
        """Prints an error message in red."""
        print(f"{Fore.RED}{Style.BRIGHT}{message}{Style.RESET_ALL}")

    @staticmethod
    def debug(message: str):
        """Prints a debug message in a dimmer style."""
        print(f"{Style.DIM}{message}{Style.RESET_ALL}")

    @staticmethod
    def header(message: str):
        """Prints a prominent header message."""
        print(f"\n{Fore.MAGENTA}{Style.BRIGHT}--- {message} ---{Style.RESET_ALL}")
        
    @staticmethod
    def step(message: str):
        """Prints a step message."""
        print(f"\n{Fore.WHITE}{Style.BRIGHT}{message}{Style.RESET_ALL}")

# Create a default logger instance to be imported by other modules
logger = ColorLogger()

# --- Test Script for this module ---
if __name__ == '__main__':
    print("--- Testing ColorLogger ---")
    logger.info("This is an informational message.")
    logger.success("This is a success message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.debug("This is a debug message.")
    logger.header("This is a header.")
    logger.step("This is a step message.")
    print("--- Test Complete ---")