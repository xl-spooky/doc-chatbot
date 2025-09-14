import sys

from app.ui import App

# Prevent Python from writing .pyc files (__pycache__) during imports
sys.dont_write_bytecode = True


if __name__ == "__main__":
    app = App()
    app.mainloop()
