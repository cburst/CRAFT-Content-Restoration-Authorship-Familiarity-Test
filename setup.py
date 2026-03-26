from setuptools import setup

APP = ['launch.py']

OPTIONS = {
    'argv_emulation': False,
    'iconfile': 'icon.icns',   # your icon
    'packages': [],
    'includes': [],
}

setup(
    app=APP,
    name="CRAFT",
    options={'py2app': OPTIONS},
)